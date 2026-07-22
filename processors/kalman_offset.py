#!/usr/bin/env python3
"""
UTM Assessment and Vehicle Offset Processing Script

Performs UTM assessment and adjusts vehicle positioning.
After offsetting the vehicle's (x, y) and initially adjusting depth
based on the sampled center pixel of the GeoTIFF, this function further
checks the 3x3 neighborhood around the offset position. If any neighboring
pixel is within 1m of the vehicle's depth (i.e. if the vehicle is less than
1m above the maximum neighbor value), the depth is increased to be 1m above
that neighbor value.

After this neighbor evaluation, the script re-evaluates that the new depth
is at least 1m above the terrain (center pixel value) and adjusts if necessary.

Finally, the script produces a scatter plot of the offset positions colored by
depth difference (Depth - Terrain) in which any negative depth difference is shown in RED.
It also prints an on-screen summary indicating how many depth values remain below terrain.
"""

from pathlib import Path
import pandas as pd
import csv
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Proj, Transformer

from processors.common import expedition_dive_from_processed_dir, utm_proj_string

# Vehicle position is shifted this many meters backwards along the heading.
OFFSET_M = 2.0
# Adjusted depth must end up at least this many meters above the terrain.
MIN_CLEARANCE_M = 1.0

def safe_get_loc(df, col_name):
    """
    Returns the integer location of the specified column.
    If multiple indices are returned (duplicate column names),
    returns the first index.
    """
    loc = df.columns.get_loc(col_name)
    try:
        return int(loc)
    except TypeError:
        # If loc is array-like, return the first element.
        return int(loc[0])

def process_data(raw_dir, processed_dir):
    """
    Performs UTM assessment and adjusts vehicle positioning.
    After offsetting the vehicle's (x, y) and initially adjusting depth
    based on the sampled center pixel of the GeoTIFF, this function further
    checks the 3x3 neighborhood around the offset position. If any neighboring
    pixel is within 1m of the vehicle's depth (i.e. if the vehicle is less than
    1m above the maximum neighbor value), the depth is increased to be 1m above
    that neighbor value.

    After this neighbor evaluation, the script re-evaluates that the new depth
    is at least 1m above the terrain (center pixel value) and adjusts if necessary.

    Finally, the script produces a scatter plot of the offset positions colored by
    depth difference (Depth - Terrain) in which any negative depth difference is shown in RED.
    It also prints an on-screen summary indicating how many depth values remain below terrain.
    """
    print("Running UTM Assessment Process...")

    # Convert raw_dir and processed_dir to absolute Path objects.
    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()

    expedition, dive = expedition_dive_from_processed_dir(processed_dir)

    # Locate the dive GeoTIFF by pattern; the UTM zone suffix varies by region
    # (e.g. ..._utm4n.tif, ..._utm53n.tif), so don't hard-code it.
    candidates = sorted(raw_dir.glob(f"{dive}_k2mapping_geotiff*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No GeoTIFF matching '{dive}_k2mapping_geotiff*.tif' found in {raw_dir}"
        )
    if len(candidates) > 1:
        print(f"Warning: multiple GeoTIFFs found, using the first: "
              f"{[c.name for c in candidates]}")
    geotiff_path = candidates[0]
    print(f"Using GeoTIFF: {geotiff_path.name}")

    csv_path = processed_dir / f"{expedition}_{dive}_final_datatable.csv"
    output_file = processed_dir / f"{expedition}_{dive}_filtered_offset_final.csv"

    df = pd.read_csv(csv_path)

    # Define required columns.
    x_col, y_col, depth_col = 'kalman_x', 'kalman_y', 'kalman_depth'
    heading_rad_col = 'Heading_rad'  # Raw OCTANS heading in radians (compass convention)

    # -------------------------------------------------------------------------
    # 1) Preserve the original heading before using it for filtering/offset
    # -------------------------------------------------------------------------
    df['_orig_heading_rad'] = df[heading_rad_col].copy()

    # Offset (x, y) OFFSET_M meters backwards along the heading.
    # Heading is a compass bearing (0 = North = +y/northing, 90 = East =
    # +x/easting, clockwise), so the forward unit vector in UTM is
    # (sin(h), cos(h)) and "backwards" subtracts it.
    # (Previous code used (cos(h), sin(h)) -- the math-angle convention -- which
    # pointed the offset in the wrong direction for compass headings.)
    df[x_col] = df[x_col] - (OFFSET_M * np.sin(df[heading_rad_col]))
    df[y_col] = df[y_col] - (OFFSET_M * np.cos(df[heading_rad_col]))

    # Save offset positions for later neighbor evaluation.
    df['Offset_x'] = df[x_col]
    df['Offset_y'] = df[y_col]

    # ------------------------------------------------------------------
    # Rebuild the UTM projection kalman_filter used (derived from the same
    # first valid lat/long, so it selects the same zone), then transform the
    # offset positions into the raster's CRS for sampling. This makes the
    # sampling correct even when the GeoTIFF is in a different UTM zone or
    # datum than the kalman coordinates.
    # ------------------------------------------------------------------
    valid_ll = df[['kalman_lat', 'kalman_long']].dropna()
    if valid_ll.empty:
        raise ValueError("No kalman_lat/kalman_long values in the final datatable; "
                         "cannot determine the vehicle UTM zone.")
    first_lat = float(valid_ll.iloc[0]['kalman_lat'])
    first_lon = float(valid_ll.iloc[0]['kalman_long'])
    vehicle_proj_str = utm_proj_string(first_lon, first_lat)
    vehicle_crs = CRS.from_proj4(vehicle_proj_str)
    vehicle_proj = Proj(vehicle_proj_str)

    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        print(f"Vehicle CRS: {vehicle_crs.to_string()}")
        print(f"GeoTIFF CRS: {raster_crs}")
        if raster_crs is None:
            raise ValueError(f"GeoTIFF {geotiff_path.name} has no CRS; cannot sample safely.")
        to_raster = Transformer.from_crs(vehicle_crs, raster_crs, always_xy=True)
        raster_x, raster_y = to_raster.transform(
            df['Offset_x'].to_numpy(), df['Offset_y'].to_numpy()
        )

        # Center-pixel sample at every offset position. Off-raster and nodata
        # samples become NaN so they impose no depth constraint (previously
        # they sampled as 0 and forced the depth to the sea surface).
        coords = list(zip(raster_x, raster_y))
        sampled = [
            float(val[0]) if not np.ma.is_masked(val[0]) else np.nan
            for val in src.sample(coords, masked=True)
        ]
        df['geotiff_value'] = sampled
        n_nodata = int(np.isnan(sampled).sum())
        if n_nodata:
            print(f"Note: {n_nodata} of {len(sampled)} positions are off-raster or "
                  f"nodata; their depths are left unadjusted.")
        df['below_surface'] = df[depth_col] < df['geotiff_value'] + MIN_CLEARANCE_M
        df.loc[df['below_surface'], depth_col] = df['geotiff_value'] + MIN_CLEARANCE_M

        # Neighbor evaluation: raise depth to MIN_CLEARANCE_M above the highest
        # pixel in the 3x3 window around each offset position.
        from rasterio.windows import Window

        def max_neighbor_value(x, y, window_size=3):
            row_idx, col_idx = src.index(x, y)
            half = window_size // 2
            # Clamp the window to the raster extent (avoids boundless reads,
            # which are buggy in some rasterio versions and slower).
            r0, c0 = max(row_idx - half, 0), max(col_idx - half, 0)
            r1 = min(row_idx + half + 1, src.height)
            c1 = min(col_idx + half + 1, src.width)
            if r1 <= r0 or c1 <= c0:
                return -np.inf  # entirely off-raster: no neighbor constraint
            data = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
            if np.ma.is_masked(data) and data.mask.all():
                return -np.inf  # all nodata: no neighbor constraint
            return float(data.max())

        depths = df[depth_col].to_numpy(copy=True)
        for pos, (x, y) in enumerate(coords):
            max_nb = max_neighbor_value(x, y)
            if depths[pos] - max_nb < MIN_CLEARANCE_M:
                depths[pos] = max_nb + MIN_CLEARANCE_M
        df[depth_col] = depths

    # Re-evaluate: ensure each new depth is at least MIN_CLEARANCE_M above the
    # center pixel's terrain value.
    below_threshold = df[depth_col] < (df['geotiff_value'] + MIN_CLEARANCE_M)
    num_adjustments = below_threshold.sum()
    if num_adjustments > 0:
        df.loc[below_threshold, depth_col] = df.loc[below_threshold, 'geotiff_value'] + MIN_CLEARANCE_M
    print(f"After re-evaluation, adjusted {num_adjustments} depth values to ensure "
          f"they are at least {MIN_CLEARANCE_M}m above terrain.")

    # ---- Visualization & Summary Section ----
    # Compute depth difference: (adjusted Depth) - (center pixel value).
    df['depth_diff'] = df[depth_col] - df['geotiff_value']
    total_rows = len(df)
    num_below = ((df[depth_col] - df['geotiff_value']) < MIN_CLEARANCE_M).sum()
    print(f"Depth Summary: {num_below} out of {total_rows} depth values are "
          f"< {MIN_CLEARANCE_M}m above terrain (should be 0).")

    # Create a scatter plot of offset positions (NaN depth_diff = off-raster,
    # excluded from both groups).
    neg_mask = df['depth_diff'] < 0
    pos_mask = df['depth_diff'] >= 0

    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 8))
    if pos_mask.sum() > 0:
        scatter_ax.scatter(
            df.loc[pos_mask, 'Offset_x'],
            df.loc[pos_mask, 'Offset_y'],
            c=df.loc[pos_mask, 'depth_diff'],
            cmap='viridis',
            s=10,
            label="Depth Diff >= 0"
        )
    if neg_mask.sum() > 0:
        scatter_ax.scatter(
            df.loc[neg_mask, 'Offset_x'],
            df.loc[neg_mask, 'Offset_y'],
            color='red',
            s=10,
            label="Depth Diff < 0"
        )
    scatter_ax.set_xlabel("Offset X")
    scatter_ax.set_ylabel("Offset Y")
    scatter_ax.set_title("Dive Track (Offset Positions) Colored by Depth Difference")
    scatter_ax.legend()
    scatter_path = processed_dir / f"{expedition}_{dive}_dive_track_scatter.png"
    plt.savefig(scatter_path)
    plt.close(scatter_fig)
    print(f"Dive track scatter plot saved to: {scatter_path}")

    # --------------------------------------------------------------------------
    # Carryover corrected UTM converted coordinates.
    # Create new columns for the final output.
    df['UTM_X'] = df['Offset_x']
    df['UTM_Y'] = df['Offset_y']

    # Keep Latitude/Longitude consistent with the offset UTM position
    # (previously the file carried the un-offset kalman lat/long alongside
    # the offset UTM_X/UTM_Y).
    off_lon, off_lat = vehicle_proj(df['Offset_x'].to_numpy(),
                                    df['Offset_y'].to_numpy(), inverse=True)
    df['kalman_lat'] = off_lat
    df['kalman_long'] = off_lon
    # --------------------------------------------------------------------------

    # Rename columns for final CSV.
    rename_map = {
        'kalman_lat': 'Latitude',
        'kalman_long': 'Longitude',
        'kalman_depth': 'Depth',
        # New renames for the requested fields:
        'vehicleRealtimeDualHDGrabData.camera_name_2_value': 'Capture_1',
        'vehicleRealtimeDualHDGrabData.camera_name_value': 'Capture_2',
        'vehicleRealtimeDualHDGrabData.filename_2_value': 'Capture_1_image_path',
        'vehicleRealtimeDualHDGrabData.filename_value': 'Capture_2_image_path'
    }
    df.rename(columns=rename_map, inplace=True)

    # --------------------------------------------------------------------------
    # 2) Do NOT rely on the (filtered) heading for final output.
    #    Instead, convert the *original* heading (which we stored in `_orig_heading_rad`)
    #    to degrees and keep that for the final CSV.
    # --------------------------------------------------------------------------
    df['Heading'] = np.degrees(df['_orig_heading_rad'])
    df['Pitch'] = np.degrees(df['Pitch_rad'])
    df['Roll'] = np.degrees(df['Roll_rad'])

    # Reorder columns: insert Heading, Pitch, Roll immediately to the left of "Depth".
    if 'Depth' in df.columns:
        depth_idx = safe_get_loc(df, 'Depth')
        for col in reversed(['Heading', 'Pitch', 'Roll']):
            series = df.pop(col)
            df.insert(depth_idx, col, series)

    # Drop unnecessary columns.
    drop_cols = [
        'x_usbl', 'y_usbl', 'x_dvl', 'y_dvl',
        'Heading_rad',       # the filtered heading we used; discard from final
        'Pitch_rad', 'Roll_rad',
        'kalman_yaw_deg', 'kalman_roll_deg', 'kalman_pitch_deg',
        'kalman_x', 'kalman_y',
        'geotiff_value', 'below_surface', 'depth_diff',
        '_orig_heading_rad',
        # Requested drops for the *uom fields:
        'vehicleRealtimeDualHDGrabData.camera_name_2_uom',
        'vehicleRealtimeDualHDGrabData.camera_name_uom',
        'vehicleRealtimeDualHDGrabData.filename_2_uom',
        'vehicleRealtimeDualHDGrabData.filename_uom'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Save final CSV with all fields quoted.
    df.to_csv(output_file, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    print(f"UTM assessment results saved to: {output_file}")

if __name__ == "__main__":
    raw_dir = input("Enter the raw directory for processing: ").strip()
    processed_dir = input("Enter the processed directory for output: ").strip()
    process_data(raw_dir, processed_dir)
