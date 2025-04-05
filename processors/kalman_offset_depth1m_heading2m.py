#!/usr/bin/env python3
"""
UTM Assessment and Vehicle Offset Processing Script

Performs UTM assessment and adjusts vehicle positioning.
After offsetting the vehicle’s (x, y) and initially adjusting depth
based on the sampled center pixel of the GeoTIFF, this function further
checks the 3x3 neighborhood around the offset position. If any neighboring
pixel is within 1m of the vehicle’s depth (i.e. if the vehicle is less than
1m above the maximum neighbor value), the depth is increased to be 1m above
that neighbor value.

After this neighbor evaluation, the script re-evaluates that the new depth
is at least 1m above the terrain (center pixel value) and adjusts if necessary.

Finally, the script produces a scatter plot of the offset positions colored by
depth difference (Depth – Terrain) in which any negative depth difference is shown in RED.
It also prints an on-screen summary indicating how many depth values remain below terrain.
"""

from pathlib import Path
import pandas as pd
import csv
import rasterio
import numpy as np
import matplotlib.pyplot as plt

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

def process_data(raw_dir, processed_dir, geotiff_file):
    """
    Performs UTM assessment and adjusts vehicle positioning.
    After offsetting the vehicle’s (x, y) and initially adjusting depth
    based on the sampled center pixel of the GeoTIFF, this function further
    checks the 3x3 neighborhood around the offset position. If any neighboring
    pixel is within 1m of the vehicle’s depth (i.e. if the vehicle is less than
    1m above the maximum neighbor value), the depth is increased to be 1m above
    that neighbor value.

    After this neighbor evaluation, the script re-evaluates that the new depth
    is at least 1m above the terrain (center pixel value) and adjusts if necessary.

    Finally, the script produces a scatter plot of the offset positions colored by
    depth difference (Depth – Terrain) in which any negative depth difference is shown in RED.
    It also prints an on-screen summary indicating how many depth values remain below terrain.

    The output filenames incorporate the dive and expedition information, which are
    derived from the raw_dir path. The GeoTIFF location is provided by the user.
    """
    print("Running UTM Assessment Process...")

    # Convert raw_dir and processed_dir to absolute Path objects.
    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    geotiff_path = Path(geotiff_file).resolve()

    # Extract expedition and dive from raw_dir.
    # Assumes raw_dir is: <root_dir> / "RUMI_processed" / <dive>
    expedition = raw_dir.parent.parent.name
    dive = raw_dir.name

    # Build filenames incorporating the dive and expedition.
    csv_path = processed_dir / f"{expedition}_{dive}_final_datatable.csv"
    output_file = processed_dir / f"{expedition}_{dive}_filtered_offset_final.csv"

    df = pd.read_csv(csv_path)

    # Define required columns.
    x_col, y_col, depth_col = 'kalman_x', 'kalman_y', 'kalman_depth'
    heading_rad_col = 'Heading_rad'  # Filtered heading in radians

    # -------------------------------------------------------------------------
    # 1) Preserve the original heading before using it for filtering/offset
    # -------------------------------------------------------------------------
    df['_orig_heading_rad'] = df[heading_rad_col].copy()

    # Offset (x, y) by 2m backwards along Heading_rad.
    df[x_col] = df[x_col] - (2 * np.cos(df[heading_rad_col]))
    df[y_col] = df[y_col] - (2 * np.sin(df[heading_rad_col]))

    # Save offset positions for later neighbor evaluation.
    df['Offset_x'] = df[x_col]
    df['Offset_y'] = df[y_col]

    # Function to sample the center pixel from the raster.
    def sample_raster_values(raster_path, x_series, y_series):
        coords = list(zip(x_series, y_series))
        with rasterio.open(raster_path) as src:
            sampled = list(src.sample(coords))
        return [val[0] for val in sampled]

    # Resample GeoTIFF at the offset positions (center pixel).
    df['geotiff_value'] = sample_raster_values(geotiff_path, df[x_col], df[y_col])
    df['below_surface'] = df[depth_col] < df['geotiff_value']
    df.loc[df['below_surface'], depth_col] = df['geotiff_value'] + 0.5

    # New function: sample neighboring pixel values from a window.
    def sample_neighbor_values(raster_path, x, y, window_size=3):
        from rasterio.windows import Window
        with rasterio.open(raster_path) as src:
            # Convert geographic coordinate (x, y) to row, col indices.
            row, col = src.index(x, y)
            half = window_size // 2
            window = Window(col - half, row - half, window_size, window_size)
            data = src.read(1, window=window, boundless=True)
        return data.flatten()

    # New function: adjust depth based on neighbor pixel values.
    def adjust_depth_by_neighbors(row, raster_path, window_size=3):
        x = row['Offset_x']
        y = row['Offset_y']
        current_depth = row[depth_col]
        neighbors = sample_neighbor_values(raster_path, x, y, window_size)
        max_neighbor = np.max(neighbors)
        # If current_depth is less than 1m above the maximum neighbor elevation,
        # adjust it to be exactly 1m above.
        if current_depth - max_neighbor < 1:
            current_depth = max_neighbor + 0.5
        return current_depth

    # Apply neighbor evaluation to adjust depth further if needed.
    df[depth_col] = df.apply(
        lambda row: adjust_depth_by_neighbors(row, geotiff_path, window_size=3),
        axis=1
    )

    # Re-evaluate: ensure each new depth is at least 1m above the center pixel's terrain value.
    below_threshold = df[depth_col] < (df['geotiff_value'])
    num_adjustments = below_threshold.sum()
    if num_adjustments > 0:
        df.loc[below_threshold, depth_col] = df.loc[below_threshold, 'geotiff_value'] + 1
    print(f"After re-evaluation, adjusted {num_adjustments} depth values to ensure they are at least 1m above terrain.")

    # ---- Visualization & Summary Section ----
    # Compute depth difference: (adjusted Depth) - (center pixel value).
    df['depth_diff'] = df[depth_col] - df['geotiff_value']
    total_rows = len(df)
    num_below = ((df[depth_col] - df['geotiff_value']) < 1).sum()
    print(f"Depth Summary: {num_below} out of {total_rows} depth values are < 1m above terrain (should be 0).")

    # Create a scatter plot of offset positions.
    neg_mask = df['depth_diff'] < 0
    pos_mask = ~neg_mask

    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 8))
    if pos_mask.sum() > 0:
        scatter_ax.scatter(
            df.loc[pos_mask, 'Offset_x'],
            df.loc[pos_mask, 'Offset_y'],
            c=df.loc[pos_mask, 'depth_diff'],
            cmap='viridis',
            s=10,
            label="Depth Diff ≥ 0"
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
    geotiff_file = input("Enter the full path for the GeoTIFF file: ").strip()
    process_data(raw_dir, processed_dir, geotiff_file)
