#!/usr/bin/env python3
"""
kalman_filter.py

Applies a Kalman Filter to the merged ROV data, preserving ISO8601 timestamps
(e.g., "2023-11-01T19:00:01Z") in the final CSV.

This version handles heading data separately from the main Kalman filter
to properly account for the circular nature of angular data.

Intended to be executed via the data processing orchestrator which passes the
raw and processed directories (including dive folder information).
"""

from pathlib import Path
import sys
import traceback
import csv
import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from pyproj import Proj
from scipy.ndimage import gaussian_filter1d


def deg2rad(deg):
    """Safely convert degrees to radians, handling NaNs."""
    try:
        return np.deg2rad(float(deg))
    except (ValueError, TypeError):
        return np.nan


def rad2deg_scalar(rad):
    """Safely convert radians to degrees, handling NaNs."""
    try:
        return np.rad2deg(float(rad))
    except (ValueError, TypeError):
        return np.nan


def wrap_angle(angle):
    """Ensure angles remain within [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def filter_heading(headings, window_size=11):
    """
    Filter heading data using a specialized approach for circular quantities.
    Uses a Gaussian-weighted window to properly handle angle wrapping.

    Args:
        headings: Array of heading values in degrees.
        window_size: Size of the filtering window (odd number recommended).

    Returns:
        Array of filtered heading values in degrees [0, 360).
    """
    if len(headings) == 0 or np.all(np.isnan(headings)):
        return np.full_like(headings, np.nan)

    # Convert to radians and then to sine and cosine components
    rad = np.deg2rad(headings)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)

    # Interpolate NaN values
    valid_mask = ~np.isnan(headings)
    if np.any(~valid_mask):
        sin_vals = np.interp(np.arange(len(sin_vals)), np.where(valid_mask)[0], sin_vals[valid_mask])
        cos_vals = np.interp(np.arange(len(cos_vals)), np.where(valid_mask)[0], cos_vals[valid_mask])

    # Apply Gaussian smoothing
    sigma = max(window_size / 5.0, 1.0)
    sin_smooth = gaussian_filter1d(sin_vals, sigma, mode='nearest')
    cos_smooth = gaussian_filter1d(cos_vals, sigma, mode='nearest')

    # Convert back to degrees in [0, 360)
    filtered_rad = np.arctan2(sin_smooth, cos_smooth)
    filtered_deg = np.mod(np.rad2deg(filtered_rad), 360)
    return filtered_deg


def determine_utm_zone(lon, lat):
    """Determine the UTM zone for given longitude and latitude coordinates."""
    zone_number = int((lon + 180) / 6) + 1

    # Special cases for Norway and Svalbard
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone_number = 32
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone_number = 31
        elif 9.0 <= lon < 21.0:
            zone_number = 33
        elif 21.0 <= lon < 33.0:
            zone_number = 35
        elif 33.0 <= lon < 42.0:
            zone_number = 37

    hemisphere = "north" if lat >= 0 else "south"
    return zone_number, hemisphere


def latlon_to_utm(df, lat_col, lon_col, x_col, y_col):
    """Convert latitude/longitude to UTM coordinates dynamically determining UTM zone."""
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    valid_count = valid_mask.sum()
    if valid_count == 0:
        print(f"No valid {lat_col}/{lon_col} coordinates found.")
        return 0, None

    first_valid_row = df[valid_mask].iloc[0]
    first_lat = float(first_valid_row[lat_col])
    first_lon = float(first_valid_row[lon_col])
    zone_number, hemisphere = determine_utm_zone(first_lon, first_lat)

    utm_proj_str = f"+proj=utm +zone={zone_number} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    utm_proj = Proj(utm_proj_str)

    print(f"Using UTM Zone {zone_number}{hemisphere[0].upper()} for {lat_col}/{lon_col} conversion")
    print(f"Converting {valid_count} points from {lat_col}/{lon_col} to UTM...")

    df[x_col] = np.nan
    df[y_col] = np.nan
    success_count = 0
    for idx, row in df[valid_mask].iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            x, y = utm_proj(lon, lat)
            df.at[idx, x_col] = x
            df.at[idx, y_col] = y
            success_count += 1
        except Exception as e:
            print(f"Error converting coordinates ({lat}, {lon}): {e}")

    print(f"Successfully converted {success_count} of {valid_count} points.")
    if success_count > 0:
        sample = df[[lat_col, lon_col, x_col, y_col]].dropna().head(3)
        print("Sample conversions:")
        for _, row in sample.iterrows():
            print(f"  {row[lat_col]}, {row[lon_col]} → {row[x_col]}, {row[y_col]}")

    return success_count, utm_proj


def process_data(raw_dir, processed_dir):
    """
    Processes the merged ROV data by applying a Kalman filter.

    Args:
        raw_dir (Path or str): Directory containing the raw input file.
                                This should include the dive folder information.
        processed_dir (Path or str): Directory where output files will be saved.
    """
    try:
        # Convert input directories to absolute Path objects.
        raw_dir = Path(raw_dir).resolve()
        processed_dir = Path(processed_dir).resolve()

        # Extract expedition and dive from raw_dir.
        # Assumes raw_dir is: <root_dir>/RUMI_processed/<dive>
        expedition = raw_dir.parent.parent.name
        dive = raw_dir.name

        # Setup file paths using provided directories.
        input_file = raw_dir / "kalman_prepped_datamerge.csv"
        output_file = processed_dir / f"{expedition}_{dive}_kalman_filtered_data.csv"

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found at {input_file}")

        print(f"Reading from: {input_file}")
        print(f"Output will be saved to: {output_file}")

        # Read CSV with timestamp parsing.
        df = pd.read_csv(input_file, parse_dates=["Timestamp"])
        print(f"Loaded {len(df)} rows from input file.")

        # Filter rows with depth <= -20 m.
        original_count = len(df)
        df = df[df["Herc_Depth_1"] <= -20]
        print(f"Filtered to {len(df)} rows where depth <= -20m (removed {original_count - len(df)} rows)")
        df.sort_values(by="Timestamp", inplace=True)

        # Remove duplicate timestamps BEFORE Kalman filter processing.
        if df["Timestamp"].duplicated().any():
            print("Duplicate timestamps detected. Removing duplicates (preferring rows with null event_value)...")
            if "event_value" in df.columns:
                df = df.sort_values("event_value", na_position="first").drop_duplicates(subset=["Timestamp"], keep="first")
            else:
                df = df.drop_duplicates(subset=["Timestamp"])
            print(f"After deduplication, {len(df)} rows remain.")

        # Convert to UTM coordinates for USBL and DVL data.
        usbl_success, usbl_proj = latlon_to_utm(df, "Lat_USBL", "Long_USBL", "x_usbl", "y_usbl")
        dvl_success, dvl_proj = latlon_to_utm(df, "Lat_DVL", "Long_DVL", "x_dvl", "y_dvl")
        utm_proj = usbl_proj if usbl_success > 0 else (dvl_proj if dvl_success > 0 else None)
        if usbl_success == 0 and dvl_success == 0:
            print("WARNING: No coordinates could be converted to UTM. Check lat/long data.")
            if len(df) > 0:
                print("\nFirst few rows of lat/long data:")
                print(df[["Lat_USBL", "Long_USBL", "Lat_DVL", "Long_DVL"]].head())

        # Convert orientation to radians.
        df["Heading_rad"] = df["Heading"].apply(deg2rad)
        df["Pitch_rad"] = df["Pitch"].apply(deg2rad)
        df["Roll_rad"] = df["Roll"].apply(deg2rad)

        # Process heading separately.
        print("Processing heading data with specialized circular filter...")
        if "Heading" in df.columns:
            df["kalman_yaw_deg"] = filter_heading(df["Heading"].values, window_size=15)
            print(f"Filtered {len(df)} heading values")
        else:
            print("WARNING: No heading data found")
            df["kalman_yaw_deg"] = np.nan

        # Initialize an 8D Kalman Filter (excluding yaw, which is handled separately).
        kf = KalmanFilter(dim_x=8, dim_z=1)
        init_x = df["x_usbl"].dropna().iloc[0] if not df["x_usbl"].dropna().empty else 0.0
        init_y = df["y_usbl"].dropna().iloc[0] if not df["y_usbl"].dropna().empty else 0.0
        init_z = df["Herc_Depth_1"].dropna().iloc[0] if not df["Herc_Depth_1"].dropna().empty else 0.0
        init_roll = df["Roll_rad"].dropna().iloc[0] if not df["Roll_rad"].dropna().empty else 0.0
        init_pitch = df["Pitch_rad"].dropna().iloc[0] if not df["Pitch_rad"].dropna().empty else 0.0

        print(f"Initial state: x={init_x}, y={init_y}, z={init_z}")
        kf.x = np.array([init_x, init_y, init_z, init_roll, init_pitch, 0.0, 0.0, 0.0], dtype=float)
        kf.Q = np.diag([
            0.3 ** 2, 0.3 ** 2, 0.3 ** 2,  # Position noise
            0.01 ** 2, 0.01 ** 2,           # Orientation noise (roll, pitch)
            0.05 ** 2, 0.05 ** 2, 0.05 ** 2  # Velocity noise
        ])
        kf.P = np.diag([
            1000, 1000, 1000,
            (math.radians(20)) ** 2,
            (math.radians(20)) ** 2,
            100, 100, 100
        ])

        # Prepare columns for Kalman filter outputs.
        df["kalman_x"] = np.nan
        df["kalman_y"] = np.nan
        df["kalman_lat"] = np.nan
        df["kalman_long"] = np.nan
        df["kalman_depth"] = np.nan
        df["kalman_roll_deg"] = np.nan
        df["kalman_pitch_deg"] = np.nan

        prev_time = None
        recent_usbl_x = []
        recent_usbl_y = []
        if utm_proj is None:
            print("Warning: No UTM projection could be determined. Using default UTM Zone 4N.")
            utm_proj = Proj("+proj=utm +zone=4 +datum=WGS84 +units=m +no_defs")

        print("Starting Kalman filter processing...")
        updates_applied = 0
        for i, row in df.iterrows():
            current_time = row["Timestamp"]
            dt = 1.0 if prev_time is None else max((current_time - prev_time).total_seconds(), 0.001)
            prev_time = current_time

            # Build state transition matrix.
            F = np.eye(8)
            F[0, 5] = dt
            F[1, 6] = dt
            F[2, 7] = dt
            kf.F = F
            kf.predict()

            # Depth update.
            if not np.isnan(row.get("Herc_Depth_1", np.nan)):
                H_depth = np.zeros((1, 8))
                H_depth[0, 2] = 1.0
                R_depth = np.array([[0.1 ** 2]])
                kf.update(row["Herc_Depth_1"], H=H_depth, R=R_depth)
                updates_applied += 1

            # USBL update with an outlier check.
            if not np.isnan(row.get("x_usbl", np.nan)) and not np.isnan(row.get("y_usbl", np.nan)):
                recent_usbl_x.append(row["x_usbl"])
                recent_usbl_y.append(row["y_usbl"])
                if len(recent_usbl_x) > 20:
                    recent_usbl_x.pop(0)
                    recent_usbl_y.pop(0)
                if len(recent_usbl_x) >= 2:
                    mean_x = np.mean(recent_usbl_x)
                    mean_y = np.mean(recent_usbl_y)
                    std_x = np.std(recent_usbl_x)
                    std_y = np.std(recent_usbl_y)
                    if (std_x > 0 and std_y > 0 and
                            abs(row["x_usbl"] - mean_x) <= 3 * std_x and
                            abs(row["y_usbl"] - mean_y) <= 3 * std_y):
                        H_x = np.zeros((1, 8))
                        H_x[0, 0] = 1.0
                        R_x = np.array([[((row["Accuracy_USBL"] if not np.isnan(row.get("Accuracy_USBL", np.nan))
                                            else 5.0) ** 2)]])
                        kf.update(np.array([row["x_usbl"]]), H=H_x, R=R_x)
                        updates_applied += 1
                        H_y = np.zeros((1, 8))
                        H_y[0, 1] = 1.0
                        R_y = np.array([[((row["Accuracy_USBL"] if not np.isnan(row.get("Accuracy_USBL", np.nan))
                                            else 5.0) ** 2)]])
                        kf.update(np.array([row["y_usbl"]]), H=H_y, R=R_y)
                        updates_applied += 1
                else:
                    H_x = np.zeros((1, 8))
                    H_x[0, 0] = 1.0
                    R_x = np.array([[((row["Accuracy_USBL"] if not np.isnan(row.get("Accuracy_USBL", np.nan))
                                        else 5.0) ** 2)]])
                    kf.update(np.array([row["x_usbl"]]), H=H_x, R=R_x)
                    updates_applied += 1
                    H_y = np.zeros((1, 8))
                    H_y[0, 1] = 1.0
                    R_y = np.array([[((row["Accuracy_USBL"] if not np.isnan(row.get("Accuracy_USBL", np.nan))
                                        else 5.0) ** 2)]])
                    kf.update(np.array([row["y_usbl"]]), H=H_y, R=R_y)
                    updates_applied += 1

            # DVL update if depth >= -30.
            if not np.isnan(row.get("x_dvl", np.nan)) and not np.isnan(row.get("y_dvl", np.nan)):
                if row["Herc_Depth_1"] >= -30:
                    H_x_dvl = np.zeros((1, 8))
                    H_x_dvl[0, 0] = 1.0
                    R_x_dvl = np.array([[3.0 ** 2]])
                    kf.update(np.array([row["x_dvl"]]), H=H_x_dvl, R=R_x_dvl)
                    updates_applied += 1
                    H_y_dvl = np.zeros((1, 8))
                    H_y_dvl[0, 1] = 1.0
                    R_y_dvl = np.array([[3.0 ** 2]])
                    kf.update(np.array([row["y_dvl"]]), H=H_y_dvl, R=R_y_dvl)
                    updates_applied += 1

            # Orientation updates.
            if not np.isnan(row.get("Roll_rad", np.nan)):
                H_roll = np.zeros((1, 8))
                H_roll[0, 3] = 1.0
                R_roll = np.array([[0.017 ** 2]])
                kf.update(np.array([row["Roll_rad"]]), H=H_roll, R=R_roll)
                updates_applied += 1

            if not np.isnan(row.get("Pitch_rad", np.nan)):
                H_pitch = np.zeros((1, 8))
                H_pitch[0, 4] = 1.0
                R_pitch = np.array([[0.017 ** 2]])
                kf.update(np.array([row["Pitch_rad"]]), H=H_pitch, R=R_pitch)
                updates_applied += 1

            # Wrap orientation angles.
            kf.x[3] = wrap_angle(kf.x[3])
            kf.x[4] = wrap_angle(kf.x[4])

            # Save filtered state.
            x_est, y_est, z_est, r_est, p_est, vx_est, vy_est, vz_est = kf.x
            df.at[i, "kalman_x"] = x_est
            df.at[i, "kalman_y"] = y_est
            try:
                lon_est, lat_est = utm_proj(x_est, y_est, inverse=True)
                df.at[i, "kalman_lat"] = lat_est
                df.at[i, "kalman_long"] = lon_est
            except Exception as e:
                print(f"Error converting UTM back to lat/long for ({x_est}, {y_est}): {e}")
            df.at[i, "kalman_depth"] = z_est
            df.at[i, "kalman_roll_deg"] = np.degrees(r_est)
            df.at[i, "kalman_pitch_deg"] = np.degrees(p_est)

        print(f"Kalman filter processing complete. Applied {updates_applied} updates.")

        # Convert Timestamp back to ISO8601.
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).apply(
            lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        # Ensure the processed directory exists.
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save filtered data.
        try:
            df.to_csv(output_file, index=False, mode='w')
            print(f"\nSaved Kalman-filtered data to {output_file}")
        except PermissionError:
            local_output = Path.cwd() / "kalman_filtered_data.csv"
            df.to_csv(local_output, index=False, mode='w')
            print(f"\nPermission denied on original path. Saved output to {local_output}")

        # Build and save the final datatable.
        final_columns = [
            "Timestamp", "Vehicle", "x_usbl", "y_usbl", "x_dvl", "y_dvl",
            "Heading_rad", "Pitch_rad", "Roll_rad", "kalman_yaw_deg",
            "kalman_x", "kalman_y", "kalman_lat", "kalman_long", "kalman_depth",
            "kalman_roll_deg", "kalman_pitch_deg", "O2_Concentration", "O2_Saturation",
            "Temperature", "Conductivity", "Pressure", "Salinity", "Sound_Velocity",
            "event_value", "event_free_text", "event_option.channel", "event_option.milestone",
            "event_option.rating", "event_option.vehicle",
            "vehicleRealtimeDualHDGrabData.camera_name_2_uom", "vehicleRealtimeDualHDGrabData.camera_name_2_value",
            "vehicleRealtimeDualHDGrabData.camera_name_uom", "vehicleRealtimeDualHDGrabData.camera_name_value",
            "vehicleRealtimeDualHDGrabData.filename_2_uom", "vehicleRealtimeDualHDGrabData.filename_2_value",
            "vehicleRealtimeDualHDGrabData.filename_uom", "vehicleRealtimeDualHDGrabData.filename_value"
        ]
        for col in final_columns:
            if col not in df.columns:
                df[col] = np.nan

        final_df = df[final_columns]
        final_output_file = processed_dir / "NA156_H2024_final_datatable.csv"
        final_df.to_csv(final_output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved final datatable to {final_output_file}")

        print("Processing complete. All UTM and Kalman-filtered data included in output.")
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    # For testing purposes, if run directly, allow optional command-line arguments.
    # Otherwise, default to current directory as raw_dir and a "processed" subdirectory.
    if len(sys.argv) >= 3:
        raw_directory = Path(sys.argv[1])
        processed_directory = Path(sys.argv[2])
    else:
        raw_directory = Path.cwd().resolve()
        processed_directory = raw_directory / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)
    exit_code = process_data(raw_directory, processed_directory)
    sys.exit(exit_code)
