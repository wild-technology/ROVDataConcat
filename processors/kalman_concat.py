from pathlib import Path
import pandas as pd
import csv

def remove_duplicate_timestamps_prioritizing_event(df):
    """
    Removes duplicate rows in df based on the 'Timestamp' column.
    For each group of rows sharing the same Timestamp, it prioritizes keeping
    the row where the 'event_value' column is not null (if available).

    Parameters:
      df (pd.DataFrame): The merged DataFrame with a 'Timestamp' column.

    Returns:
      new_df (pd.DataFrame): DataFrame with duplicates removed.
      duplicate_counts (dict): Dictionary with Timestamp as key and number of duplicates removed.
      total_removed (int): Total number of rows removed.
    """
    unique_rows = []
    duplicate_counts = {}
    # Group by Timestamp (as datetime)
    for ts, group in df.groupby("Timestamp"):
        count = len(group)
        if count > 1:
            duplicate_counts[ts] = count - 1
        # If 'event_value' exists, prioritize rows where event_value is not null
        if 'event_value' in group.columns:
            non_null = group[group['event_value'].notnull()]
            if not non_null.empty:
                chosen = non_null.iloc[0]
            else:
                chosen = group.iloc[0]
        else:
            chosen = group.iloc[0]
        unique_rows.append(chosen)
    new_df = pd.DataFrame(unique_rows)
    total_removed = df.shape[0] - new_df.shape[0]
    return new_df, duplicate_counts, total_removed

def process_data(raw_dir, processed_dir):
    """
    Merges various navigation data sources for ROV navigation, applies an outlier filter for
    pitch and roll values based on mean and standard deviation, and removes duplicate timestamps
    prioritizing non-null event_value rows.

    Parameters
    ----------
    raw_dir : Path or str
        Directory containing the input files. This path should already include the dive folder.
    processed_dir : Path or str
        Directory where the output file will be saved.
    """
    # Convert to Path objects and resolve to absolute paths.
    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()

    print("Running Kalman Concat Process...")

    usbl_file = processed_dir / "NA156_H2024_USBL_Hercules.csv"
    octans_file = processed_dir / "NA156_H2024_pitch_roll_heading_octans.csv"
    dvl_file = processed_dir / "NA156_H2024_dvl_lat_long.csv"
    depth_file = processed_dir / "NA156_H2024_sealog_sensors_merged.csv"
    output_file = processed_dir / "kalman_prepped_datamerge.csv"

    # Read each file with Timestamp parsing
    octans_df = pd.read_csv(octans_file, parse_dates=["Timestamp"]).sort_values("Timestamp")
    usbl_df = pd.read_csv(usbl_file, parse_dates=["Timestamp"]).sort_values("Timestamp")
    dvl_df = pd.read_csv(dvl_file, parse_dates=["Timestamp"]).sort_values("Timestamp")
    depth_df = pd.read_csv(depth_file, parse_dates=["Timestamp"], quotechar='"').sort_values("Timestamp")

    # Rename columns for clarity
    usbl_df.rename(columns={"Latitude": "Lat_USBL", "Longitude": "Long_USBL", "Accuracy": "Accuracy_USBL"},
                   inplace=True)
    dvl_df.rename(columns={"Latitude": "Lat_DVL", "Longitude": "Long_DVL"}, inplace=True)

    # Merge all datasets using outer join on Timestamp
    merged_df = pd.merge(octans_df, usbl_df, on="Timestamp", how="outer")
    merged_df = pd.merge(merged_df, dvl_df, on="Timestamp", how="outer")
    merged_df = pd.merge(merged_df, depth_df, on="Timestamp", how="outer")
    merged_df.sort_values("Timestamp", inplace=True)

    print(f"Merged dataframe shape before filtering: {merged_df.shape}")

    # Identify pitch and roll columns in a case-insensitive way
    pitch_col = next((col for col in merged_df.columns if col.lower() == "pitch"), None)
    roll_col = next((col for col in merged_df.columns if col.lower() == "roll"), None)

    # Initialize outlier counts per column
    pitch_outlier_count = 0
    roll_outlier_count = 0

    if pitch_col or roll_col:
        outlier_mask = pd.Series([False] * len(merged_df))
        threshold = 3  # Using 3 standard deviations as cutoff

        if pitch_col:
            pitch_mean = merged_df[pitch_col].mean()
            pitch_std = merged_df[pitch_col].std()
            pitch_outliers = (merged_df[pitch_col] < pitch_mean - threshold * pitch_std) | \
                             (merged_df[pitch_col] > pitch_mean + threshold * pitch_std)
            pitch_outlier_count = pitch_outliers.sum()
            print(
                f"Pitch ({pitch_col}): mean = {pitch_mean:.4f}, std = {pitch_std:.4f}, outliers = {pitch_outlier_count}")
            outlier_mask |= pitch_outliers
        if roll_col:
            roll_mean = merged_df[roll_col].mean()
            roll_std = merged_df[roll_col].std()
            roll_outliers = (merged_df[roll_col] < roll_mean - threshold * roll_std) | \
                            (merged_df[roll_col] > roll_mean + threshold * roll_std)
            roll_outlier_count = roll_outliers.sum()
            print(f"Roll ({roll_col}): mean = {roll_mean:.4f}, std = {roll_std:.4f}, outliers = {roll_outlier_count}")
            outlier_mask |= roll_outliers

        total_outliers = outlier_mask.sum()
        print(f"Filtering out a total of {total_outliers} outlier rows based on statistical assessment for pitch/roll.")
        merged_df = merged_df[~outlier_mask]
    else:
        print("No pitch/roll columns found for filtering.")

    print(f"Merged dataframe shape after filtering: {merged_df.shape}")

    # Run duplicate timestamp check with priority for non-null event_value
    merged_df, duplicate_counts, total_removed = remove_duplicate_timestamps_prioritizing_event(merged_df)
    print(f"After duplicate removal prioritizing event_value, removed {total_removed} duplicate rows.")
    print(f"Merged dataframe shape after duplicate removal: {merged_df.shape}")

    # Optionally, print a sample of per-Timestamp duplicate counts
    if duplicate_counts:
        print("Duplicate counts per Timestamp (sample):")
        for ts, cnt in list(duplicate_counts.items())[:5]:
            print(f"  {ts}: {cnt} duplicates")
    else:
        print("No duplicate timestamps found.")

    # Convert Timestamp to ISO8601 with seconds only
    merged_df["Timestamp"] = merged_df["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure the processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Export to CSV with quoting enabled
    merged_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"Kalman merged data saved to: {output_file}")

# The __main__ block is omitted so that the main script (which passes raw_dir and processed_dir)
# handles user interaction. For testing purposes you might add an interactive block here.
