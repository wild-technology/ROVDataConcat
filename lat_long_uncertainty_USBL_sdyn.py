from pathlib import Path
import re
import pandas as pd
from datetime import datetime, timedelta, timezone

def parse_sdyn_file(filepath):
    """
    Parses an SDYN file and extracts USBL fix data.

    The filename is expected to follow the format: YYYYMMDD_HHMM.SDYN.
    Each matching line in the file is expected to be a GPGGA sentence with fields for:
      - Time (HHMMSS.SSS)
      - Latitude (degrees and minutes with N/S)
      - Longitude (degrees and minutes with E/W)
      - Accuracy, Depth, and Beacon index (to determine the vehicle).

    The file start time is derived from the filename and is combined with the fix time
    from each line. Latitude and longitude are converted to decimal degrees.

    Parameters:
        filepath (Path or str): The full path to the SDYN file.

    Returns:
        pandas.DataFrame: DataFrame with columns ["Timestamp", "Latitude", "Longitude",
        "Accuracy", "Depth", "Vehicle"].
    """
    filepath = Path(filepath)
    filename = filepath.name
    base = filepath.stem  # Gets the filename without extension

    try:
        file_start = datetime.strptime(base, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Filename {filename} does not match expected format YYYYMMDD_HHMM.SDYN")
        return pd.DataFrame()

    pattern = re.compile(
        r'\$GPGGA,'
        r'(\d+\.\d+),'  # Time (HHMMSS.SSS)
        r'(\d{2})(\d{2}\.\d+),'  # Latitude degrees and minutes
        r'([NS]),'
        r'(\d{3})(\d{2}\.\d+),'  # Longitude degrees and minutes
        r'([EW]),'
        r'\d+,\d+,'  # Skip satellites and fix quality
        r'([\d.]+),'  # Accuracy
        r'([-0-9.]+),'  # Depth
        r'M,0\.0,M,0\.0,'
        r'(\d{4})\*'  # Beacon index
        r'([0-9A-F]+)'
    )

    data = []
    with filepath.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                (time_str, lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir,
                 accuracy, depth, beacon_index, checksum) = match.groups()

                try:
                    fix_time = datetime.strptime(time_str, "%H%M%S.%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    fix_time = datetime.strptime(time_str, "%H%M%S").replace(tzinfo=timezone.utc)

                full_timestamp = file_start.replace(
                    hour=fix_time.hour, minute=fix_time.minute,
                    second=fix_time.second, microsecond=fix_time.microsecond
                )
                if full_timestamp < file_start:
                    full_timestamp += timedelta(days=1)

                # Convert timestamp to pandas datetime (microseconds kept for now)
                full_timestamp = pd.to_datetime(full_timestamp.isoformat(), utc=True)

                # Convert latitude and longitude to decimal degrees
                lat = float(lat_deg) + float(lat_min) / 60.0
                if lat_dir.upper() == 'S':
                    lat = -lat
                lon = float(lon_deg) + float(lon_min) / 60.0
                if lon_dir.upper() == 'W':
                    lon = -lon

                beacon_int = int(beacon_index)
                # Skip processing of Atalanta data (beacon index 2)
                if beacon_int == 2:
                    continue
                elif beacon_int == 1:
                    vehicle = "Hercules"
                else:
                    vehicle = "Unknown"

                data.append([full_timestamp, lat, lon, float(accuracy), float(depth), vehicle])

    return pd.DataFrame(data, columns=["Timestamp", "Latitude", "Longitude", "Accuracy", "Depth", "Vehicle"])

def process_all_sdyn_files(root_directory):
    """
    Processes all SDYN files found in <root_directory>/raw/datalog.

    Parameters:
        root_directory (Path or str): The base directory containing the raw data.

    Returns:
        pandas.DataFrame: Combined DataFrame containing data from all SDYN files.
    """
    root_directory = Path(root_directory)
    sdyn_dir = root_directory / "raw" / "datalog"

    if not sdyn_dir.exists():
        raise FileNotFoundError(f"SDYN directory not found at {sdyn_dir}")

    # Use glob to find all files with .SDYN extension (case-insensitive)
    files = list(sdyn_dir.glob("*.SDYN")) + list(sdyn_dir.glob("*.sdyn"))

    dataframes = []
    for filepath in files:
        df = parse_sdyn_file(filepath)
        if not df.empty:
            dataframes.append(df)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()

def preserve_closest_fix_per_second(df):
    """
    For each unique second (truncating microseconds):
      - If only one fix exists in that second, truncate the microseconds.
      - If multiple fixes occur, choose the row with the best (lowest) accuracy.

    The final Timestamp is forced to be a string in ISO8601 format (UTC) without subseconds.

    Parameters:
        df (pandas.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pandas.DataFrame: Deduplicated DataFrame with Timestamp as a string.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["truncated"] = df["Timestamp"].apply(lambda dt: dt.replace(microsecond=0))
    chosen_rows = []
    for truncated_val, group in df.groupby("truncated"):
        # Choose the row with the lowest Accuracy value
        best_idx = group["Accuracy"].idxmin()
        best_row = group.loc[best_idx].copy()
        best_row["Timestamp"] = truncated_val.strftime("%Y-%m-%dT%H:%M:%SZ")
        chosen_rows.append(best_row)

    df_final = pd.DataFrame(chosen_rows)
    if "truncated" in df_final.columns:
        df_final.drop(columns=["truncated"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    df_final["Timestamp"] = df_final["Timestamp"].apply(
        lambda t: t if isinstance(t, str) else t.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return df_final

def remove_timestamp_duplicates(df):
    """
    Removes rows with duplicate timestamps by keeping only the row with the best (lowest) accuracy.

    Parameters:
        df (pandas.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pandas.DataFrame: DataFrame with duplicate timestamps removed.
        int: Number of duplicate rows removed.
    """
    if df.empty:
        return df.copy(), 0

    before_count = len(df)
    # Sort by Accuracy so that the best (lowest) value appears first,
    # then drop duplicates based on Timestamp.
    df_no_dupes = df.sort_values("Accuracy").drop_duplicates(subset=["Timestamp"], keep="first")
    removed_count = before_count - len(df_no_dupes)

    return df_no_dupes, removed_count

def process_dive_vehicle(dive_summary, sdyn_data):
    """
    Filters and processes SDYN data for a given dive.

    Uses the dive summary (which must contain 'dive', 'Launch Time', and 'Recovery Time')
    to filter the USBL fixes in sdyn_data to those within the dive's time window.
    For each vehicle (other than "Unknown"), duplicate fixes are culled.

    Parameters:
        dive_summary (pandas.Series): A row from the dive summaries DataFrame.
        sdyn_data (pandas.DataFrame): DataFrame containing parsed SDYN data.

    Returns:
        dict: A dictionary with keys as vehicle names and values as DataFrames of processed fixes.
    """
    dive_id = str(dive_summary["dive"]).strip()
    launch_time = dive_summary["Launch Time"]
    recovery_time = dive_summary["Recovery Time"]

    sdyn_data["Timestamp"] = pd.to_datetime(sdyn_data["Timestamp"], utc=True)
    df_dive = sdyn_data[(sdyn_data["Timestamp"] >= launch_time) & (sdyn_data["Timestamp"] <= recovery_time)]
    if df_dive.empty:
        print(f"No USBL fixes for dive {dive_id}.")
        return {}

    processed = {}
    for vehicle in df_dive["Vehicle"].unique():
        if vehicle == "Unknown":
            continue
        df_vehicle = df_dive[df_dive["Vehicle"] == vehicle].copy()
        df_vehicle = preserve_closest_fix_per_second(df_vehicle)

        # Remove any remaining duplicate timestamps by selecting the best accuracy.
        df_vehicle, dupes_removed = remove_timestamp_duplicates(df_vehicle)
        if dupes_removed > 0:
            print(f"Removed {dupes_removed} additional duplicate timestamps for dive {dive_id}, vehicle {vehicle}")

        processed[vehicle] = df_vehicle

    return processed

def process_data(root_directory):
    """
    Main processing function for USBL data.

    Reads dive summaries from <root_directory>/RUMI_processed/all_dive_summaries.csv,
    processes SDYN files from <root_directory>/raw/datalog,
    and for each dive, filters USBL fixes within the dive's time window,
    applies duplicate culling based on best accuracy, and saves processed data as CSV files.

    Processed CSV files are saved under <root_directory>/RUMI_processed/<dive_id>/,
    with filenames formatted as: <expedition>_<dive_id>_USBL_<vehicle>.csv.

    Parameters:
        root_directory (Path or str): The base directory containing raw data and processed data.
    """
    root_directory = Path(root_directory)
    processed_dir = root_directory / "RUMI_processed"
    summary_path = processed_dir / "all_dive_summaries.csv"

    if not summary_path.exists():
        print(f"Error: Dive summary file not found at {summary_path}")
        return

    try:
        dive_summaries = pd.read_csv(summary_path)
        dive_summaries["Launch Time"] = pd.to_datetime(dive_summaries["Launch Time"], utc=True, errors="coerce")
        dive_summaries["Recovery Time"] = pd.to_datetime(dive_summaries["Recovery Time"], utc=True, errors="coerce")
    except Exception as e:
        print(f"Error reading dive summaries from {summary_path}: {e}")
        return

    try:
        sdyn_data = process_all_sdyn_files(root_directory)
    except Exception as e:
        print(f"Error processing SDYN files: {e}")
        return

    if sdyn_data.empty:
        print("No USBL fixes found.")
        return

    # Process each dive and save output files
    for _, dive_row in dive_summaries.iterrows():
        dive_id = str(dive_row["dive"]).strip()
        expedition = str(dive_row.get("expedition", "NA")).strip()
        processed = process_dive_vehicle(dive_row, sdyn_data)
        if processed:
            dive_out_dir = processed_dir / dive_id
            dive_out_dir.mkdir(parents=True, exist_ok=True)

            for vehicle, df_vehicle in processed.items():
                # Final check for any duplicate timestamps based on best accuracy
                df_final, final_dupes_removed = remove_timestamp_duplicates(df_vehicle)
                if final_dupes_removed > 0:
                    print(
                        f"Removed {final_dupes_removed} final duplicate timestamps for dive {dive_id}, vehicle {vehicle}")

                fname = f"{expedition}_{dive_id}_USBL_{vehicle}.csv"
                outpath = dive_out_dir / fname
                try:
                    df_final.to_csv(outpath, index=False)
                    print(f"Saved processed data for dive {dive_id}, vehicle {vehicle} to: {outpath}")
                except Exception as e:
                    print(f"Error saving file {outpath}: {e}")
        else:
            print(f"No valid data to process for dive {dive_id}.")
