from pathlib import Path
import re
import pandas as pd
from datetime import datetime, timedelta, timezone

from processors.common import best_fix_per_second, drop_duplicate_timestamps
from processors.report import RunReport

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
        r'\d+,\d+,'  # Skip fix quality and satellite count
        # This field occupies the HDOP slot of a standard GPGGA sentence, but
        # in the Sonardyne SDYN output it behaves as an estimated positional
        # accuracy in meters: on NA167/H2075 it is ~14.5 m at ~1024 m depth
        # (~1.4% of slant range, matching typical USBL error specs), whereas a
        # dimensionless HDOP would sit near 1-2. Downstream (kalman_filter)
        # squares it as measurement variance in m^2, which is consistent with
        # that interpretation.
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

                # GPGGA time is fixed-width HHMMSS[.sss]; zero-pad defensively so
                # a value like '12345.6' parses as 01:23:45.6 rather than being
                # misread by strptime's lenient 1-2 digit matching.
                if "." in time_str:
                    int_part, frac_part = time_str.split(".", 1)
                    time_str = f"{int_part.zfill(6)}.{frac_part}"
                else:
                    time_str = time_str.zfill(6)
                try:
                    fix_time = datetime.strptime(time_str, "%H%M%S.%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        fix_time = datetime.strptime(time_str, "%H%M%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue  # malformed time field; skip this fix

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
        # Keep the best-accuracy fix per second; result is chronologically sorted.
        df_vehicle, orig, final = best_fix_per_second(df_vehicle, quality_col="Accuracy")
        if orig != final:
            print(f"Reduced {orig} fixes to {final} (best accuracy per second) "
                  f"for dive {dive_id}, vehicle {vehicle}")

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

    report = RunReport("usbl_sdyn", processed_dir)
    report.metric("raw_fixes_parsed", len(sdyn_data))

    if sdyn_data.empty:
        print("No USBL fixes found.")
        report.error("no-data", "no USBL fixes parsed from any SDYN file")
        report.finalize()
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
                # Final safety net: drop any duplicate timestamps, keep chronological order.
                df_final, final_dupes_removed = drop_duplicate_timestamps(df_vehicle)
                if final_dupes_removed > 0:
                    print(
                        f"Removed {final_dupes_removed} final duplicate timestamps for dive {dive_id}, vehicle {vehicle}")

                fname = f"{expedition}_{dive_id}_USBL_{vehicle}.csv"
                outpath = dive_out_dir / fname
                try:
                    df_final.to_csv(outpath, index=False)
                    print(f"Saved processed data for dive {dive_id}, vehicle {vehicle} to: {outpath}")
                    report.add_output(outpath, rows=len(df_final))
                except Exception as e:
                    print(f"Error saving file {outpath}: {e}")
                    report.error("write-failed", f"could not write {outpath}: {e}")
        else:
            print(f"No valid data to process for dive {dive_id}.")
            report.warn("no-data", f"dive {dive_id}: no USBL fixes in the dive window")

    report.finalize()
