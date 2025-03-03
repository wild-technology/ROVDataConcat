import os
import re
import csv  # ✅ Ensure CSV module is imported
import pandas as pd
from datetime import datetime, timedelta, timezone

def parse_sdyn_file(filepath):
    """
    Parses an SDYN file containing USBL fixes from the datalog directory.
    Extracts timestamp, latitude, longitude, accuracy, depth, and vehicle.
    """
    filename = os.path.basename(filepath)
    base, _ = os.path.splitext(filename)  
    try:
        file_start = datetime.strptime(base, "%Y%m%d_%H%M")
    except ValueError:
        print(f"Filename {filename} does not match expected format YYYYMMDD_HHMM.SDYN")
        return pd.DataFrame()
    
    file_start = file_start.replace(tzinfo=timezone.utc)

    pattern = re.compile(
        r'\$GPGGA,'
        r'(\d+\.\d+),'  
        r'(\d{2})(\d{2}\.\d+),'  
        r'([NS]),'
        r'(\d{3})(\d{2}\.\d+),'  
        r'([EW]),'
        r'\d+,\d+,'  
        r'([\d.]+),'  
        r'([-0-9.]+),'  
        r'M,0\.0,M,0\.0,'
        r'(\d{4})\*'  
        r'([0-9A-F]+)'
    )

    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                (time_str, lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir,
                 accuracy, depth, beacon_index, checksum) = match.groups()

                try:
                    fix_time = datetime.strptime(time_str, "%H%M%S.%f")
                except ValueError:
                    fix_time = datetime.strptime(time_str, "%H%M%S")

                fix_time = fix_time.replace(tzinfo=timezone.utc)
                full_timestamp = file_start.replace(hour=fix_time.hour,
                                                    minute=fix_time.minute,
                                                    second=fix_time.second,
                                                    microsecond=fix_time.microsecond)
                if full_timestamp < file_start:
                    full_timestamp += timedelta(days=1)

                lat = float(lat_deg) + float(lat_min) / 60.0
                if lat_dir.upper() == 'S':
                    lat = -lat

                lon = float(lon_deg) + float(lon_min) / 60.0
                if lon_dir.upper() == 'W':
                    lon = -lon

                vehicle = "Unknown"
                beacon_int = int(beacon_index)
                if beacon_int == 1:
                    vehicle = "Hercules"
                elif beacon_int == 2:
                    vehicle = "Atalanta"

                data.append([full_timestamp, lat, lon, float(accuracy), float(depth), vehicle])

    return pd.DataFrame(data, columns=["Timestamp", "Latitude", "Longitude", "Accuracy", "Depth", "Vehicle"])

def process_all_sdyn_files(root_directory):
    """
    Reads all .SDYN files from raw/datalog and extracts USBL navigation fixes.
    """
    sdyn_dir = os.path.join(root_directory, "raw", "datalog")
    if not os.path.exists(sdyn_dir):
        raise FileNotFoundError(f"SDYN directory not found at {sdyn_dir}")

    files = [f for f in os.listdir(sdyn_dir) if f.endswith(".SDYN")]
    combined_data = pd.DataFrame()

    for file in files:
        filepath = os.path.join(sdyn_dir, file)
        df = parse_sdyn_file(filepath)
        if not df.empty:
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data

def compute_summary_stats(df, vehicle):
    """
    Computes number of fixes, average time interval, and standard deviation for a given vehicle's data.
    """
    count = len(df)
    if count < 2:
        return count, None, None  # Not enough data for interval calculations

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    intervals = df["Timestamp"].diff().dropna().dt.total_seconds()
    
    return count, intervals.mean(), intervals.std()

def process_dive_vehicle(dive_summary, sdyn_data):
    """
    Filters USBL data for a given dive, rounds timestamps, and returns processed data for each vehicle.
    """
    dive_id = str(dive_summary["dive"]).strip()
    launch_time = dive_summary["Launch Time"]
    recovery_time = dive_summary["Recovery Time"]

    df_dive = sdyn_data[(sdyn_data["Timestamp"] >= launch_time) & (sdyn_data["Timestamp"] <= recovery_time)]
    if df_dive.empty:
        print(f"No USBL fixes for dive {dive_id}.")
        return {}

    processed = {}
    for vehicle in df_dive["Vehicle"].unique():
        if vehicle == "Unknown":
            continue  
        df_vehicle = df_dive[df_dive["Vehicle"] == vehicle].copy()
        df_vehicle["Timestamp"] = pd.to_datetime(df_vehicle["Timestamp"], utc=True)
        df_vehicle["Timestamp"] = df_vehicle["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        processed[vehicle] = df_vehicle

    return processed

def output_dive_vehicle_files(root_directory, dive_summary, processed_by_vehicle):
    """
    Saves USBL data for each vehicle and prints summary statistics.
    """
    expedition = dive_summary["expedition"].strip()
    dive_id = str(dive_summary["dive"]).strip()

    dive_output_dir = os.path.join(root_directory, "RUMI_processed", dive_id)
    os.makedirs(dive_output_dir, exist_ok=True)

    print(f"\n--- Summary for Dive {dive_id} ({expedition}) ---")
    
    for vehicle, df in processed_by_vehicle.items():
        filename = f"{expedition}_{dive_id}_USBL_{vehicle}.csv"
        output_path = os.path.join(dive_output_dir, filename)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved: {output_path}")

        # Compute and print summary statistics
        count, avg_interval, std_dev = compute_summary_stats(df, vehicle)
        print(f"  Vehicle: {vehicle}")
        print(f"    Number of fixes: {count}")
        if avg_interval is not None:
            print(f"    Average interval between fixes: {avg_interval:.2f} seconds")
            print(f"    Standard deviation of intervals: {std_dev:.2f} seconds")
        else:
            print(f"    Not enough data for interval calculations.")

def process_data(root_directory):
    """
    Main processing function for USBL data.
    """
    try:
        dive_summaries = pd.read_csv(os.path.join(root_directory, "RUMI_processed", "all_dive_summaries.csv"))
        dive_summaries["Launch Time"] = pd.to_datetime(dive_summaries["Launch Time"], utc=True)
        dive_summaries["Recovery Time"] = pd.to_datetime(dive_summaries["Recovery Time"], utc=True)
    except Exception as e:
        print(f"Error loading dive summaries: {e}")
        return

    try:
        sdyn_data = process_all_sdyn_files(root_directory)
    except Exception as e:
        print(f"Error processing SDYN files: {e}")
        return

    if sdyn_data.empty:
        print("No USBL fixes found.")
        return

    for _, dive_row in dive_summaries.iterrows():
        processed = process_dive_vehicle(dive_row, sdyn_data)
        if processed:
            output_dive_vehicle_files(root_directory, dive_row, processed)

if __name__ == "__main__":
    root_dir = input("Enter the base directory for processing: ").strip()
    process_data(root_dir)
