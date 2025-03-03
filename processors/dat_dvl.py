import os
import re
import pandas as pd
from datetime import datetime, timezone

def parse_dat_file(filepath):
    """
    Parses a .DAT file for lines starting with 'OCT' from Hercules data.
    Extracts timestamp, heading, pitch, roll, acceleration, and angular velocity.
    The output is structured for Kalman filtering against USBL fixes.
    """
    pattern = re.compile(
        r'^OCT\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d+.\d+)\s+Hercules\s+'
        r'[\-\d.]+\s+[\-\d.]+\s+[\-\d.]+\s+'
        r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+'
        r'[\-\d.]+\s+[\-\d.]+\s+[\-\d.]+\s+'
        r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+'
        r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+'
        r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)'
    )
    data = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                date_str, time_str = m.group(1), m.group(2)
                dt_str = date_str + " " + time_str  
                try:
                    dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
                
                dt = dt.replace(tzinfo=timezone.utc)

                heading, pitch, roll = map(float, m.group(3, 4, 5))
                heading_rate, pitch_rate, roll_rate = map(float, m.group(6, 7, 8))
                x_accel, y_accel, z_accel = map(float, m.group(9, 10, 11))
                heading_accel, pitch_accel, roll_accel = map(float, m.group(12, 13, 14))

                data.append([
                    dt, heading, pitch, roll, heading_rate, pitch_rate, roll_rate,
                    x_accel, y_accel, z_accel, heading_accel, pitch_accel, roll_accel
                ])
    
    return pd.DataFrame(data, columns=[
        "Timestamp", "Heading", "Pitch", "Roll",
        "HeadingRate", "PitchRate", "RollRate",
        "XAccel", "YAccel", "ZAccel",
        "HeadingAccel", "PitchAccel", "RollAccel"
    ])

def process_all_dat_files(root_dir):
    """
    Reads all .DAT files and extracts Hercules OCT lines.
    Returns a combined DataFrame sorted by Timestamp.
    """
    navest_dir = os.path.join(root_dir, "raw", "nav", "navest")  # ✅ Use root_dir

    if not os.path.exists(navest_dir):
        raise FileNotFoundError(f"NavEst directory not found at {navest_dir}")
    
    all_files = [f for f in os.listdir(navest_dir) if f.lower().endswith(".dat")]
    combined = pd.DataFrame()
    
    for fname in all_files:
        filepath = os.path.join(navest_dir, fname)
        df = parse_dat_file(filepath)
        if not df.empty:
            combined = pd.concat([combined, df], ignore_index=True)

    if not combined.empty:
        combined.sort_values("Timestamp", inplace=True)
    return combined


def preserve_closest_fix_per_second(df):
    """
    Rounds timestamps to the nearest second and keeps only one row per second.
    """
    if df.empty:
        return df, 0, 0, 0

    orig_count = len(df)
    df = df.copy()
    
    df["rounded_dt"] = df["Timestamp"].apply(lambda dt: datetime.fromtimestamp(round(dt.timestamp()), tz=dt.tzinfo))
    df["diff"] = (df["Timestamp"] - df["rounded_dt"]).abs()
    
    df_uniq = df.loc[df.groupby("rounded_dt")["diff"].idxmin()].copy()

    final_count = len(df_uniq)
    duplicates_removed = orig_count - final_count

    df_uniq.sort_values("rounded_dt", inplace=True)
    df_uniq["Timestamp"] = df_uniq["rounded_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df_uniq.drop(columns=["rounded_dt", "diff"], inplace=True)
    df_uniq.reset_index(drop=True, inplace=True)

    return df_uniq, orig_count, final_count, duplicates_removed

def process_dive_vehicle_rows(dive_info, dvl_data):
    """
    Filters DVL data for a given dive, rounds timestamps, and removes duplicates.
    """
    dive_id = str(dive_info["dive"]).strip()
    launch, recovery = dive_info["Launch Time"], dive_info["Recovery Time"]
    
    expected_seconds = int((recovery - launch).total_seconds())

    df_sub = dvl_data[(dvl_data["Timestamp"] >= launch) & (dvl_data["Timestamp"] <= recovery)].copy()
    if df_sub.empty:
        return pd.DataFrame(), 0, 0, 0, expected_seconds

    df_rounded, orig_count, rounded_count, duplicates = preserve_closest_fix_per_second(df_sub)

    df_rounded["Timestamp"] = pd.to_datetime(df_rounded["Timestamp"], errors="coerce", utc=True)
    df_rounded["Timestamp"] = df_rounded["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df_rounded, orig_count, rounded_count, duplicates, expected_seconds

def output_dive_csv(root_dir, expedition_name, dive_id, df):
    """Saves processed DVL data for a given dive in its own folder."""
    if df.empty:
        return None

    dive_output_dir = os.path.join(root_dir, "RUMI_processed", dive_id)  # ✅ Create dive subdir
    os.makedirs(dive_output_dir, exist_ok=True)

    fname = f"{expedition_name}_{dive_id}_DVL.csv"
    outpath = os.path.join(dive_output_dir, fname)  # ✅ Store in dive folder
    df.to_csv(outpath, index=False)

    print(f"Saved DVL data to: {outpath}")
    return outpath


def process_data(root_dir):
    """
    Main processing function that is called from main.py.
    Handles all steps of DVL data processing.
    """
    navest_dir = os.path.join(root_dir, "raw", "nav", "navest")  # ✅ Use root_dir directly
    if not os.path.exists(navest_dir):
        print(f"Error: NavEst directory not found at {navest_dir}")
        return
    
    # Load .DAT files
    try:
        dvl_data = process_all_dat_files(root_dir)  # ✅ Use root_dir instead of base_dir
    except Exception as e:
        print(f"Error reading .DAT files: {e}")
        return
    
    if dvl_data.empty:
        print("No Hercules OCT lines found in any .DAT file.")
        return
    
    # Load dive summaries
    try:
        dive_summary_path = os.path.join(root_dir, "RUMI_processed", "all_dive_summaries.csv")  # ✅ Corrected Path
        if not os.path.exists(dive_summary_path):
            raise FileNotFoundError(f"Dive summary file not found at {dive_summary_path}")

        ds = pd.read_csv(dive_summary_path)
        ds["Launch Time"] = pd.to_datetime(ds["Launch Time"], utc=True, errors="coerce")
        ds["Recovery Time"] = pd.to_datetime(ds["Recovery Time"], utc=True, errors="coerce")

    except Exception as e:
        print(f"Error loading dive summaries: {e}")
        return

    total_dive_fixes = 0  # ✅ Track total fixes processed across all dives
    total_expected_seconds = 0

    for _, row in ds.iterrows():
        expedition = str(row["expedition"]).strip()
        dive_id = str(row["dive"]).strip()
        
        df_dive, orig_count, rounded_count, duplicates, expected_seconds = process_dive_vehicle_rows(row, dvl_data)
        total_expected_seconds += expected_seconds

        if df_dive.empty:
            print(f"⚠️ Dive {dive_id}: No DVL data found for this dive window.")
            continue

        output_dive_csv(root_dir, expedition, dive_id, df_dive)  # ✅ Pass `root_dir`

        total_dive_fixes += rounded_count  # ✅ Track total fixes

        # ✅ Print a detailed summary for each dive
        print(f"\n✅ Dive {dive_id} Summary:")
        print(f"  - Estimated Duration: {expected_seconds} seconds")
        print(f"  - Original DVL Fixes: {orig_count}")
        print(f"  - After Rounding: {rounded_count}")
        print(f"  - Duplicates Removed: {duplicates}")
        print(f"  - Percentage of Expected Coverage: {rounded_count / expected_seconds * 100:.2f}%")

    # ✅ Print a final summary after all dives are processed
    print("\n✅ DVL Processing Complete!")
    print(f"  - Total Dives Processed: {len(ds)}")
    print(f"  - Total Expected Seconds: {total_expected_seconds}")
    print(f"  - Total DVL Fixes Processed: {total_dive_fixes}")
    print(f"  - Data stored in {os.path.join(root_dir, 'RUMI_processed')}")
