import os
import pandas as pd
import csv
from datetime import datetime

def get_file_paths(base_directory, dive_number):
    """
    Constructs file paths for various datasets related to a specific dive.
    """
    expedition = os.path.basename(os.path.normpath(base_directory))
    sampled_dir = os.path.join(base_directory, "processed", "dive_reports", dive_number, "sampled")
    
    return {
        "ctd": os.path.join(sampled_dir, f"{dive_number}.CTD.sampled.tsv"),
        "sealog": os.path.join(base_directory, "raw", "sealog", "sealog-herc", dive_number, f"{dive_number}_sealogExport.csv"),
        "summary": os.path.join(base_directory, "RUMI_processed", "all_dive_summaries.csv"),
        "output_dir": base_directory,
        "expedition": expedition,
        "sampled_dir": sampled_dir,
    }

def to_iso8601_str(dt_series):
    """
    Converts a datetime Series to ISO8601 format with 'Z' to indicate UTC.
    """
    return dt_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def load_tsv_file(file_path, sensor_name=None, enforce_negative=False, column_names=None, drop_temperature=False):
    """
    Loads a TSV file and processes timestamps, enforcing negative values if required.
    If column_names are provided, only the first (1 + len(column_names)) columns are used.
    """
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, quoting=csv.QUOTE_MINIMAL)
        
        if column_names:
            expected = 1 + len(column_names)
            if df.shape[1] > expected:
                df = df.iloc[:, :expected]
            elif df.shape[1] < expected:
                print(f"Warning: Expected {expected} columns, found {df.shape[1]} in {file_path}.")
            df.columns = ["Timestamp"] + column_names
        else:
            df.columns = ["Timestamp"] + [f"{sensor_name}_{i+1}" for i in range(df.shape[1]-1)]
        
        df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip(), utc=True, errors="coerce")
        
        if enforce_negative:
            for col in df.columns[1:]:
                df[col] = -df[col].abs()
        
        if drop_temperature and "Temperature" in df.columns:
            df = df.drop(columns=["Temperature"], errors="ignore")
        
        return df.drop_duplicates(subset=["Timestamp"])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_sealog_file(file_path):
    """
    Loads and processes a Sealog CSV file.
    If some of the desired columns are missing, it reads all columns then keeps only those available.
    For the Timestamp conversion, it first attempts to use a specific format to avoid warnings.
    """
    if not os.path.exists(file_path):
        return None
    try:
        desired_columns = [
            "ts", "event_value", "event_free_text", "event_option.channel", "event_option.milestone",
            "event_option.rating", "event_option.vehicle", "vehicleRealtimeDualHDGrabData.camera_name_2_uom",
            "vehicleRealtimeDualHDGrabData.camera_name_2_value", "vehicleRealtimeDualHDGrabData.camera_name_uom",
            "vehicleRealtimeDualHDGrabData.camera_name_value", "vehicleRealtimeDualHDGrabData.filename_2_uom",
            "vehicleRealtimeDualHDGrabData.filename_2_value", "vehicleRealtimeDualHDGrabData.filename_uom",
            "vehicleRealtimeDualHDGrabData.filename_value"
        ]
        # Read CSV without using usecols.
        df = pd.read_csv(file_path, low_memory=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Loaded {len(df)} rows from Sealog: {file_path}")
        
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "Timestamp"})
        # Filter to keep only desired columns that are present.
        available = [col for col in desired_columns if col in df.columns]
        if "Timestamp" not in available and "Timestamp" in df.columns:
            available = ["Timestamp"] + available
        df = df[available]
        
        # Attempt to parse Timestamp using a specific format.
        ts_str = df["Timestamp"].astype(str).str.strip()
        try:
            df["Timestamp"] = pd.to_datetime(ts_str, format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)
        except Exception:
            df["Timestamp"] = pd.to_datetime(ts_str, utc=True, errors="coerce")
        df["Timestamp"] = df["Timestamp"].dt.round("s")
        return df.drop_duplicates(subset=["Timestamp"])
    except Exception as e:
        print(f"Error reading Sealog file {file_path}: {e}")
        return None

def process_single_dive(root_dir, expedition, dive_number):
    """
    Processes a single dive by merging various sensor datasets and saves the output in the correct dive folder.
    """
    dive_output_dir = os.path.join(root_dir, "RUMI_processed", dive_number)
    os.makedirs(dive_output_dir, exist_ok=True)
    
    paths = get_file_paths(root_dir, dive_number)
    
    ctd_columns = ["Temperature", "Conductivity", "Pressure", "Salinity", "Sound_Velocity"]
    o2s_columns = ["O2_Concentration", "O2_Saturation"]
    
    ctd_df = load_tsv_file(paths["ctd"], sensor_name="CTD", column_names=ctd_columns)
    if ctd_df is None:
        print(f"Skipping {dive_number} due to missing CTD data.")
        return
    
    initial_rows = len(ctd_df)
    print(f"CTD dataset initial rows: {initial_rows}")
    
    datasets = {
        "Sealog": load_sealog_file(paths["sealog"]),
        "Herc_Depth": load_tsv_file(os.path.join(paths["sampled_dir"], f"{dive_number}.DEP1.sampled.tsv"),
                                     sensor_name="Herc_Depth", enforce_negative=True),
        "Atalanta_Depth": load_tsv_file(os.path.join(paths["sampled_dir"], f"{dive_number}.DEP2.sampled.tsv"),
                                        sensor_name="Atalanta_Depth", enforce_negative=True),
        "O2S": load_tsv_file(os.path.join(paths["sampled_dir"], f"{dive_number}.O2S.sampled.tsv"),
                             sensor_name="O2S", column_names=o2s_columns, drop_temperature=True),
    }
    
    for name, df in datasets.items():
        if df is not None:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
            ctd_df = ctd_df.merge(df, on="Timestamp", how="left")
    
    final_rows = len(ctd_df)
    ctd_df["Timestamp"] = to_iso8601_str(ctd_df["Timestamp"])
    
    output_path = os.path.join(dive_output_dir, f"{expedition}_{dive_number}_merged.csv")
    ctd_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"Summary for dive {dive_number}:")
    print(f"  - Initial CTD rows: {initial_rows}")
    print(f"  - Final merged file rows: {final_rows}")
    print(f"Saved: {output_path}\n")

def process_data(root_dir):
    """
    Main processing function for dive sensor and Sealog data.
    Processes only those dives that are listed in the dive summaries CSV.
    """
    expedition = os.path.basename(os.path.normpath(root_dir))
    summary_path = os.path.join(root_dir, "RUMI_processed", "all_dive_summaries.csv")
    print("Looking for dive summaries at:", os.path.abspath(summary_path))
    if not os.path.exists(summary_path):
        print("Dive summaries file not found. Cannot process sensor data.")
        return
    
    try:
        summary_df = pd.read_csv(summary_path)
        # Assume the dive summaries CSV has a column named "dive"
        valid_dives = set(summary_df["dive"].astype(str).str.upper().str.strip())
        dive_reports_dir = os.path.join(root_dir, "processed", "dive_reports")
        all_dives = [item.upper() for item in os.listdir(dive_reports_dir)
                     if os.path.isdir(os.path.join(dive_reports_dir, item)) and item.upper().startswith("H")]
        # Filter to only include dives that are in the dive summaries CSV.
        dive_list = [d for d in all_dives if d in valid_dives]
        if not dive_list:
            print("No valid dives found in dive summaries. Cannot process sensor data.")
            return
    except Exception as e:
        print(f"Error loading dive summaries: {e}")
        return
    
    for dive_num in dive_list:
        print(f"Processing dive {dive_num}...")
        process_single_dive(root_dir, expedition, dive_num)

if __name__ == "__main__":
    root_directory = input("Enter the base directory for processing: ").strip()
    process_data(root_directory)
