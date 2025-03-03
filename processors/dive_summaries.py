import os
import csv  # ✅ Ensure csv module is imported
import pandas as pd
from datetime import timedelta

def extract_objective(summary_filepath):
    """Reads 'Objective:' line from summary file."""
    try:
        with open(summary_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Objective:"):
                    return line[len("Objective:"):].strip()
    except Exception as e:
        print(f"Error reading summary file {summary_filepath}: {e}")
    return ""

def convert_to_iso(dt_series):
    """Converts a Pandas Series of datetime values to ISO8601 format without subseconds."""
    return dt_series.apply(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notnull(dt) else dt)

def read_tsv_with_commented_header(tsv_filepath):
    """Reads a TSV file whose first header line is commented (starts with '##')."""
    with open(tsv_filepath, "r", encoding="utf-8") as f:
        header_line = f.readline().lstrip("#").strip()  # ✅ Remove leading ##
        headers = header_line.split("\t")

    # ✅ Rename the first column explicitly to 'expedition'
    headers[0] = "expedition"

    df = pd.read_csv(tsv_filepath, sep="\t", skiprows=1, header=None, names=headers)

    return df

def process_dive_folder(dive_folder_path, dive_number):
    """Processes a single dive folder by reading stats and summary files."""
    stats_filepath = os.path.join(dive_folder_path, f"{dive_number}-stats.tsv")
    summary_filepath = os.path.join(dive_folder_path, f"{dive_number}-summary.txt")

    if not os.path.exists(stats_filepath) or not os.path.exists(summary_filepath):
        print(f"Skipping {dive_number}: Missing required files.")
        return None

    try:
        df = read_tsv_with_commented_header(stats_filepath)
    except Exception as e:
        print(f"Error reading stats file {stats_filepath}: {e}")
        return None

    if "inwatertime" not in df.columns or "totaltime(hours)" not in df.columns:
        print(f"Skipping {dive_number}: Required columns missing.")
        return None

    try:
        df["Launch Time"] = pd.to_datetime(df["inwatertime"], utc=True, errors='coerce')
        df["Recovery Time"] = df["Launch Time"] + pd.to_timedelta(df["totaltime(hours)"], unit='h')
    except Exception as e:
        print(f"Skipping {dive_number}: Error parsing timestamps: {e}")
        return None

    if (df["Recovery Time"].iloc[0] - df["Launch Time"].iloc[0]) < timedelta(hours=2):
        print(f"Skipping {dive_number}: Dive too short (< 2 hours).")
        return None

    # ✅ Strip subseconds from time fields
    time_fields = ["Launch Time", "On Bottom Time", "Off Bottom Time", "Recovery Time"]
    for field in time_fields:
        if field in df.columns:
            df[field] = convert_to_iso(pd.to_datetime(df[field], utc=True, errors='coerce'))

    df["Dive End"] = convert_to_iso(pd.to_datetime(df["Recovery Time"], utc=True, errors='coerce'))
    df["Objective"] = extract_objective(summary_filepath)

    # ✅ Ensure correct column names
    rename_dict = {
        "inwatertime": "Launch Time",
        "onbottomtime": "On Bottom Time",
        "offbottomtime": "Off Bottom Time",
        "ondecktime": "Recovery Time",
        "hercmaxdepth": "Herc Max Depth",
        "hercavgdepth": "Herc Avg Depth",
        "argusmaxdepth": "Atalanta Max Depth",
        "argusavgdepth": "Atalanta Avg Depth",
        "totaltime(hours)": "Total Time (hours)",
        "bottomtime(hours)": "Bottom Time (hours)"
    }
    df.rename(columns=rename_dict, inplace=True)

    # ✅ Strip underscores from `site` column
    if "site" in df.columns:
        df["site"] = df["site"].astype(str).str.replace("_", " ")

    return df

def concatenate_dive_summaries(root_dir):
    """Processes all dive reports and saves a combined summary without subseconds in timestamps."""
    dive_reports_path = os.path.join(root_dir, "processed", "dive_reports")
    if not os.path.exists(dive_reports_path):
        raise FileNotFoundError(f"Dive reports directory not found at {dive_reports_path}")

    combined_dfs = []
    for item in os.listdir(dive_reports_path):
        dive_folder = os.path.join(dive_reports_path, item)
        if os.path.isdir(dive_folder) and item.upper().startswith("H"):
            df = process_dive_folder(dive_folder, item.upper())
            if df is not None:
                combined_dfs.append(df)

    if not combined_dfs:
        print("No dive summaries were processed.")
        return

    all_dive_df = pd.concat(combined_dfs, ignore_index=True)

    # ✅ Debugging: Print column names before processing timestamps
    print(f"✅ Column Names Before Processing: {list(all_dive_df.columns)}")

    # ✅ Remove duplicate columns (only keep the first occurrence)
    all_dive_df = all_dive_df.loc[:, ~all_dive_df.columns.duplicated()]

    # ✅ Ensure 'expedition' column is included
    if "expedition" not in all_dive_df.columns:
        print("❌ Warning: 'expedition' column is missing! Check read_tsv_with_commented_header().")

    # ✅ List of time-related columns to clean
    time_fields = ["Launch Time", "On Bottom Time", "Off Bottom Time", "Recovery Time", "Dive End"]

    # ✅ Ensure all time columns are stored as strings formatted without subseconds
    for field in time_fields:
        if field in all_dive_df.columns:
            all_dive_df[field] = pd.to_datetime(all_dive_df[field], utc=True, errors='coerce').dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ✅ Save correctly formatted `all_dive_summaries.csv`
    summary_csv = os.path.join(root_dir, "RUMI_processed", "all_dive_summaries.csv")
    all_dive_df.to_csv(summary_csv, index=False, quoting=csv.QUOTE_ALL)

    print(f"\n✅ Combined dive summaries saved to: {summary_csv} (Subseconds Removed)")

def process_data(root_dir):
    """Processes dive summaries from the correct root directory."""
    dive_reports_path = os.path.join(root_dir, "processed", "dive_reports")

    if not os.path.exists(dive_reports_path):
        print(f"Error: Dive reports directory not found at {dive_reports_path}")
        return

    print(f"Processing dive summaries from {dive_reports_path}...")

    # ✅ Call the function to process dive summaries
    concatenate_dive_summaries(root_dir)
