from pathlib import Path
import re
import csv
import pandas as pd
from datetime import datetime, timezone, timedelta

# ------------------------------------------------------------------------------
# Function: split_lat_long
# ------------------------------------------------------------------------------
def split_lat_long(df, column_name):
	"""
	Splits a column containing a combined latitude-longitude string (separated by a space)
	into two separate numeric columns: {column_name}_lat and {column_name}_long.
	"""
	if column_name in df.columns:
		df[[f"{column_name}_lat", f"{column_name}_long"]] = (
			df[column_name].str.split(" ", expand=True).astype(float)
		)
		df.drop(columns=[column_name], inplace=True)
		old_cols = list(df.columns)
		lat_col, long_col = f"{column_name}_lat", f"{column_name}_long"
		for c in [lat_col, long_col]:
			if c in old_cols:
				old_cols.remove(c)
		old_cols.append(lat_col)
		old_cols.append(long_col)
		df = df[old_cols]
	return df

# ------------------------------------------------------------------------------
# Function: extract_objective
# ------------------------------------------------------------------------------
def extract_objective(summary_filepath):
	"""
	Extracts the objective text from a summary file.
	Reads each line until one starting with 'Objective:' is found.
	"""
	try:
		with summary_filepath.open("r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("Objective:"):
					return line[len("Objective:"):].strip()
	except Exception as e:
		print(f"Error reading summary file {summary_filepath}: {e}")
	return ""

# ------------------------------------------------------------------------------
# Function: convert_to_iso
# ------------------------------------------------------------------------------
def convert_to_iso(dt_series):
	"""
	Converts a Pandas Series of datetimes to ISO8601 format (UTC) without sub-seconds.
	"""
	return dt_series.apply(
		lambda dt: dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notnull(dt) else dt
	)

# ------------------------------------------------------------------------------
# Function: parse_dat_file_both
# ------------------------------------------------------------------------------
def parse_dat_file_both(filepath):
	"""
	Opens a .DAT file once and extracts two sets of data:
	  - OCT lines (Hercules pitch/roll/heading)
	  - VFR lines (Hercules lat/long with fix type SOLN_DEADRECK)

	Returns two DataFrames:
	  oct_df: columns=["Timestamp", "Heading", "Pitch", "Roll"]
	  vfr_df: columns=["Timestamp", "Longitude", "Latitude"]
	"""
	# Regex for OCT lines
	oct_pattern = re.compile(
		r'^OCT\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d+\.\d+)\s+Hercules\s+'
		r'[\-\d.]+\s+[\-\d.]+\s+[\-\d.]+\s+'
		r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+'
		r'[\-\d.]+\s+[\-\d.]+\s+[\-\d.]+\s+'
		r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+'
		r'([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)'
	)
	# Regex for VFR lines
	vfr_pattern = re.compile(
		r"^VFR\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+([-\d.]+)\s+([-\d.]+)"
	)

	oct_data = []
	vfr_data = []

	with filepath.open('r', encoding='utf-8', errors='ignore') as f:
		for line in f:
			line_str = line.strip()
			if not line_str:
				continue

			# Check for OCT match
			oct_match = oct_pattern.match(line_str)
			if oct_match:
				date_str, time_str = oct_match.group(1), oct_match.group(2)
				dt_str = f"{date_str} {time_str}"
				try:
					dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
				except ValueError:
					dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
				dt = dt.replace(tzinfo=timezone.utc)
				heading, pitch, roll = map(float, oct_match.group(3, 4, 5))
				oct_data.append([dt, heading, pitch, roll])
				continue

			# Check for VFR match
			vfr_match = vfr_pattern.match(line_str)
			if vfr_match:
				vehicle_number = vfr_match.group(4)
				fix_type = vfr_match.group(5)
				if vehicle_number != "0" or fix_type != "SOLN_DEADRECK":
					continue
				date_str, time_str = vfr_match.group(1), vfr_match.group(2)
				dt_str = f"{date_str} {time_str}"
				try:
					dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
				except ValueError:
					dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
				dt = dt.replace(tzinfo=timezone.utc)
				lon_str = vfr_match.group(6)
				lat_str = vfr_match.group(7)
				try:
					lon_val = float(lon_str)
					lat_val = float(lat_str)
					if abs(lon_val) < 1 and abs(lat_val) < 1:
						continue
				except ValueError:
					continue
				vfr_data.append([dt, lon_str, lat_str])

	oct_df = pd.DataFrame(oct_data, columns=["Timestamp", "Heading", "Pitch", "Roll"])
	vfr_df = pd.DataFrame(vfr_data, columns=["Timestamp", "Longitude", "Latitude"])
	return oct_df, vfr_df

# ------------------------------------------------------------------------------
# Function: process_all_dat_files_both
# ------------------------------------------------------------------------------
def process_all_dat_files_both(root_dir):
	"""
	Iterates over all .DAT files in <root_dir>/raw/nav/navest/ and extracts both OCT and VFR data.

	Returns two DataFrames: one for OCT and one for VFR.
	"""
	root_dir = Path(root_dir)  # Convert to Path if it's a string
	navest_dir = root_dir / "raw" / "nav" / "navest"
	if not navest_dir.exists():
		raise FileNotFoundError(f"NavEst directory not found at {navest_dir}")

	all_files = list(navest_dir.glob("*.dat")) + list(navest_dir.glob("*.DAT"))
	all_oct = pd.DataFrame()
	all_vfr = pd.DataFrame()

	for filepath in all_files:
		oct_df, vfr_df = parse_dat_file_both(filepath)
		if not oct_df.empty:
			all_oct = pd.concat([all_oct, oct_df], ignore_index=True)
		if not vfr_df.empty:
			all_vfr = pd.concat([all_vfr, vfr_df], ignore_index=True)

	if not all_oct.empty:
		all_oct.sort_values("Timestamp", inplace=True)
	if not all_vfr.empty:
		all_vfr.sort_values("Timestamp", inplace=True)

	return all_oct, all_vfr

# ------------------------------------------------------------------------------
# Function: preserve_closest_fix_per_second
# ------------------------------------------------------------------------------
def preserve_closest_fix_per_second(df):
	"""
	Rounds timestamps to the nearest second and, for each unique second,
	retains the row closest to that second. Any fix more than one second away is discarded.
	Finally, ensures that no duplicate timestamps exist in the final output.
	"""
	if df.empty:
		return df, 0, 0, 0

	orig_count = len(df)
	df = df.copy()

	# Round timestamps to the nearest second
	df["rounded_dt"] = df["Timestamp"].apply(
		lambda dt: datetime.fromtimestamp(round(dt.timestamp()), tz=dt.tzinfo)
	)
	df["diff"] = (df["Timestamp"] - df["rounded_dt"]).abs()

	# Keep the closest fix per second
	best_rows = df.groupby("rounded_dt")["diff"].idxmin()
	df_unique = df.loc[best_rows].copy()
	df_unique = df_unique[df_unique["diff"] <= timedelta(seconds=1)]
	df_unique.sort_values("rounded_dt", inplace=True)

	# Format timestamps as ISO8601 strings
	df_unique["Timestamp"] = df_unique["rounded_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

	# Check for and remove any remaining duplicate timestamps
	before_dedup = len(df_unique)
	df_unique = df_unique.drop_duplicates(subset=["Timestamp"])
	duplicates_removed_extra = before_dedup - len(df_unique)

	if duplicates_removed_extra > 0:
		print(f"  - Additional duplicates removed: {duplicates_removed_extra}")

	df_unique.drop(columns=["rounded_dt", "diff"], inplace=True)
	df_unique.reset_index(drop=True, inplace=True)

	final_count = len(df_unique)
	duplicates_removed = orig_count - final_count

	return df_unique, orig_count, final_count, duplicates_removed

# ------------------------------------------------------------------------------
# Function: remove_timestamp_duplicates
# ------------------------------------------------------------------------------
def remove_timestamp_duplicates(df):
	"""
	Final check to ensure there are no duplicate timestamps in the DataFrame.

	Parameters:
		df (pandas.DataFrame): DataFrame containing a 'Timestamp' column.

	Returns:
		pandas.DataFrame: DataFrame with duplicate timestamps removed.
		int: Number of duplicate rows removed.
	"""
	if df.empty:
		return df.copy(), 0

	# Count rows before deduplication
	before_count = len(df)

	# Remove duplicates keeping the first occurrence
	df_no_dupes = df.drop_duplicates(subset=["Timestamp"])

	# Count how many duplicates were removed
	removed_count = before_count - len(df_no_dupes)

	return df_no_dupes, removed_count

# ------------------------------------------------------------------------------
# Function: process_dive_vehicle_rows_oct
# ------------------------------------------------------------------------------
def process_dive_vehicle_rows_oct(dive_info, oct_data):
	"""
	Filters the OCT data to the dive window defined by [Launch Time, Recovery Time],
	and deduplicates fixes by rounding timestamps.
	"""
	dive_id = str(dive_info["dive"]).strip()
	launch = dive_info["Launch Time"]
	recovery = dive_info["Recovery Time"]

	if pd.isnull(launch) or pd.isnull(recovery):
		return pd.DataFrame(), 0, 0, 0, 0

	expected_seconds = int((recovery - launch).total_seconds())
	if expected_seconds < 0:
		return pd.DataFrame(), 0, 0, 0, 0

	df_sub = oct_data[
		(oct_data["Timestamp"] >= launch) &
		(oct_data["Timestamp"] <= recovery)
		].copy()

	if df_sub.empty:
		return pd.DataFrame(), 0, 0, 0, expected_seconds

	df_rounded, orig_count, final_count, duplicates = preserve_closest_fix_per_second(df_sub)
	return df_rounded, orig_count, final_count, duplicates, expected_seconds

# ------------------------------------------------------------------------------
# Function: output_dive_csv_oct
# ------------------------------------------------------------------------------
def output_dive_csv_oct(root_dir, expedition_name, dive_id, df):
	"""
	Saves the OCT data (pitch/roll/heading) to a CSV file in <root_dir>/RUMI_processed/<dive_id>/.
	Performs a final check for duplicate timestamps before writing.
	"""
	if df.empty:
		return None

	root_dir = Path(root_dir)  # Convert to Path if it's a string
	outdir = root_dir / "RUMI_processed" / dive_id
	outdir.mkdir(parents=True, exist_ok=True)

	# Final check for duplicates
	df_final, dupes_removed = remove_timestamp_duplicates(df)
	if dupes_removed > 0:
		print(f"  - Final duplicate check: Removed {dupes_removed} duplicate timestamps")

	fname = f"{expedition_name}_{dive_id}_pitch_roll_heading_octans.csv"
	outpath = outdir / fname
	df_final.to_csv(outpath, index=False)
	print(f"Saved OCT data to: {outpath}")
	return outpath

# ------------------------------------------------------------------------------
# Function: process_dive_vehicle_rows_latlong
# ------------------------------------------------------------------------------
def process_dive_vehicle_rows_latlong(dive_info, vfr_data):
	"""
	Filters VFR (lat/long) data to the window defined by [On Bottom Time, Off Bottom Time]
	and deduplicates fixes.
	"""
	dive_id = str(dive_info["dive"]).strip()
	on_bottom = dive_info.get("On Bottom Time", None)
	off_bottom = dive_info.get("Off Bottom Time", None)

	if pd.isnull(on_bottom) or pd.isnull(off_bottom):
		return pd.DataFrame(), 0, 0, 0, 0

	if off_bottom < on_bottom:
		return pd.DataFrame(), 0, 0, 0, 0

	expected_seconds = int((off_bottom - on_bottom).total_seconds())
	df_sub = vfr_data[
		(vfr_data["Timestamp"] >= on_bottom) &
		(vfr_data["Timestamp"] <= off_bottom)
		].copy()

	if df_sub.empty:
		return pd.DataFrame(), 0, 0, 0, expected_seconds

	df_rounded, orig_count, final_count, duplicates = preserve_closest_fix_per_second(df_sub)
	return df_rounded, orig_count, final_count, duplicates, expected_seconds

# ------------------------------------------------------------------------------
# Function: output_dive_csv_latlong
# ------------------------------------------------------------------------------
def output_dive_csv_latlong(root_dir, expedition_name, dive_id, df):
	"""
	Saves DVL lat/long data to a CSV file in <root_dir>/RUMI_processed/<dive_id>/.
	The output columns are ordered: Timestamp, Latitude, Longitude.
	Performs a final check for duplicate timestamps before writing.
	"""
	if df.empty:
		return None

	root_dir = Path(root_dir)  # Convert to Path if it's a string
	outdir = root_dir / "RUMI_processed" / dive_id
	outdir.mkdir(parents=True, exist_ok=True)

	desired_cols = ["Timestamp", "Latitude", "Longitude"]
	for col in desired_cols:
		if col not in df.columns:
			return None
	df = df[desired_cols]

	# Final check for duplicates
	df_final, dupes_removed = remove_timestamp_duplicates(df)
	if dupes_removed > 0:
		print(f"  - Final duplicate check: Removed {dupes_removed} duplicate timestamps")

	fname = f"{expedition_name}_{dive_id}_dvl_lat_long.csv"
	outpath = outdir / fname
	df_final.to_csv(outpath, index=False)
	print(f"Saved DVL lat/long data to: {outpath}")
	return outpath

# ------------------------------------------------------------------------------
# Main process_data
# ------------------------------------------------------------------------------
def process_data(root_dir):
	"""
	Processes the .DAT files using dive time windows from the existing dive summaries CSV.
	Assumes dive summaries are available at <root_dir>/RUMI_processed/all_dive_summaries.csv.

	For each dive summary:
	  - OCT data is filtered using Launch Time and Recovery Time.
	  - VFR (DVL lat/long) data is filtered using On Bottom Time and Off Bottom Time.

	Results are saved as CSVs in <root_dir>/RUMI_processed/<dive_id>/.
	"""
	root_dir = Path(root_dir)  # Convert to Path if it's a string
	summary_path = root_dir / "RUMI_processed" / "all_dive_summaries.csv"

	if not summary_path.exists():
		print(f"Error: Dive summary file not found at {summary_path}")
		return

	try:
		ds = pd.read_csv(summary_path)
		ds["Launch Time"] = pd.to_datetime(ds["Launch Time"], utc=True, errors="coerce")
		ds["Recovery Time"] = pd.to_datetime(ds["Recovery Time"], utc=True, errors="coerce")
		if "On Bottom Time" in ds.columns:
			ds["On Bottom Time"] = pd.to_datetime(ds["On Bottom Time"], utc=True, errors="coerce")
		if "Off Bottom Time" in ds.columns:
			ds["Off Bottom Time"] = pd.to_datetime(ds["Off Bottom Time"], utc=True, errors="coerce")
	except Exception as e:
		print(f"Error reading dive summaries: {e}")
		return

	print("\nReading all .DAT files once to capture both OCT and VFR data...")
	all_oct, all_vfr = process_all_dat_files_both(root_dir)
	if all_oct.empty and all_vfr.empty:
		print("No OCT or VFR data found in any .DAT file.")
		return

	print("\n=== Processing Dives ===")
	total_oct_fixes = 0
	total_oct_expected = 0
	total_vfr_fixes = 0
	total_vfr_expected = 0

	for _, row in ds.iterrows():
		expedition = str(row.get("expedition", "NA")).strip()
		dive_id = str(row.get("dive", "UNKNOWN")).strip()

		df_oct, orig_oct, final_oct, dup_oct, exp_oct = process_dive_vehicle_rows_oct(row, all_oct)
		total_oct_expected += exp_oct
		if not df_oct.empty:
			output_dive_csv_oct(root_dir, expedition, dive_id, df_oct)
			total_oct_fixes += final_oct
			coverage_oct = (final_oct / exp_oct * 100) if exp_oct else 0
			print(f"\n(OCT) Dive {dive_id} Summary:")
			print(f"  - Duration: {exp_oct} seconds")
			print(f"  - Original Fixes: {orig_oct}")
			print(f"  - After Rounding: {final_oct}")
			print(f"  - Duplicates Removed: {dup_oct}")
			print(f"  - Coverage: {coverage_oct:.2f}%")
		else:
			print(f"⚠️ Dive {dive_id}: No OCT data within the defined window.")

		df_vfr, orig_vfr, final_vfr, dup_vfr, exp_vfr = process_dive_vehicle_rows_latlong(row, all_vfr)
		total_vfr_expected += exp_vfr
		if not df_vfr.empty:
			output_dive_csv_latlong(root_dir, expedition, dive_id, df_vfr)
			total_vfr_fixes += final_vfr
			coverage_vfr = (final_vfr / exp_vfr * 100) if exp_vfr else 0
			print(f"\n(LAT/LONG) Dive {dive_id} Summary:")
			print(f"  - Duration: {exp_vfr} seconds")
			print(f"  - Original Fixes: {orig_vfr}")
			print(f"  - After Rounding: {final_vfr}")
			print(f"  - Duplicates Removed: {dup_vfr}")
			print(f"  - Coverage: {coverage_vfr:.2f}%")
		else:
			print(f"⚠️ Dive {dive_id}: No VFR data within the On Bottom/Off Bottom window.")

	print("\n=== Processing Complete! ===")
	print(f"• OCT total expected: {total_oct_expected} seconds, total fixes: {total_oct_fixes}")
	print(f"• VFR total expected: {total_vfr_expected} seconds, total fixes: {total_vfr_fixes}")
	print(f"Data stored in {root_dir / 'RUMI_processed'}")

# ------------------------------------------------------------------------------
# End of Script
# ------------------------------------------------------------------------------