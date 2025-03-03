import os
import re
import csv
import pandas as pd
from datetime import datetime, timedelta, timezone

import processors.dive_summaries as dive_summaries
import processors.dat_dvl as dat_dvl
import processors.sdyn_usbl as sdyn_usbl
import processors.sensors_sealog as sensors_sealog


def get_root_directory():
    """Prompt user for root data directory and set up output structure."""
    root_dir = input("Enter the root directory for raw data: ").strip()

    if not os.path.isdir(root_dir):
        print(f"Error: The directory '{root_dir}' does not exist.")
        return get_root_directory()

    processed_dir = os.path.join(root_dir, "RUMI_processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Processed data will be stored in: {processed_dir}")
    return root_dir, processed_dir


def process_script(script_module, dive_name, root_dir):
    """Runs a processing script, ensuring it looks in the correct data location."""
    proceed = input(f"Do you want to process {script_module.__name__}? (yes/no): ").strip().lower()
    if proceed == "yes":
        print(f"Processing {script_module.__name__}...")
        script_module.process_data(root_dir)  # ✅ Pass root directory instead of output directory
        print(f"Finished processing {script_module.__name__}. Data stored in {root_dir}/RUMI_processed/{dive_name}.\n")
    else:
        print(f"Skipping {script_module.__name__}.")

    return proceed  # ✅ Return whether the user chose to process this module


def main():
    """Main execution flow for data processing."""
    root_dir, processed_dir = get_root_directory()

    # ✅ Ask the user before processing dive summaries
    dive_summaries_processed = process_script(dive_summaries, "dive_summaries", root_dir)

    # ✅ Check for `all_dive_summaries.csv` in `RUMI_processed/`
    summary_file = os.path.join(root_dir, "RUMI_processed", "all_dive_summaries.csv")
    if not os.path.exists(summary_file):
        if dive_summaries_processed == "no":
            print(f"Error: {summary_file} is missing, and dive summaries were skipped. Cannot continue processing.")
            return
        else:
            print(f"Error: {summary_file} was not created after processing dive summaries. Cannot continue processing.")
            return

    process_script(dat_dvl, "all_dives", root_dir)
    process_script(sdyn_usbl, "all_dives", root_dir)
    process_script(sensors_sealog, "all_dives", root_dir)

    print("All selected processes completed.")

    print("All selected processes completed.")


if __name__ == "__main__":
    main()
