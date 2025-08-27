#!/usr/bin/env python3
"""
Expedition Data Processing Orchestrator (v2) — fixed pathing

Key change:
- Processed output directory is now always <root_dir>/RUMI_processed to match
  current processors which only accept `root_dir` and write relative to it.
"""

import logging
from pathlib import Path

# Third-party / stdlib
import pandas as pd  # noqa: F401  (kept if processors import relies on pandas presence)

# Processor imports
import processors.dive_summaries as dive_summaries
import processors.process_dat as process_dat
import processors.lat_long_uncertainty_USBL_sdyn as sdyn_usbl
import processors.sensors_sealog as sensors_sealog
import processors.stillcam_images as stillcam_images

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')


def get_directories():
    """
    Ask for the raw data directory (root_dir).
    The processed directory is fixed to <root_dir>/RUMI_processed.
    """
    logging.debug("Entered get_directories()")
    default_dir = Path("Z:/NA173")
    logging.debug(f"Default directory set to: {default_dir}")

    print("Where is your raw data located?")
    raw_input_val = input(f"Enter the path to the directory containing raw data [default: {default_dir}]: ").strip()
    logging.debug(f"User input for raw data directory: '{raw_input_val}'")
    root_dir = Path(raw_input_val) if raw_input_val else default_dir
    logging.debug(f"Using raw data directory: {root_dir}")

    while not root_dir.is_dir():
        logging.debug(f"Directory '{root_dir}' does not exist.")
        print(f"Error: The directory '{root_dir}' does not exist. Please try again.")
        raw_input_val = input(f"Enter the path to the directory containing raw data [default: {default_dir}]: ").strip()
        logging.debug(f"User re-input for raw data directory: '{raw_input_val}'")
        root_dir = Path(raw_input_val) if raw_input_val else default_dir

    # Processed dir is always under root_dir to match current processors' behavior
    processed_dir = root_dir / "RUMI_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Processed data directory created or verified: {processed_dir}")

    print(f"\n  • Raw data directory:     {root_dir}")
    print(f"  • Processed data folder:  {processed_dir}")

    logging.debug("Exiting get_directories()")
    return root_dir, processed_dir


def process_script(script_module, root_dir):
    """
    Prompt to run script_module.process_data(root_dir).
    """
    logging.debug(f"Processing module: {script_module.__name__}")
    module_name = script_module.__name__.split('.')[-1] if '.' in script_module.__name__ else script_module.__name__
    proceed = input(f"Do you want to process {module_name}? (yes/no): ").strip().lower()
    logging.debug(f"User response for {module_name}: '{proceed}'")
    if proceed == "yes":
        print(f"\nProcessing {module_name}...")
        logging.debug(f"Calling process_data() for {module_name}")
        script_module.process_data(root_dir)
        logging.debug(f"Finished process_data() for {module_name}")
        print(f"Finished processing {module_name}.")
    else:
        print(f"Skipping {module_name}.")
        logging.debug(f"Skipped processing for {module_name}")
    return proceed


def main():
    logging.debug("Starting main()")
    print("--------------------------------------------------")
    print("     Expedition Data Processing Orchestrator (v2) ")
    print("--------------------------------------------------")

    # 1) Get directories
    root_dir, processed_dir = get_directories()
    logging.debug(f"Directories set. Root: {root_dir}, Processed: {processed_dir}")

    # 2) Dive summaries
    print("\n[ Step 1 ]: Dive Summaries")
    dive_summaries_proceed = process_script(dive_summaries, root_dir)
    logging.debug(f"Dive summaries processing decision: {dive_summaries_proceed}")

    # Verify presence of 'all_dive_summaries.csv' where the module writes it
    summary_file = processed_dir / "all_dive_summaries.csv"
    logging.debug(f"Checking for existence of summary file: {summary_file}")
    if not summary_file.exists():
        if dive_summaries_proceed == "no":
            print(f"\nError: {summary_file} does not exist, and dive summaries were skipped.")
            print("Cannot continue with .DAT processing because the summary file is missing.")
            logging.debug("Aborting further processing due to missing dive summaries file (skipped).")
            return
        else:
            print(f"\nError: {summary_file} was not created after processing dive summaries.")
            print("Cannot continue with .DAT processing.")
            logging.debug("Aborting further processing due to missing dive summaries file (processing attempted).")
            return
    else:
        print(f"\nDive summaries are present at: {summary_file}")
        logging.debug("Verified existence of dive summaries file.")

    # 3) Combined .DAT
    print("\n[ Step 2 ]: Combined .DAT Processing (OCT + VFR)")
    process_script(process_dat, root_dir)

    # 4) USBL
    print("\n[ Step 3 ]: USBL Lat/Long Uncertainty")
    process_script(sdyn_usbl, root_dir)

    # 5) Sealog
    print("\n[ Step 4 ]: Sealog Sensor Data")
    process_script(sensors_sealog, root_dir)

    # 6) StillCam conversion
    print("\n[ Step 5 ]: Convert StillCam PNGs to JPGs")
    process_script(stillcam_images, root_dir)

    print("\n--------------------------------------------------")
    print("All selected processes completed.")
    print(f"Check '{processed_dir}' for output files.")
    print("--------------------------------------------------")
    logging.debug("Exiting main()")


if __name__ == "__main__":
    logging.debug("Script started")
    main()
    logging.debug("Script finished")
