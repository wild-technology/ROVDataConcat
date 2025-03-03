#!/usr/bin/env python3
"""
Expedition Data Processing Orchestrator (v2)

This script serves as an interactive driver for all major expedition data-processing steps.
It guides the user through specifying:
  1) A directory containing raw data
  2) A directory in which to create 'RUMI_processed' for output

Then, it sequentially prompts the user whether or not to run each processing module:
  1) Dive Summaries
  2) Combined .DAT Parsing (OCT + VFR)
  3) USBL Lat/Long Uncertainty
  4) Sealog Sensor Data
  5) StillCam Image Conversion (PNG → JPG)

Each module has a `process_data(root_dir)` function, and is imported from the `processors` package.
Users can select "yes" or "no" for each module. If a critical file (`all_dive_summaries.csv`)
is missing and the user opts to skip generating it, the script will abort further steps that depend on it.

Usage:
  python main.py

Example flow:
  1) Prompt: "Where is your raw data located?"
     -> user enters path to a folder containing raw data (or press Enter to default)
  2) Prompt: "Where should 'RUMI_processed' be created?"
     -> user selects or creates an output folder (or press Enter to default)
  3) Module prompts:
     a) "Do you want to process dive_summaries? (yes/no)"
     b) "Do you want to process process_dat? (yes/no)"
     c) etc.
  4) The script either runs or skips each processing step.
  5) Final summary message.

Author: Jonathan Fiely
Date: 20 Feb 2025
"""

import os
import re
import csv
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Processor imports
import processors.dive_summaries as dive_summaries
import processors.process_dat as process_dat
import processors.lat_long_uncertainty_USBL_sdyn as sdyn_usbl
import processors.sensors_sealog as sensors_sealog
import processors.stillcam_images as stillcam_images

def get_directories():
    """
    Interactively asks the user to specify two directories:
      (1) The directory containing raw data
      (2) The parent directory where the 'RUMI_processed' folder will be created

    Defaults are provided if the user inputs nothing.
    Default: "E:/RUMI/NAUTILUS-CRUISE-COPY2/NA156"

    This function:
      - Ensures the raw data directory exists
      - Ensures (or creates) the parent directory for 'RUMI_processed'
      - Builds the path <parent_dir>/RUMI_processed
      - Prints a summary of these chosen paths
      - Returns (root_dir, processed_dir) as a tuple

    Returns
    -------
    root_dir : Path
        The validated path to the user's raw data directory.
    processed_dir : Path
        The path to the newly created or existing 'RUMI_processed' directory.
    """
    logging.debug("Entered get_directories()")
    # Use pathlib for consistent path handling
    default_dir = Path("E:/RUMI/NAUTILUS-CRUISE-COPY2/NA156")
    logging.debug(f"Default directory set to: {default_dir}")

    # Prompt for the raw data directory (root_dir)
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

    # Prompt for the parent directory where we will create 'RUMI_processed'
    print("\nWhere should the 'RUMI_processed' folder be created for storing processed data?")
    parent_input = input(f"Enter the directory where 'RUMI_processed' will be placed [default: {default_dir}]: ").strip()
    logging.debug(f"User input for parent directory: '{parent_input}'")
    parent_dir = Path(parent_input) if parent_input else default_dir
    logging.debug(f"Using parent directory: {parent_dir}")

    # Even if parent_dir doesn't exist, we attempt to create it
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Ensured parent directory exists: {parent_dir}")
    except Exception as e:
        logging.error(f"Error creating '{parent_dir}': {e}")
        print(f"Error creating '{parent_dir}': {e}\nPlease try a different path.")
        exit(1)

    # Construct the path: <parent_dir>/RUMI_processed using pathlib
    processed_dir = parent_dir / "RUMI_processed"
    processed_dir.mkdir(exist_ok=True)
    logging.debug(f"Processed data directory created or verified: {processed_dir}")

    print(f"\n  • Raw data directory:     {root_dir}")
    print(f"  • Processed data folder: {processed_dir}")

    logging.debug("Exiting get_directories()")
    return root_dir, processed_dir


def process_script(script_module, root_dir):
    """
    Prompts whether to run the `process_data(root_dir)` function from a given script module.

    Parameters
    ----------
    script_module : module
        The module containing a 'process_data(root_dir)' function.
    root_dir : Path
        The path to the user-specified root directory for raw data.

    Returns
    -------
    proceed : str
        "yes" if the user chooses to run the script module's process_data;
        "no" if the user chooses to skip.
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
    """
    Main execution function for the Expedition Data Processing Orchestrator.

    Flow:
      1) Prompt user to specify:
         - A valid raw data directory
         - A parent directory for storing 'RUMI_processed'
      2) Retrieve these paths as (root_dir, processed_dir).
      3) Prompt whether to run each processor in sequence:
         a) Dive Summaries
         b) Combined .DAT Parsing (OCT + VFR)
         c) USBL Lat/Long Uncertainty
         d) Sealog Sensor Data
         e) StillCam Image Conversion (PNG → JPG)
      4) If the user skips Dive Summaries and 'all_dive_summaries.csv'
         is missing, the script will abort subsequent steps that depend on it.
      5) Final summary message indicates all selected processing steps are done.

    Requirements:
      - Must be run in an environment where `input()` can read from stdin (i.e., interactive).
      - The `processors` package must contain each of the modules with a
        `process_data(root_dir)` function.

    Returns
    -------
    None
        Prints progress messages and completes interactive data processing.
    """
    logging.debug("Starting main()")
    print("--------------------------------------------------")
    print("     Expedition Data Processing Orchestrator (v2) ")
    print("--------------------------------------------------")

    # 1) Get directories
    root_dir, processed_dir = get_directories()
    logging.debug(f"Directories set. Root: {root_dir}, Processed: {processed_dir}")

    # 2) Process dive summaries
    print("\n[ Step 1 ]: Dive Summaries")
    dive_summaries_proceed = process_script(dive_summaries, root_dir)
    logging.debug(f"Dive summaries processing decision: {dive_summaries_proceed}")

    # Verify the presence of 'all_dive_summaries.csv'
    summary_file = processed_dir / "all_dive_summaries.csv"
    logging.debug(f"Checking for existence of summary file: {summary_file}")
    if not summary_file.exists():
        if dive_summaries_proceed == "no":
            # If user skipped dive summaries but the summary file isn't there,
            # further steps that rely on this file cannot proceed safely.
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

    # 3) Process combined .DAT (OCT + VFR)
    print("\n[ Step 2 ]: Combined .DAT Processing (OCT + VFR)")
    process_script(process_dat, root_dir)

    # 4) Process USBL data
    print("\n[ Step 3 ]: USBL Lat/Long Uncertainty")
    process_script(sdyn_usbl, root_dir)

    # 5) Process Sealog sensor data
    print("\n[ Step 4 ]: Sealog Sensor Data")
    process_script(sensors_sealog, root_dir)

    # 6) Convert StillCam PNGs to JPGs
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