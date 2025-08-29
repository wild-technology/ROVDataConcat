#!/usr/bin/env python3
"""
Expedition Data Processing Orchestrator (v2) — fixed pathing, non-interactive

Runs all modules automatically, aborts on first error.
Processed output directory: <root_dir>/RUMI_processed
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

    processed_dir = root_dir / "RUMI_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Processed data directory created or verified: {processed_dir}")

    print(f"\n  • Raw data directory:     {root_dir}")
    print(f"  • Processed data folder:  {processed_dir}")

    logging.debug("Exiting get_directories()")
    return root_dir, processed_dir


def run_step(script_module, root_dir) -> bool:
    """
    Non-interactive: run script_module.process_data(root_dir).
    Returns True on success, False on any exception (after logging).
    """
    name = script_module.__name__.split('.')[-1]
    logging.debug(f"Starting {name}")
    print(f"\nProcessing {name}...")
    try:
        script_module.process_data(root_dir)
        print(f"Finished {name}.")
        logging.debug(f"Finished {name}")
        return True
    except Exception as e:
        logging.exception(f"{name} failed")
        print(f"\nERROR in {name}: {e}\nAborting subsequent steps.")
        return False


def main():
    logging.debug("Starting main()")
    print("--------------------------------------------------")
    print("     Expedition Data Processing Orchestrator (v2) ")
    print("                 Non-interactive                  ")
    print("--------------------------------------------------")

    # 1) Get directories
    root_dir, processed_dir = get_directories()
    logging.debug(f"Directories set. Root: {root_dir}, Processed: {processed_dir}")

    # 2) Dive summaries
    print("\n[ Step 1 ]: Dive Summaries")
    if not run_step(dive_summaries, root_dir):
        return

    # Require 'all_dive_summaries.csv' for downstream steps
    summary_file = processed_dir / "all_dive_summaries.csv"
    logging.debug(f"Checking for existence of summary file: {summary_file}")
    if not summary_file.exists():
        print(f"\nError: {summary_file} was not created. Cannot continue with .DAT processing.")
        logging.debug("Aborting: missing dive summaries output.")
        return
    else:
        print(f"\nDive summaries present: {summary_file}")
        logging.debug("Verified existence of dive summaries file.")

    # 3) Combined .DAT
    print("\n[ Step 2 ]: Combined .DAT Processing (OCT + VFR)")
    if not run_step(process_dat, root_dir):
        return

    # 4) USBL
    print("\n[ Step 3 ]: USBL Lat/Long Uncertainty")
    if not run_step(sdyn_usbl, root_dir):
        return

    # 5) Sealog
    print("\n[ Step 4 ]: Sealog Sensor Data")
    if not run_step(sensors_sealog, root_dir):
        return

    # 6) StillCam conversion
    print("\n[ Step 5 ]: Convert StillCam PNGs to JPGs")
    if not run_step(stillcam_images, root_dir):
        return

    print("\n--------------------------------------------------")
    print("All processes completed.")
    print(f"Outputs in: '{processed_dir}'")
    print("--------------------------------------------------")
    logging.debug("Exiting main()")


if __name__ == "__main__":
    logging.debug("Script started")
    main()
    logging.debug("Script finished")
