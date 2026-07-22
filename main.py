#!/usr/bin/env python3
"""
Expedition Data Processing Orchestrator (v2) -- fixed pathing, non-interactive

Runs all modules automatically, aborts on first error.
Processed output directory: <root_dir>/RUMI_processed
"""

import argparse
import logging
from pathlib import Path

# Processor imports
import processors.dive_summaries as dive_summaries
import processors.process_dat as process_dat
import processors.usbl_sdyn as sdyn_usbl
import processors.sensors_sealog as sensors_sealog
import processors.stillcam_images as stillcam_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

DEFAULT_RAW_DIR = Path("Z:/NA173")


def get_directories(cli_dir=None):
    """
    Resolve the raw data directory (root_dir), from the CLI argument if given,
    otherwise interactively. The processed directory is fixed to
    <root_dir>/RUMI_processed.
    """
    logging.debug("Entered get_directories()")
    default_dir = DEFAULT_RAW_DIR

    if cli_dir is not None:
        root_dir = Path(cli_dir)
        if not root_dir.is_dir():
            raise SystemExit(f"Error: The directory '{root_dir}' does not exist.")
    else:
        print("Where is your raw data located?")
        raw_input_val = input(f"Enter the path to the directory containing raw data [default: {default_dir}]: ").strip()
        root_dir = Path(raw_input_val) if raw_input_val else default_dir

    while not root_dir.is_dir():
        logging.debug(f"Directory '{root_dir}' does not exist.")
        print(f"Error: The directory '{root_dir}' does not exist. Please try again.")
        raw_input_val = input(f"Enter the path to the directory containing raw data [default: {default_dir}]: ").strip()
        logging.debug(f"User re-input for raw data directory: '{raw_input_val}'")
        root_dir = Path(raw_input_val) if raw_input_val else default_dir

    processed_dir = root_dir / "RUMI_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Processed data directory created or verified: {processed_dir}")

    print(f"\n  * Raw data directory:     {root_dir}")
    print(f"  * Processed data folder:  {processed_dir}")

    logging.debug("Exiting get_directories()")
    return root_dir, processed_dir


# Restart support: how to tell whether a step already produced its outputs.
# Each entry maps a step name to a glob (relative to <root>/RUMI_processed)
# that matches at least one of its output files.
STEP_OUTPUT_GLOBS = {
    "dive_summaries": "all_dive_summaries.csv",
    "process_dat": "*/[!.]*_pitch_roll_heading_octans.csv",
    "usbl_sdyn": "*/[!.]*_USBL_Hercules.csv",
    "sensors_sealog": "*/[!.]*_sealog_sensors_merged.csv",
    # stillcam_images resumes per image internally -- always run it.
}


def step_outputs_exist(name, root_dir):
    pattern = STEP_OUTPUT_GLOBS.get(name)
    if pattern is None:
        return False
    return any((Path(root_dir) / "RUMI_processed").glob(pattern))


def run_step(script_module, root_dir, force=False) -> str:
    """
    Run script_module.process_data(root_dir).
    Returns 'ok', 'skipped', or 'failed'.
    """
    name = script_module.__name__.split('.')[-1]
    if not force and step_outputs_exist(name, root_dir):
        print(f"\n[resume] {name}: outputs already present -- skipping "
              f"(rerun with --force to regenerate).")
        return "skipped"
    logging.debug(f"Starting {name}")
    print(f"\nProcessing {name}...")
    try:
        script_module.process_data(root_dir)
        print(f"Finished {name}.")
        logging.debug(f"Finished {name}")
        return "ok"
    except Exception as e:
        logging.exception(f"{name} failed")
        print(f"\nERROR in {name}: {e}\nAborting subsequent steps.")
        return "failed"


def main():
    parser = argparse.ArgumentParser(description="ROV raw data extraction pipeline (stage 1)")
    parser.add_argument("--dir", help="Raw data root directory (skips the interactive prompt)")
    parser.add_argument("--force", action="store_true",
                        help="Rerun every step even if its outputs already exist")
    args = parser.parse_args()

    logging.debug("Starting main()")
    print("--------------------------------------------------")
    print("     Expedition Data Processing Orchestrator (v2) ")
    print("--------------------------------------------------")

    # 1) Get directories
    root_dir, processed_dir = get_directories(args.dir)
    logging.debug(f"Directories set. Root: {root_dir}, Processed: {processed_dir}")

    steps = [
        ("Dive Summaries", dive_summaries),
        ("Combined .DAT Processing (OCT + VFR)", process_dat),
        ("USBL Lat/Long Uncertainty", sdyn_usbl),
        ("Sealog Sensor Data", sensors_sealog),
        ("Convert StillCam PNGs to JPGs", stillcam_images),
    ]

    statuses = {}
    for idx, (title, module) in enumerate(steps, start=1):
        print(f"\n[ Step {idx} ]: {title}")
        status = run_step(module, root_dir, force=args.force)
        statuses[module.__name__.split('.')[-1]] = status
        if status == "failed":
            break

        # Downstream steps need the dive summaries file regardless of whether
        # the step ran or was skipped on resume.
        if module is dive_summaries:
            summary_file = processed_dir / "all_dive_summaries.csv"
            if not summary_file.exists():
                print(f"\nError: {summary_file} was not created. "
                      f"Cannot continue with .DAT processing.")
                statuses[module.__name__.split('.')[-1]] = "failed"
                break
            print(f"\nDive summaries present: {summary_file}")

    print("\n--------------------------------------------------")
    print("Run summary:")
    for name, status in statuses.items():
        marker = {"ok": "done", "skipped": "skipped (resume)", "failed": "FAILED"}[status]
        print(f"  {name:20s} {marker}")
    if any(s == "failed" for s in statuses.values()):
        print("Pipeline aborted at the failed step. Fix the issue and rerun --")
        print("completed steps will be skipped automatically (use --force to redo).")
    else:
        print(f"Outputs in: '{processed_dir}'")
    print("--------------------------------------------------")
    logging.debug("Exiting main()")


if __name__ == "__main__":
    logging.debug("Script started")
    main()
    logging.debug("Script finished")
