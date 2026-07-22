from pathlib import Path
import argparse
import sys
import importlib

def prompt_directory(prompt, default=None, must_exist=True):
    """
    Prompts the user for a directory, using a default if no input is given.
    If must_exist is True, the prompt repeats until the provided directory exists.
    """
    while True:
        if default is not None:
            response = input(f"{prompt} [default: {default}]: ").strip()
            path = Path(response) if response else Path(default)
        else:
            response = input(f"{prompt}: ").strip()
            path = Path(response)
        if must_exist and not path.is_dir():
            print(f"Error: The directory '{path}' does not exist. Please try again.")
        else:
            return path.resolve()

def get_directories(args):
    """Resolve base, expedition, dive (from CLI args or prompts); use
    RUMI_processed as canonical root.
    Structure:
      processed_dir = <base>/<EXPEDITION>/RUMI_processed/<DIVE>
      raw_dir = processed_dir  # inputs live here as well
    """
    default_base = Path("Z:/")
    if args.base:
        base_dir = Path(args.base)
        if not base_dir.is_dir():
            sys.exit(f"Error: The base directory '{base_dir}' does not exist.")
    else:
        base_dir = prompt_directory("Enter the base directory containing expeditions", default_base)

    expedition = (args.expedition or "").strip()
    while not expedition:
        expedition = input("Enter the expedition (e.g., NA173): ").strip()
        if not expedition:
            print("Error: Expedition cannot be empty.")

    dive = (args.dive or "").strip()
    while not dive:
        dive = input("Enter the dive folder (e.g., H2075): ").strip()
        if not dive:
            print("Error: Dive folder cannot be empty.")

    processed_dir = (base_dir / expedition / "RUMI_processed" / dive).resolve()
    if not processed_dir.is_dir():
        print(f"Error: The dive folder '{processed_dir}' does not exist.")
        sys.exit(1)

    raw_dir = processed_dir  # keep module signatures, but point at processed_dir

    print(f"\n  * Expedition: {expedition}")
    print(f"  * Dive: {dive}")
    print(f"  * Data directory (raw_dir): {raw_dir}")
    print(f"  * Processed directory:      {processed_dir}")
    return raw_dir, processed_dir

def process_module(module_name, raw_dir, processed_dir, auto_yes=False):
    """
    Prompts the user to run a specific processing module.

    Parameters
    ----------
    module_name : str
        Name of the module to process (without the 'processes.' prefix).
    raw_dir : Path
        Directory containing raw data.
    processed_dir : Path
        Directory where processed data will be saved.
    """
    try:
        module = importlib.import_module(f"processors.{module_name}")

        # Add a special note for the UTM assessment module.
        if module_name == "kalman_offset":
            print(
                "\nNOTE: The offset Assessment step will offset the vehicle's location and save a final data file for upload in Unreal."
            )

        if auto_yes:
            proceed = "yes"
        else:
            proceed = input(f"Do you want to process {module_name}? (yes/no): ").strip().lower()
        if proceed == "yes":
            print(f"\nProcessing {module_name}...")
            module.process_data(raw_dir, processed_dir)
            print(f"Finished processing {module_name}.")
        else:
            print(f"Skipping {module_name}.")
    except ImportError as e:
        print(f"Error importing processes.{module_name}: {e}")
        sys.exit(1)


def main():
    """
    Main script orchestrating the expedition data processing.
    """
    parser = argparse.ArgumentParser(description="ROV Kalman filter pipeline (stage 2)")
    parser.add_argument("--base", help="Base directory containing expeditions (e.g. Z:/)")
    parser.add_argument("--expedition", help="Expedition identifier (e.g. NA173)")
    parser.add_argument("--dive", help="Dive folder (e.g. H2075)")
    parser.add_argument("--yes", action="store_true",
                        help="Run all modules without per-module confirmation")
    args = parser.parse_args()

    print("--------------------------------------------------")
    print("     KALMAN FILTER DATA PROCESSING SCRIPT         ")
    print("--------------------------------------------------")

    raw_dir, processed_dir = get_directories(args)

    # Process modules in the desired order.
    modules = [
        "kalman_concat",
        "kalman_filter",
        "kalman_assess",
        "kalman_offset"
    ]

    for module in modules:
        process_module(module, raw_dir, processed_dir, auto_yes=args.yes)

    print("\n--------------------------------------------------")
    print("All selected processes completed.")
    print(f"Check '{processed_dir}' for output files.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()


