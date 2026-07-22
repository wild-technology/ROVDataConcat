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

# Restart support: the file each stage produces, used to detect completed work.
MODULE_OUTPUTS = {
    "kalman_concat": "{exp}_{dive}_filtered_datatable.csv",
    "kalman_filter": "{exp}_{dive}_final_datatable.csv",
    "kalman_assess": "{exp}_{dive}_kalman_assessment.csv",
    "kalman_offset": "{exp}_{dive}_filtered_offset_final.csv",
}


def module_output_path(module_name, processed_dir):
    dive = processed_dir.name
    exp = processed_dir.parent.parent.name
    pattern = MODULE_OUTPUTS.get(module_name)
    return processed_dir / pattern.format(exp=exp, dive=dive) if pattern else None


def process_module(module_name, raw_dir, processed_dir, auto_yes=False, force=False):
    """
    Runs one processing module, with resume support: when the module's output
    already exists it is skipped (interactive mode asks; --yes mode skips
    automatically unless --force is given).
    Returns 'ok', 'skipped', or 'failed'.
    """
    try:
        module = importlib.import_module(f"processors.{module_name}")
    except ImportError as e:
        print(f"Error importing processors.{module_name}: {e}")
        sys.exit(1)

    out_path = module_output_path(module_name, processed_dir)
    already_done = out_path is not None and out_path.exists()

    if already_done and not force:
        if auto_yes:
            print(f"\n[resume] {module_name}: output {out_path.name} already exists -- "
                  f"skipping (use --force to regenerate).")
            return "skipped"
        redo = input(f"{module_name}: output {out_path.name} already exists. "
                     f"Reprocess it? (yes/no): ").strip().lower()
        if redo != "yes":
            print(f"Skipping {module_name} (output kept).")
            return "skipped"
    elif not auto_yes and not already_done:
        proceed = input(f"Do you want to process {module_name}? (yes/no): ").strip().lower()
        if proceed != "yes":
            print(f"Skipping {module_name}.")
            return "skipped"

    # Add a special note for the UTM assessment module.
    if module_name == "kalman_offset":
        print(
            "\nNOTE: The offset Assessment step will offset the vehicle's location and save a final data file for upload in Unreal."
        )

    print(f"\nProcessing {module_name}...")
    try:
        module.process_data(raw_dir, processed_dir)
    except Exception as e:
        print(f"\nERROR in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return "failed"
    print(f"Finished processing {module_name}.")
    return "ok"


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
    parser.add_argument("--force", action="store_true",
                        help="Rerun modules even when their outputs already exist")
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

    statuses = {}
    for module in modules:
        status = process_module(module, raw_dir, processed_dir,
                                auto_yes=args.yes, force=args.force)
        statuses[module] = status
        if status == "failed":
            print(f"\nAborting: {module} failed; downstream modules depend on its output.")
            break

    print("\n--------------------------------------------------")
    print("Run summary:")
    for name, status in statuses.items():
        marker = {"ok": "done", "skipped": "skipped", "failed": "FAILED"}[status]
        print(f"  {name:16s} {marker}")
    if any(s == "failed" for s in statuses.values()):
        print("Fix the issue and rerun -- completed modules will be skipped")
        print("automatically (use --force to redo them).")
    else:
        print(f"Check '{processed_dir}' for output files.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()


