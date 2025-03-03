from pathlib import Path
import sys
import importlib

def get_directories():
	"""
    Prompts user for the root directory, dive folder to process,
    and the processed output directory.
    """
	# Set the default root directory
	default_root = Path("E:/RUMI/NAUTILUS-CRUISE-COPY2/NA156")
	root_input = input(f"Enter the root directory [default: {default_root}]: ").strip()
	root_dir = Path(root_input) if root_input else default_root

	while not root_dir.is_dir():
		print(f"Error: The directory '{root_dir}' does not exist. Please try again.")
		root_input = input(f"Enter the root directory [default: {default_root}]: ").strip()
		root_dir = Path(root_input) if root_input else default_root

	# Ask for the dive folder (e.g., H2021)
	dive = input("Enter the dive folder to process (e.g., H2021): ").strip()
	while not dive:
		print("Error: Dive folder cannot be empty.")
		dive = input("Enter the dive folder to process (e.g., H2021): ").strip()

	# Construct the raw data directory based on the dive folder.
	raw_dir = root_dir / "RUMI_processed" / dive
	if not raw_dir.is_dir():
		print(f"Error: The raw data directory '{raw_dir}' does not exist.")
		sys.exit(1)

	# Set the default processed directory.
	processed_default = root_dir / "RUMI_processed" / dive
	processed_input = input(f"Enter the directory for processed data [default: {processed_default}]: ").strip()
	processed_dir = Path(processed_input) if processed_input else processed_default
	processed_dir.mkdir(parents=True, exist_ok=True)

	print(f"\n  • Raw data directory: {raw_dir}")
	print(f"  • Processed data directory: {processed_dir}")
	return raw_dir, processed_dir

def process_module(module_name, raw_dir, processed_dir):
	"""
    Prompts user to run a specific processing module.

    Parameters
    ----------
    module_name : str
        Name of the module to process
    raw_dir : Path
        Directory containing raw data
    processed_dir : Path
        Directory where processed data will be saved
    """
	try:
		module = importlib.import_module(f"processors.{module_name}")
		# Add a special note for UTM assessment module.
		if module_name == "utm_assessment":
			print(
				"\nNOTE: The UTM Assessment step will offset the vehicle's location and save a final data file for upload in Unreal.")
		proceed = input(f"Do you want to process {module_name}? (yes/no): ").strip().lower()
		if proceed == "yes":
			print(f"\nProcessing {module_name}...")
			module.process_data(raw_dir, processed_dir)
			print(f"Finished processing {module_name}.")
		else:
			print(f"Skipping {module_name}.")
	except ImportError as e:
		print(f"Error importing {module_name}: {e}")
		sys.exit(1)

def main():
	"""
    Main script orchestrating the expedition data processing.
    """
	print("--------------------------------------------------")
	print("     KALMAN FILTER DATA PROCESSING SCRIPT         ")
	print("--------------------------------------------------")

	raw_dir, processed_dir = get_directories()

	# Process modules in the desired order (skipping Nav_depth_offset).
	modules = [
		"kalman_concat",
		"kalman_filter",
		"kalman_assess",
		"utm_assessment"
	]

	for module in modules:
		process_module(module, raw_dir, processed_dir)

	print("\n--------------------------------------------------")
	print("All selected processes completed.")
	print(f"Check '{processed_dir}' for output files.")
	print("--------------------------------------------------")

if __name__ == "__main__":
	main()