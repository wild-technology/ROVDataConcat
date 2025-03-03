from pathlib import Path
import pandas as pd
import csv
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def process_data(raw_dir, processed_dir):
	"""
	Performs Kalman filter assessment on processed navigation data.

	Parameters
	----------
	raw_dir : Path or str
		Directory containing raw data
	processed_dir : Path or str
		Directory where processed data is saved and where output will be saved
	"""
	# Convert to Path objects if they're strings
	raw_dir = Path(raw_dir)
	processed_dir = Path(processed_dir)

	print("Running Kalman Assessment Process...")

	input_file = processed_dir / "kalman_filtered_data.csv"
	output_file = processed_dir / "kalman_assessment.csv"

	# Added low_memory=False to fix the DtypeWarning
	df = pd.read_csv(input_file, parse_dates=['Timestamp'], low_memory=False)

	# Define columns for analysis
	required_columns = ['Timestamp', 'Herc_Depth_1', 'Roll', 'Pitch', 'Heading', 'x_usbl', 'y_usbl',
						'kalman_depth', 'kalman_roll_deg', 'kalman_pitch_deg', 'kalman_yaw_deg',
						'kalman_x', 'kalman_y']
	missing_columns = [col for col in required_columns if col not in df.columns]

	if missing_columns:
		print(f"Warning: Missing required columns: {missing_columns}")

	# Function to compute smoothness
	def calculate_smoothness(data):
		return data.diff().dropna().std() if data is not None and len(data) > 1 else np.nan

	# Function to compute consistency
	def calculate_consistency(filtered, raw):
		min_len = min(len(filtered), len(raw))
		return np.abs(filtered[:min_len] - raw[:min_len]).mean() if filtered is not None and raw is not None else np.nan

	# Smoothness metrics
	smoothness_metrics = {
		'Depth': {'Raw': calculate_smoothness(df['Herc_Depth_1']),
				  'Filtered': calculate_smoothness(df['kalman_depth'])},
		'Roll': {'Raw': calculate_smoothness(df['Roll']), 'Filtered': calculate_smoothness(df['kalman_roll_deg'])},
		'Pitch': {'Raw': calculate_smoothness(df['Pitch']), 'Filtered': calculate_smoothness(df['kalman_pitch_deg'])},
		'Yaw': {'Raw': calculate_smoothness(df['Heading']), 'Filtered': calculate_smoothness(df['kalman_yaw_deg'])}
	}

	# Consistency metrics
	consistency_metrics = {
		'Depth': calculate_consistency(df['kalman_depth'], df['Herc_Depth_1']),
		'Roll': calculate_consistency(df['kalman_roll_deg'], df['Roll']),
		'Pitch': calculate_consistency(df['kalman_pitch_deg'], df['Pitch']),
		'Yaw': calculate_consistency(df['kalman_yaw_deg'], df['Heading']),
		'X': calculate_consistency(df['kalman_x'], df['x_usbl']),
		'Y': calculate_consistency(df['kalman_y'], df['y_usbl'])
	}

	# Save results
	assessment_results = []
	for param, values in smoothness_metrics.items():
		for data_type, value in values.items():
			assessment_results.append(["Smoothness", f"{param}_{data_type}", value])
	for param, value in consistency_metrics.items():
		assessment_results.append(["Consistency", param, value])

	assessment_df = pd.DataFrame(assessment_results, columns=['Metric', 'Parameter', 'Value'])
	assessment_df.to_csv(output_file, index=False)
	print(f"Kalman assessment results saved to: {output_file}")

	# Generate plots
	fig, axes = plt.subplots(3, 2, figsize=(12, 12))
	axes = axes.ravel()
	plot_configs = [
		('Herc_Depth_1', 'kalman_depth', 'Depth', 'Depth (m)', 0),
		('Roll', 'kalman_roll_deg', 'Roll', 'Roll (deg)', 1),
		('Pitch', 'kalman_pitch_deg', 'Pitch', 'Pitch (deg)', 2),
		('Heading', 'kalman_yaw_deg', 'Heading', 'Heading (deg)', 3),
		('x_usbl', 'kalman_x', 'X Position', 'X (m)', 4),
		('y_usbl', 'kalman_y', 'Y Position', 'Y (m)', 5)
	]

	for raw_col, filtered_col, label, ylabel, plot_idx in plot_configs:
		ax = axes[plot_idx]
		if raw_col in df.columns:
			ax.plot(df['Timestamp'], df[raw_col], label=f'Raw {label}')
		if filtered_col in df.columns:
			ax.plot(df['Timestamp'], df[filtered_col], label=f'Filtered {label}')
		ax.set_xlabel('Timestamp')
		ax.set_ylabel(ylabel)
		ax.legend()

	plt.tight_layout()
	plot_file = processed_dir / "kalman_assessment_plots.png"
	plt.savefig(plot_file)
	print(f"Kalman assessment plots saved to: {plot_file}")