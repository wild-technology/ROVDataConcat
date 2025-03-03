import os
import pandas as pd
import csv
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def safe_get_loc(df, col_name):
	"""
    Returns the integer location of the specified column.
    If multiple indices are returned (duplicate column names),
    returns the first index.
    """
	loc = df.columns.get_loc(col_name)
	try:
		return int(loc)
	except TypeError:
		# If loc is array-like, return the first element.
		return int(loc[0])

def process_data(raw_dir, processed_dir):
	"""
    Performs UTM assessment and adjusts vehicle positioning.

    After offsetting the vehicle’s (x, y) and initially adjusting depth
    based on the sampled center pixel of the GeoTIFF, this function further
    checks the 3x3 neighborhood around the offset position. If any neighboring
    pixel is within 1m of the vehicle’s depth (i.e. if the vehicle is less than
    1m above the maximum neighbor value), the depth is increased to be 1m above
    that neighbor value.

    After this neighbor evaluation, the script re-evaluates that the new depth
    is at least 1m above the terrain (i.e. the center pixel value). If not, it
    adjusts the depth accordingly.

    Finally, the script produces two PNG visualizations (a histogram of depth differences
    and a scatter plot of the offset positions colored by depth difference) and prints an
    on-screen summary that includes how many depth values after processing are still below
    the terrain (which should be 0 after the re-evaluation).
    """
	print("Running UTM Assessment Process...")

	geotiff_path = os.path.join(raw_dir, "H2021_k2mapping_geotiff_utm4n.tif")
	csv_path = os.path.join(processed_dir, "NA156_H2021_final_datatable.csv")
	output_file = os.path.join(processed_dir, "NA156_H2021_offset.csv")

	df = pd.read_csv(csv_path)

	# Define required columns
	x_col, y_col, depth_col = 'kalman_x', 'kalman_y', 'kalman_depth'
	heading_rad_col = 'Heading_rad'

	# Offset (x, y) by 2m backwards along Heading_rad (per code; comment mentioned 1m)
	df[x_col] = df[x_col] - (2 * np.cos(df[heading_rad_col]))
	df[y_col] = df[y_col] - (2 * np.sin(df[heading_rad_col]))

	# Save offset positions for later neighbor evaluation
	df['Offset_x'] = df[x_col]
	df['Offset_y'] = df[y_col]

	# Function to sample the center pixel from the raster
	def sample_raster_values(raster_path, x_series, y_series):
		coords = list(zip(x_series, y_series))
		with rasterio.open(raster_path) as src:
			sampled = list(src.sample(coords))
		return [val[0] for val in sampled]

	# Resample GeoTIFF at the offset positions (center pixel)
	df['geotiff_value'] = sample_raster_values(geotiff_path, df[x_col], df[y_col])
	df['below_surface'] = df[depth_col] < df['geotiff_value']
	df.loc[df['below_surface'], depth_col] = df['geotiff_value'] + 1

	# New function: sample neighboring pixel values from a window
	def sample_neighbor_values(raster_path, x, y, window_size=3):
		from rasterio.windows import Window
		with rasterio.open(raster_path) as src:
			# Convert geographic coordinate (x, y) to row, col indices
			row, col = src.index(x, y)
			half = window_size // 2
			window = Window(col - half, row - half, window_size, window_size)
			data = src.read(1, window=window, boundless=True)
		return data.flatten()

	# New function: adjust depth based on neighbor pixel values
	def adjust_depth_by_neighbors(row, raster_path, window_size=3):
		x = row['Offset_x']
		y = row['Offset_y']
		current_depth = row[depth_col]
		neighbors = sample_neighbor_values(raster_path, x, y, window_size)
		max_neighbor = np.max(neighbors)
		# If current_depth is less than 1m above the maximum neighbor elevation,
		# adjust it to be exactly 1m above.
		if current_depth - max_neighbor < 1:
			current_depth = max_neighbor + 1
		return current_depth

	# Apply neighbor evaluation to adjust depth further if needed
	df[depth_col] = df.apply(lambda row: adjust_depth_by_neighbors(row, geotiff_path, window_size=3), axis=1)

	# Re-evaluate: ensure each new depth is at least 1m above the center pixel's terrain value.
	below_threshold = df[depth_col] < (df['geotiff_value'] + 1)
	num_adjustments = below_threshold.sum()
	if num_adjustments > 0:
		df.loc[below_threshold, depth_col] = df.loc[below_threshold, 'geotiff_value'] + 1
	print(f"After re-evaluation, adjusted {num_adjustments} depth values to ensure they are at least 1m above terrain.")

	# ---- Visualization & Summary Section ----
	# Compute depth difference: (adjusted Depth) - (center pixel value)
	df['depth_diff'] = df[depth_col] - df['geotiff_value']
	total_rows = len(df)
	num_below = ((df[depth_col] - df['geotiff_value']) < 1).sum()
	print(
		f"Depth Summary: {num_below} out of {total_rows} depth values are less than 1m above the terrain (should be 0).")

	# Create a histogram of depth differences
	hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
	hist_ax.hist(df['depth_diff'], bins=30, edgecolor='k')
	hist_ax.set_xlabel("Depth Difference (Depth - Terrain)")
	hist_ax.set_ylabel("Frequency")
	hist_ax.set_title("Histogram of Depth Differences")
	histogram_path = os.path.join(processed_dir, "depth_difference_histogram.png")
	plt.savefig(histogram_path)
	plt.close(hist_fig)
	print(f"Depth difference histogram saved to: {histogram_path}")

	# Create a scatter plot of offset positions colored by depth difference
	scatter_fig, scatter_ax = plt.subplots(figsize=(10, 8))
	scatter = scatter_ax.scatter(df['Offset_x'], df['Offset_y'], c=df['depth_diff'], cmap='viridis', s=10)
	scatter_ax.set_xlabel("Offset X")
	scatter_ax.set_ylabel("Offset Y")
	scatter_ax.set_title("Dive Track (Offset Positions) Colored by Depth Difference")
	cbar = scatter_fig.colorbar(scatter, ax=scatter_ax)
	cbar.set_label("Depth Difference (Depth - Terrain)")
	scatter_path = os.path.join(processed_dir, "dive_track_scatter.png")
	plt.savefig(scatter_path)
	plt.close(scatter_fig)
	print(f"Dive track scatter plot saved to: {scatter_path}")
	# ---- End Visualization & Summary Section ----

	# Rename columns
	rename_map = {
		'kalman_lat': 'Latitude',
		'kalman_long': 'Longitude',
		'kalman_depth': 'Depth'
	}
	df.rename(columns=rename_map, inplace=True)

	# Convert radians to degrees
	df['Heading'] = np.degrees(df[heading_rad_col])
	df['Pitch'] = np.degrees(df['Pitch_rad'])
	df['Roll'] = np.degrees(df['Roll_rad'])

	# Reorder columns: insert Heading, Pitch, Roll immediately to the left of "Depth".
	if 'Depth' in df.columns:
		depth_idx = safe_get_loc(df, 'Depth')
		for col in reversed(['Heading', 'Pitch', 'Roll']):
			series = df.pop(col)
			df.insert(depth_idx, col, series)

	# Drop unnecessary columns
	drop_cols = [
		'x_usbl', 'y_usbl', 'x_dvl', 'y_dvl',
		'Heading_rad', 'Pitch_rad', 'Roll_rad', 'kalman_yaw_deg',
		'kalman_x', 'kalman_y', 'kalman_roll_deg', 'kalman_pitch_deg',
		'geotiff_value', 'below_surface', 'Offset_x', 'Offset_y', 'depth_diff'
	]
	df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

	# Save final CSV with all fields quoted.
	df.to_csv(output_file, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
	print(f"UTM assessment results saved to: {output_file}")

if __name__ == "__main__":
	raw_dir = input("Enter the raw directory for processing: ").strip()
	processed_dir = input("Enter the processed directory for output: ").strip()
	process_data(raw_dir, processed_dir)
