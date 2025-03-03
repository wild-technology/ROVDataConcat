import pandas as pd
import rasterio
from pyproj import Transformer
import numpy as np
import os

def process_rov_data(tsv_path, geotiff_path, output_path):
    """
    Process ROV data: adjust depth, offset from steep terrain, and convert coordinates.

    Parameters:
        tsv_path (str): Path to the input TSV file containing ROV data.
        geotiff_path (str): Path to the GeoTIFF file with terrain data.
        output_path (str): Path to save the processed CSV file.
    """
    # Load the TSV data
    data = pd.read_csv(tsv_path, sep="\t")

    # Invert paro_depth_m and filter out rows with depth greater than -20m
    data['paro_depth_m'] = -data['paro_depth_m']  # Invert depth
    initial_row_count = len(data)
    data = data[data['paro_depth_m'] <= -20]  # Remove rows shallower than -20m
    removed_row_count = initial_row_count - len(data)
    print(f"Filtered out {removed_row_count} rows with depth shallower than -20m.")

    # Ensure all required columns exist in the dataset
    required_columns = [
        'time', 'dvl_lat', 'dvl_lon', 'paro_depth_m', 'ctd_conductivity',
        'ctd_pressure_psi', 'ctd_salinity_psu', 'ctd_sound_velocity_ms',
        'ctd_temp_c', 'octans_heading', 'octans_pitch', 'octans_roll',
        'oxygen_uncompensated_concentration_micromolar',
        'oxygen_uncompensated_saturation_percent', 'usbl_lat', 'usbl_lon'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in TSV: {missing_columns}")

    # Fill missing DVL data with USBL data
    data['raw_lat'] = data['dvl_lat'].fillna(data['usbl_lat'])
    data['raw_lon'] = data['dvl_lon'].fillna(data['usbl_lon'])

    # Convert lat/long to UTM (EPSG:26904)
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:26904", always_xy=True)
    data['raw_utm_easting'], data['raw_utm_northing'] = transformer_to_utm.transform(
        data['raw_lon'], data['raw_lat']
    )

    # Initialize GeoTIFF for sampling theoretical depths
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        resolution = src.res
        theoretical_depths = []

        # Prepare UTM coordinates for sampling
        coordinates = list(zip(data['raw_utm_easting'], data['raw_utm_northing']))
        for coord in coordinates:
            # Check if the coordinate is within GeoTIFF bounds
            if bounds.left <= coord[0] <= bounds.right and bounds.bottom <= coord[1] <= bounds.top:
                for val in src.sample([coord]):
                    theoretical_depths.append(val[0])
            else:
                theoretical_depths.append(np.nan)  # Out-of-bounds points

    # Add the theoretical depth column to the data
    data['theoretical_max_depth_m'] = theoretical_depths

    # Adjust depth to ensure ROV is always at least 1m above the seabed
    data['adjusted_depth_m'] = np.where(
        data['paro_depth_m'] <= data['theoretical_max_depth_m'] + 1,
        data['theoretical_max_depth_m'] - 1,
        data['paro_depth_m']
    )

    # Offset for steep terrain
    debug_log = 0
    with rasterio.open(geotiff_path) as src:
        for index, row in data.iterrows():
            if not np.isnan(row['theoretical_max_depth_m']):
                x, y = row['raw_utm_easting'], row['raw_utm_northing']
                adjacent_coords = [
                    (x - resolution[0], y), (x + resolution[0], y),
                    (x, y - resolution[1]), (x, y + resolution[1])
                ]
                adjacent_depths = []
                for coord in adjacent_coords:
                    if bounds.left <= coord[0] <= bounds.right and bounds.bottom <= coord[1] <= bounds.top:
                        for val in src.sample([coord]):
                            adjacent_depths.append(val[0])

                # Check for steep terrain
                if any(adj_depth > row['theoretical_max_depth_m'] + 0.5 for adj_depth in adjacent_depths):
                    debug_log += 1
                    data.at[index, 'adjusted_utm_easting'] = x + resolution[0]  # Simple offset to the east
                    data.at[index, 'adjusted_utm_northing'] = y  # Keep northing the same

    print(f"Debug: Steep terrain adjustment occurred {debug_log} times.")

    # For rows not requiring location adjustments, copy raw UTM values
    data['adjusted_utm_easting'] = np.where(
        data['adjusted_utm_easting'].isnull(),
        data['raw_utm_easting'],
        data['adjusted_utm_easting']
    )
    data['adjusted_utm_northing'] = np.where(
        data['adjusted_utm_northing'].isnull(),
        data['raw_utm_northing'],
        data['adjusted_utm_northing']
    )

    # Convert adjusted UTM coordinates back to WGS84
    transformer_to_wgs84 = Transformer.from_crs("EPSG:26904", "EPSG:4326", always_xy=True)
    data['adjusted_lon'], data['adjusted_lat'] = transformer_to_wgs84.transform(
        data['adjusted_utm_easting'], data['adjusted_utm_northing']
    )

    # Ensure all adjusted depths are negative and create final "depth" column
    data['depth'] = data['adjusted_depth_m']

    # Format the final dataset to match the expected column order
    formatted_data = pd.DataFrame({
        'Timestamp': data['time'],
        'Longitude': data['adjusted_lon'],
        'Latitude': data['adjusted_lat'],
        'Depth': data['depth'],
        'Conductivity': data['ctd_conductivity'],
        'PressurePSI': data['ctd_pressure_psi'],
        'SalinityPSU': data['ctd_salinity_psu'],
        'SoundVelocityMS': data['ctd_sound_velocity_ms'],
        'TemperatureC': data['ctd_temp_c'],
        'Heading': data['octans_heading'],
        'Pitch': data['octans_pitch'],
        'Roll': data['octans_roll'],
        'OxygenUncompensatedConcentrationMicromolar': data['oxygen_uncompensated_concentration_micromolar'],
        'OxygenUncompensatedSaturationPercent': data['oxygen_uncompensated_saturation_percent'],
        'SealogEventText': data.get('sealog_event_free_text', ""),
        'SealogEventValue': data.get('sealog_event_value', "")
    })

    # Export the formatted dataset
    formatted_data.to_csv(output_path, index=False, float_format="%.6f")

if __name__ == "__main__":
    # Prompt the user for input and output paths
    print("Welcome to the ROV Data Processing Script! This script nudges the ROV above the terrain")
    tsv_path = input("Enter the path to the input TSV file: ").strip()
    geotiff_path = input("Enter the path to the GeoTIFF file: ").strip()

    # Generate output path based on input file
    input_filename = os.path.basename(tsv_path)
    output_filename = f"processed_{os.path.splitext(input_filename)[0]}.csv"
    output_path = os.path.join(os.getcwd(), output_filename)

    print(f"The processed file will be saved as: {output_path}")

    # Process the ROV data
    process_rov_data(tsv_path=tsv_path, geotiff_path=geotiff_path, output_path=output_path)

    print(f"Processing complete. Processed data saved to {output_path}")
