import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter

# Load the dataset
file_path = "E:/RUMI/NAUTILUS-CRUISE-COPY2/NA156/NA156_H2021_concat.csv"
df = pd.read_csv(file_path)

# Initialize Kalman Filter for Latitude and Longitude
kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states (lat, lon, velocity_lat, velocity_lon), 2 measurements (lat, lon)

dt = 1  # Time step assumption (adjust based on data frequency)

# State Transition Matrix (Assuming constant velocity model)
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Measurement Function (We only measure position)
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

# Process Noise Covariance (Assumed small)
kf.Q = np.eye(4) * 0.0001

# Measurement Noise Covariance (Tune this based on USBL noise)
kf.R = np.diag([0.0005, 0.0005])  # Example values, adjust if needed

# Initial State Estimate
kf.x = np.array([df.iloc[0]['Latitude'], df.iloc[0]['Longitude'], 0, 0])

# Initial Covariance
kf.P = np.eye(4) * 1

# Lists to store filtered values
kalman_lat = []
kalman_lon = []

# Apply Kalman filter
for index, row in df.iterrows():
    z = np.array([row['Latitude'], row['Longitude']])  # Measurement

    kf.predict()  # Predict step
    kf.update(z)  # Update step with measurement

    kalman_lat.append(kf.x[0])  # Filtered Latitude
    kalman_lon.append(kf.x[1])  # Filtered Longitude

# Add filtered values to DataFrame
df["Kalman_Lat"] = kalman_lat
df["Kalman_Long"] = kalman_lon

# Save to new CSV
output_file = "E:/RUMI/NAUTILUS-CRUISE-COPY2/NA156/NA156_H2021_filtered.csv"
df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
