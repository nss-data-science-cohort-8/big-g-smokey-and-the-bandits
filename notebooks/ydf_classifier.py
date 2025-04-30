import warnings

import geopandas as gpd
import pandas as pd

# No ydf.keras import needed here

# Suppress specific warnings if needed (e.g., from geopandas deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="geopandas")


# Load & prepare data
# Note: Removed try-except block for file loading. Script will crash if files are missing.
print("Loading data...")
faults_raw = pd.read_csv(
    "../data/J1939Faults.csv", dtype={"EquipmentID": str, "spn": int}
)
diagnostics_raw = pd.read_csv("../data/VehicleDiagnosticOnboardData.csv")

print("Preparing data...")
diagnostics_raw["Value"] = diagnostics_raw["Value"].replace(
    {"FALSE": False, "TRUE": True}
)
# pivot diagnostics to long format
diagnostics = diagnostics_raw.pivot(
    index="FaultId", columns="Name", values="Value"
)
# Prepare faults
drop_cols = [
    "actionDescription",
    # "activeTransitionCount", # Keep this potentially useful feature
    "eventDescription",
    "ecuSource",
    "ecuSoftwareVersion",
    "ecuModel",
    "ecuMake",
    "faultValue",
    "MCTNumber",
    "ServiceDistance",
    "LocationTimeStamp",
    "ecuSerialNumber",  # Often redundant with EquipmentID or high cardinality
]
# Filter out columns that might not exist in the actual dataset
drop_cols = [col for col in drop_cols if col in faults_raw.columns]
faults = faults_raw.drop(columns=drop_cols)


# Join diagnostics
print("Joining fault and diagnostic data...")
joined = faults.merge(
    diagnostics, how="inner", left_on="RecordID", right_on="FaultId"
)


state_cols = [
    "active",
    "CruiseControlActive",
    "IgnStatus",
    "nearStation",
    "LampStatus",
    "ParkingBrake",
]
sensor_cols = [
    "AcceleratorPedal",
    "BarometricPressure",
    "CruiseControlSetSpeed",
    "DistanceLtd",
    "EngineCoolantTemperature",
    "EngineLoad",
    "EngineOilPressure",
    "EngineOilTemperature",
    "EngineRpm",
    "EngineTimeLtd",
    "FuelLevel",
    "FuelLtd",
    "FuelRate",
    "FuelTemperature",
    "IntakeManifoldTemperature",
    "Speed",
    "SwitchedBatteryVoltage",
    "Throttle",
    "TurboBoostPressure",
]

joined[sensor_cols] = joined[sensor_cols].astype(float)
joined[state_cols] = joined[state_cols].astype(str)

# --- Geospatial Filtering ---
# Note: Removed try-except block for geospatial operations. Script will crash on errors.
print("Labeling faults near service stations...")
joined_pre_station_filter = joined
stations = pd.DataFrame(
    {
        "lat": [36.0666667, 35.5883333, 36.1950],
        "lon": [-86.4347222, -86.4438888, -83.174722],
    }
)
threshold_miles = 0.5
threshold_meters = threshold_miles * 1609.34

# create geodataframes with geopandas
gdf_joined = gpd.GeoDataFrame(
    joined,
    geometry=gpd.points_from_xy(
        joined.Longitude, joined.Latitude
    ),  # Use Longitude, Latitude order
    crs="EPSG:4326",  # WGS84 coord ref sys (lat/lon)
)
gdf_stations = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(
        stations.lon, stations.lat
    ),  # Use lon, lat order
    crs="EPSG:4326",
)
# Choose a suitable projected CRS for the area (e.g., UTM zone or a state plane)
# Using NAD83 / Conus Albers (EPSG:5070) as an example for continental US
target_crs = "EPSG:5070"
gdf_joined_proj = gdf_joined.to_crs(target_crs)
gdf_stations_proj = gdf_stations.to_crs(target_crs)

# create buffers around stations
station_buf = gdf_stations_proj.geometry.buffer(threshold_meters)
# Use union_all() as requested (note: unary_union is generally preferred)
# Ensure union_all is available (might require specific geopandas version)
if hasattr(station_buf, "union_all"):
    combined_buffer = station_buf.union_all()
else:
    # Fallback or error if union_all is not present
    print(
        "Warning: station_buf.union_all() not available. Using unary_union as fallback."
    )
    combined_buffer = station_buf.unary_union  # Fallback

is_within = gdf_joined_proj.geometry.within(combined_buffer)
joined["nearStation"] = is_within.values
joined_post_filter = joined[~joined["nearStation"]]
print(
    f"Done! Faults within {threshold_miles} miles of service station labeled."
)
print(
    f"When filtered, this removes {len(joined_pre_station_filter) - len(joined_post_filter)} rows"
)
# Apply the filter
# joined = joined_post_filter # Uncomment this line to actually apply the filter
print("Note: Geospatial filter is calculated but NOT applied by default.")
print("Uncomment 'joined = joined_post_filter' to apply it.")


# --- Active Filter ---
print("Filtering for active=True faults...")
joined_pre_active_filter = joined
# Ensure 'active' column exists and handle potential non-boolean values if necessary
if "active" in joined.columns:
    # Convert potential string 'true'/'false' to boolean if needed
    if joined["active"].dtype == "object":
        joined["active"] = (
            joined["active"].str.lower().map({"true": True, "false": False})
        )
    # Fill NA values if any - assuming inactive if unknown
    joined["active"] = joined["active"].fillna(False)
    joined_active = joined[
        joined["active"]
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning
    print(
        f"Number of rows after filtering active=False out: {len(joined_active)}"
    )
    print(
        f"Rows removed by active filter: {len(joined_pre_active_filter) - len(joined_active)}"
    )
    joined = joined_active
else:
    print("Warning: 'active' column not found. Skipping active filter.")


# --- Time Window Calculation ---
target_spn = 5246
print(f"Calculating target window for SPN {target_spn}...")
# Ensure EventTimeStamp is datetime
joined["EventTimeStamp"] = pd.to_datetime(joined["EventTimeStamp"])
# Sort by EquipmentID and then chronologically by EventTimeStamp
print("Sorting data by EquipmentID and EventTimeStamp...")
joined = joined.sort_values(by=["EquipmentID", "EventTimeStamp"]).copy()
print("Sorting complete.")

# --- Calculate time_since_last_fault ---
print("Calculating time since last fault...")
joined["time_since_last_fault"] = joined.groupby("EquipmentID")[
    "EventTimeStamp"
].diff()
print("Calculation complete.")

# Create a Series containing only the timestamps of trigger events
trigger_timestamps_only = joined["EventTimeStamp"].where(
    joined["spn"] == target_spn
)

# For each row, find the timestamp of the *next* trigger event within its group
print("Calculating next trigger time...")
joined["next_trigger_time"] = trigger_timestamps_only.groupby(
    joined["EquipmentID"]
).bfill()
# Calculate the start of the 2-hour window before the next trigger
joined["window_start_time"] = joined["next_trigger_time"] - pd.Timedelta(
    hours=2.5
)
# Label rows as True if their timestamp falls within the window
print("Labeling derate window...")
joined["derate_window"] = (
    (joined["EventTimeStamp"] >= joined["window_start_time"])
    & (joined["EventTimeStamp"] <= joined["next_trigger_time"])
    & (joined["next_trigger_time"].notna())
)
# --- Verification ---
print("\nVerification:")
print(
    "Value counts for 'derate_window':\n",
    joined["derate_window"].value_counts(dropna=False),
)
print(
    "\nValue counts for 'spn' (to confirm target SPN exists):\n",
    joined["spn"].value_counts(dropna=False).head(),  # Show top SPNs
)
# Display some rows where derate_window is True (if any)
print("\nSample rows where derate_window is True:")
print(
    joined[joined["derate_window"]][
        ["EquipmentID", "EventTimeStamp", "spn", "next_trigger_time"]
    ].head()
)

# --- Feature Selection ---
# Define target
target = "derate_window"  # Define target column name
y = joined[target].astype(int)

# Define features for modeling
# Exclude identifiers, intermediate calculation columns, the target itself,
# and columns used only for filtering/labeling
columns_to_drop_for_X = [
    target,
    "RecordID",
    "EventTimeStamp",
    "next_trigger_time",
    "window_start_time",
    "nearStation",  # Used for filtering, not a feature itself
    "active",  # Used for filtering
    "geometry",  # Geopandas object, not usable directly
    # Add any other columns that should not be features
    "LocationMethod",  # Example: If deemed not useful or problematic
]

# Start with all columns and drop the unwanted ones
features_to_keep = joined.columns.difference(
    columns_to_drop_for_X, sort=False
).tolist()

# Ensure 'time_since_last_fault' is included if it exists
if (
    "time_since_last_fault" not in features_to_keep
    and "time_since_last_fault" in joined.columns
):
    features_to_keep.append("time_since_last_fault")

# Handle potential case where diagnostic columns might be boolean/object instead of numeric
# Convert boolean diagnostic columns to int (0/1)
# Make sure 'diagnostics' DataFrame is defined before this loop
if "diagnostics" in locals():
    bool_diagnostic_cols = diagnostics.select_dtypes(include="bool").columns
    for col in bool_diagnostic_cols:
        if col in joined.columns:
            joined[col] = joined[col].astype(int)
else:
    print(
        "Warning: 'diagnostics' DataFrame not found, skipping boolean conversion."
    )


print(f"\nFeatures selected for modeling ({len(features_to_keep)}):")
# print(features_to_keep) # Uncomment to see the full list
X = joined[features_to_keep].copy()  # Use .copy()
print(X.dtypes)

# # --- Train/Test Split ---
# print("\nSplitting data into training and testing sets...")
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, test_size=0.2, random_state=42
# )
# print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
# print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
# print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
