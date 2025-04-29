# import libraries / modules

import geopandas as gpd
import pandas as pd

# load data
faults_raw = pd.read_csv("../data/J1939Faults.csv", dtype={"EquipmentID": str})
diagnostics_raw = pd.read_csv("../data/vehiclediagnosticonboarddata.csv")
# prepare faults
faults_drop_cols = [
    "actionDescription",
    "ecuSource",
    "faultValue",
    "MCTNumber",
    "LocationTimeStamp",
]
faults = faults_raw.drop(columns=faults_drop_cols)
print("\n\n--------FAULTS_RAW COLUMNS--------")
print(faults.columns)
print("\n\n--------SHAPE OF FAULTS--------")
print(faults.shape)
print("\n\n--------NaNs--------")
print(faults.isna().sum())

# join diagnostics
print("--------DIAGNOSTICS--------")
print("Checking...")
n_ids = len(diagnostics_raw["Id"])
n_unique_id = diagnostics_raw["Id"].nunique()
n_un_faults = diagnostics_raw["FaultId"].nunique()
diagnostics_raw["Value"] = diagnostics_raw["Value"].replace(
    {"FALSE": False, "TRUE": True}
)
# pivot diagnostics to long format
diagnostics = diagnostics_raw.pivot(
    index="FaultId", columns="Name", values="Value"
)
print(f"\nlen(Id): {n_ids}", f"\nN unique_Id: {n_unique_id}")
print("\n--------RECORD ID vs FAULT ID--------")
print(
    f"n_unique FaultID: {n_un_faults}",
    f"\nn_unique RecordID: {faults['RecordID'].nunique()}",
)
joined = faults.merge(
    diagnostics, how="inner", left_on="RecordID", right_on="FaultId"
).copy()
# print("\n\n--------JOINED--------")
# print(joined.head())
print("\n\n--------JOINED COLUMNS---------")
print(joined.columns)

# filter out near service stations
joined_pre_station_filter = joined
print("Labeling faults near service stations...")
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
    geometry=gpd.points_from_xy(joined.Latitude, joined.Longitude),
    crs="EPSG:4326",  # WGS84 coord ref sys (lat/lon)
)
gdf_stations = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations.lat, stations.lon),
    crs="EPSG:4326",
)
target_crs = "EPSG:9311"
# reproject onto new crs for better distance measurement
gdf_joined_proj = gdf_joined.to_crs(target_crs)
gdf_stations_proj = gdf_stations.to_crs(target_crs)
# create buffers around stations
station_buf = gdf_stations_proj.geometry.buffer(threshold_meters)
combined_buffer = (
    station_buf.union_all()
)  # turns into single geometry which helps with efficiency
is_within = gdf_joined_proj.geometry.within(combined_buffer)
joined["nearStation"] = is_within.values
joined_post_filter = joined[~joined["nearStation"]]
print("\nDone! \nFaults within 1km of service station labeled in 'joined'.\n")
print(
    f"\nWhen filtered, this removes {len(joined_pre_station_filter['RecordID']) - len(joined_post_filter['RecordID'])} rows"
)
# filter out active=False
joined = joined[joined["active"]].copy()
print(
    f"Number of rows after filtering active=False out: {len(joined['active'])}"
)
print(
    f"Rows removed: {len(joined_pre_station_filter['RecordID']) - len(joined['active'])}"
)
print(joined["spn"].value_counts(), joined.columns)

# select out derates
full_derates_raw = joined[joined["spn"] == 5246]
partial_derates_raw = joined[(joined["spn"] == 1569) & (joined["fmi"] == 31)]
# print("--------FULL DERATES---------")
# print(f"derate shape: {full_derates_raw.shape}", full_derates_raw.head(3))
# print("\n--------PARTIAL DERATES---------")
# print(
#     f"partial derate shape: {partial_derates_raw.shape}",
#     partial_derates_raw.head(3),
# )

# look at time series by equipment id.
# label derate through first outside 2 hours as true to train model.
col_order = [
    "RecordID",
    "EquipmentID",
    "EventTimeStamp",
    "spn",
    "fmi",
    "active",
    "Latitude",
    "Longitude",
    "AcceleratorPedal",
    "BarometricPressure",
    "CruiseControlActive",
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
    "IgnStatus",
    "IntakeManifoldTemperature",
    "LampStatus",
    "ParkingBrake",
    "ServiceDistance",
    "Speed",
    "SwitchedBatteryVoltage",
    "Throttle",
    "TurboBoostPressure",
    "nearStation",
    "ESS_Id",
    "ecuSerialNumber",
]
target_spn = 5246
joined["EventTimeStamp"] = pd.to_datetime(joined["EventTimeStamp"]).copy()
joined = joined[col_order]
# Create a Series containing only the timestamps of trigger events
trigger_timestamps_only = joined["EventTimeStamp"].where(joined["spn"] == 5426)
# For each row, find the timestamp of the *next* trigger event within its group
# Group by EquipmentID and use backward fill (bfill)
# This fills NaT values with the next valid timestamp in the group
joined["next_trigger_time"] = trigger_timestamps_only.groupby(
    joined["EquipmentID"]
).bfill()
# 3. Calculate the start of the 2-hour window before the next trigger
joined["window_start_time"] = joined["next_trigger_time"] - pd.Timedelta(
    hours=2
)
# 4. Label rows as True if their timestamp falls within the window:
#    [window_start_time, next_trigger_time]
#    Also ensure that a next_trigger_time actually exists (it's not NaT)
joined["derate_window"] = (
    (joined["EventTimeStamp"] >= joined["window_start_time"])
    & (joined["EventTimeStamp"] <= joined["next_trigger_time"])
    & (joined["next_trigger_time"].notna())
)
