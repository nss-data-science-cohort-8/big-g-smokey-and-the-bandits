# import libraries / modules
import geopandas as gpd
import pandas as pd

# load data
faults_raw = pd.read_csv("../data/J1939Faults.csv")
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
print("--------NaNs--------")
print(diagnostics_raw.isna().sum())
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
)
print("\n\n--------JOINED--------")
print(joined.head())
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
joined_post_filter = joined[joined["nearStation"] == False]
print("\nDone! \nFaults within 1km of service station labeled in 'joined'.\n")
print(joined.head(3))
print(
    f"\nWhen filtered, this removes {len(joined_pre_station_filter['RecordID']) - len(joined_post_filter['RecordID'])} rows"
)

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
