# imports
import pandas as pd
from geopy.distance import geodesic

# load and prepare data
faults = pd.read_csv("../data/J1939Faults.csv")
diagnostics = pd.read_csv("../data/VehicleDiagnosticOnboardData.csv")
print("----Raw Data----")
print(
    f"faults.shape: {faults.shape}", f"diagnostics.shape: {diagnostics.shape}"
)
print(f"\nfaults: {faults}", f"\ndiagnostics: {diagnostics}")
# drop unneccesary columns
drop_list = [
    "ESS_Id",
    "actionDescription",
    "ecuSoftwareVersion",
    "ecuSerialNumber",
    "ecuModel",
    "ecuMake",
    "ecuSource",
    "faultValue",
    "LocationTimeStamp",
    "MCTNumber",
]
faults = faults.drop(columns=drop_list)
print(f"\nfaults.head(), shape: {faults.head(), faults.shape}")

# identify service station locations
service_stations = [
    (36.0666667, -86.4347222),
    (35.5883333, -86.4438888),
    (36.1950, -83.174722),
]
threshold_distance = 1.0


def is_near_service_station(lat, lon):
    point = (lat, lon)
    for station in service_stations:
        distance = geodesic(point, station).kilometers
        if distance <= threshold_distance:
            return True
    return False


# create boolean column denoting service station
faults["IsServiceStation"] = faults.apply(
    lambda row: is_near_service_station(row["Latitude"], row["Longitude"]),
    axis=1,
)
print(
    f'\nfaults["IsServiceStation"]: {faults["IsServiceStation"].value_counts(normalize=True)}'
)  # proportion near service stations.
diagnostics["Value"] = diagnostics["Value"].replace(
    {"FALSE": False, "TRUE": True}
)

# pivot diagnostics to long format
diagnostics_w = diagnostics.pivot(
    index="FaultId", columns="Name", values="Value"
)
features = diagnostics_w.reset_index()
features.columns.name = None
print("\n----Features after pivoting diagnostics----")
print(f"features.head(): {features.head()}")
combined = pd.merge(
    faults, features, left_on="RecordID", right_on="FaultId", how="left"
)
combined_filtered = combined[~combined["IsServiceStation"]]
combined["IsDerateFull"] = combined["spn"] == 5246
print(
    f"\ncombined['IsDerateFull']: {combined['IsDerateFull'].value_counts(normalize=True)}"
)
print("\n----Combined features and faults----")
print(f"combined.head(): {combined.head()}")

features = [
    "AcceleratorPedal",
    "DistanceLtd",
    "EngineOilTemperature",
    "TurboBoostPressure",
    "FuelRate",
    "EngineLoad",
    "EngineOilPressure",
    "EngineCoolantTemperature",
    "BarometricPressure",
    "EngineRpm",
    "IntakeManifoldTemperature",
    "FuelTemperature",
    "SwitchedBatteryVoltage",
]
print(combined[features].info())
print(combined[features].head())
