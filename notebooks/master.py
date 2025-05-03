import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import ydf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

faults_raw = pd.read_csv(
    "../data/J1939Faults.csv", dtype={"EquipmentID": str, "spn": int}
)
diagnostics_raw = pd.read_csv("../data/vehiclediagnosticonboarddata.csv")
# prepare faults
faults_drop_cols = [
    "actionDescription",
    "activeTransitionCount",
    "eventDescription",
    "ecuSource",
    "ecuSoftwareVersion",
    "ecuModel",
    "ecuMake",
    "faultValue",
    "MCTNumber",
    "LocationTimeStamp",
]
faults = faults_raw.drop(columns=faults_drop_cols)
print("\n\n--------SHAPE OF FAULTS--------")
print(faults.shape)
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
joined_pre_station_filter = joined
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
print("\nDone! \nFaults within 1km of service station labeled in 'joined'.")
print(
    f"When filtered, this removes {len(joined_pre_station_filter['RecordID']) - len(joined_post_filter['RecordID'])} rows"
)
# filter out active=False
joined_active = joined[joined["active"]]
joined = joined_active
print(
    f"\nNumber of rows after filtering active=False out: {len(joined_active['active'])}"
)
print(
    f"Rows removed: {len(joined_pre_station_filter['RecordID']) - len(joined_active['active'])}"
)
target_spn = 5246

# Ensure EventTimeStamp is datetime
joined["EventTimeStamp"] = pd.to_datetime(joined["EventTimeStamp"])

# SORTING STEP
# Sort by EquipmentID and then chronologically by EventTimeStamp
print("Sorting data by EquipmentID and EventTimeStamp...")
joined = joined.sort_values(by=["EquipmentID", "EventTimeStamp"]).copy()
print("Sorting complete.")

# Create a Series containing only the timestamps of trigger events
trigger_timestamps_only = joined["EventTimeStamp"].where(
    joined["spn"] == target_spn
)

# For each row, find the timestamp of the *next* trigger event within its group
# Group by EquipmentID and use backward fill (bfill)
# This fills NaT values with the next valid timestamp in the group
print("Calculating next trigger time...")
joined["next_trigger_time"] = trigger_timestamps_only.groupby(
    joined["EquipmentID"]
).bfill()

# Calculate the start of the 2-hour window before the next trigger
joined["window_start_time"] = joined["next_trigger_time"] - pd.Timedelta(
    hours=2.0
)

# Label rows as True if their timestamp falls within the window:
#    [wind`ow_start_time, next_trigger_time]
#    Also ensure that a next_trigger_time actually exists (it's not NaT)
print("Labeling derate window...")
joined["derate_window"] = (
    (joined["EventTimeStamp"] >= joined["window_start_time"])
    & (joined["EventTimeStamp"] <= joined["next_trigger_time"])
    & (joined["next_trigger_time"].notna())
)

# Verification
print("\nVerification:")
print(
    "Value counts for 'derate_window':\n",
    joined["derate_window"].value_counts(),
)
print(
    "\nValue counts for 'spn' (to confirm target SPN exists):\n",
    joined["spn"].value_counts(),
)

# Display some rows where derate_window is True (if any)
print("\nSample rows where derate_window is True:")
print(
    joined[joined["derate_window"]][
        ["EquipmentID", "EventTimeStamp", "spn", "next_trigger_time"]
    ].head()
)


joined = joined.drop(
    columns=[
        # "next_trigger_time",
        # "window_start_time",
        "CruiseControlActive",
        "AcceleratorPedal",
        "DistanceLtd",
        "FuelLevel",
        "ParkingBrake",
        "SwitchedBatteryVoltage",
        "RecordID",
        "ESS_Id",
        "ecuSerialNumber",
        "CruiseControlSetSpeed",
        "IgnStatus",
        "LampStatus",
        "IntakeManifoldTemperature",
        "ServiceDistance",
    ]
)
# some feature engineering:
joined["time_since_last_fault"] = (
    joined.groupby("EquipmentID")["EventTimeStamp"]
    .diff()
    .dt.total_seconds()
    .astype(float)
)
print(joined["time_since_last_fault"])
joined["fault_frequency"] = joined.groupby("EquipmentID")["spn"].transform(
    "count"
)
col_order = [
    "EquipmentID",
    "EventTimeStamp",
    "next_trigger_time",
    "window_start_time",
    "spn",
    "fmi",
    "active",
    "derate_window",
    "time_since_last_fault",
    "fault_frequency",
    "Latitude",
    "Longitude",
    "nearStation",
    "Speed",
    "BarometricPressure",
    "EngineCoolantTemperature",
    "EngineLoad",
    "EngineOilPressure",
    "EngineOilTemperature",
    "EngineRpm",
    "EngineTimeLtd",
    "FuelLtd",
    "FuelRate",
    "FuelTemperature",
    "Throttle",
    "TurboBoostPressure",
]
joined = joined[col_order]
joined.columns
print(joined.dtypes)
comma_sub_cols = [
    "Speed",
    "BarometricPressure",
    "EngineCoolantTemperature",
    "EngineLoad",
    "EngineOilPressure",
    "EngineOilTemperature",
    "EngineRpm",
    "FuelRate",
    "FuelTemperature",
    "Throttle",
    "TurboBoostPressure",
    "EngineTimeLtd",
    "FuelLtd",
]

for col in comma_sub_cols:
    joined[col] = joined[col].str.replace(",", ".", regex=True)

dtypes = {
    "EquipmentID": object,
    "EventTimeStamp": "datetime64[ns]",
    "spn": int,
    "fmi": int,
    "active": bool,
    "derate_window": bool,
    "time_since_last_fault": float,
    "fault_frequency": int,
    "Latitude": float,
    "Longitude": float,
    "nearStation": bool,
    "Speed": float,
    "BarometricPressure": float,
    "EngineCoolantTemperature": float,
    "EngineLoad": float,
    "EngineOilPressure": float,
    "EngineOilTemperature": float,
    "EngineRpm": float,
    "EngineTimeLtd": float,
    "FuelLtd": float,
    "FuelRate": float,
    "FuelTemperature": float,
    "Throttle": float,
    "TurboBoostPressure": float,
}
joined = joined.astype(dtype=dtypes)
for col in joined.columns:
    if joined[col].dtype == "bool":
        joined[col] = joined[col].astype(int)
print(joined.dtypes)
# EquipmentID and EventTimeStamp will ultimately be dropped before being run through the model, they're just our grouping variables mostly.
print(joined.isna().sum())
print(joined["Throttle"].value_counts())
for col in joined.columns:
    if joined[col].dtype == "int64" or joined[col].dtype == "float64":
        joined[col] = joined[col].bfill().ffill()
print(joined.isna().sum())

predictors = [
    col
    for col in joined.columns
    if col
    not in [
        "EquipmentID",
        "EventTimeStamp",
        "derate_window",
        "next_trigger_time",
        "window_start_time",
    ]  # Keep next_trigger_time out for now
]
target = "derate_window"

# prepare data for splitting
X = joined[predictors]
y = joined[target]

# storing this for later use calculating which True Positives are more than 2 hours out
original_test_info = joined[
    [
        "EventTimeStamp",
        "next_trigger_time",
        "derate_window",
        "EquipmentID",
        "spn",
    ]
].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# concatenate features and target for ydf training
train_df = pd.DataFrame(X_train)
train_df["derate_window"] = y_train
test_df = X_test.copy()  # test_df for prediction only needs features

# adjustments for model improvement
print("Starting model training with tuned hyperparameters...")

model = ydf.GradientBoostedTreesLearner(
    label="derate_window",  # Target column name
    task=ydf.Task.CLASSIFICATION,
    num_trees=500,
    max_depth=10,
    shrinkage=0.1,  # A common starting learning rate
    l2_regularization=0.01,  # ridge regression
    subsample=0.8,  # Use 80% of data per tree
).train(train_df)
print("Model training complete.")

# adjustments for evaluation
# Test the model
print("Making predictions on the test set...")
y_pred_proba = model.predict(test_df)  # get probability predictions

# convert probabilities to class predictions using a 0.5 threshold
y_pred_class = (y_pred_proba > 0.5).astype(int)  # Ensure integer type

# create a dataframe for analysis
test_results = original_test_info.loc[X_test.index].copy()

# add the predictions to this dataframe, ensuring the index aligns
test_results["predicted_derate"] = pd.Series(y_pred_class, index=X_test.index)

# find predictions before the 2-hour window
print("\n--- Analyzing Predictions Before 2-Hour Window ---")

# calculate time difference for all rows first.
# where next_trigger_time is nat, the result will also be nat (for timedelta).
time_diff = test_results["next_trigger_time"] - test_results["EventTimeStamp"]

# assign the result directly. pandas will create the column with timedelta64[ns] dtype.
test_results["time_until_trigger"] = time_diff

# define the 2-hour threshold
two_hours = pd.Timedelta(hours=2)

# filter for predictions that meet the criteria:
# 1. model predicted derate (predicted_derate == 1)
# 2. the time_until_trigger is valid (not nat)
# 3. the time difference is *more* than 2 hours
early_predictions = test_results[
    (test_results["predicted_derate"] == 1)
    & (test_results["time_until_trigger"].notna())  # Check for valid timedelta
    & (test_results["time_until_trigger"] > two_hours)
].copy()

print(
    f"\nFound {len(early_predictions)} instances where the model predicted a derate more than 2 hours before the actual trigger event."
)


# standard evaluation
# note: ydf evaluate needs the target column in the test_df
test_df_eval = pd.DataFrame(test_df)
test_df_eval["derate_window"] = y_test  # Create df with target for evaluation
evaluation = model.evaluate(test_df_eval)
print("\nFull evaluation report: ", evaluation)

# evaluate using sklearn's f1_score (using y_test and y_pred_class)
print("Calculating macro F1 score...")
macro_f1 = f1_score(y_test, y_pred_class, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# print classification report and confusion matrix for more detail
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Derate", "Derate"]
)
disp.plot()
plt.title("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    columns=["Pred No Derate", "Pred Derate"],
    index=["True No Derate", "True Derate"],
)
# standard TP, FP, FN, TN from the confusion matrix
# TN_standard = cm_df.iloc[0, 0]
FP_standard = cm_df.iloc[0, 1]  # all false positives
# FN_standard = cm_df.iloc[1, 0]
TP_standard = cm_df.iloc[1, 1]  # all true positives (within the 2hr window)

early_warning_TP_count = len(early_predictions)

FP_cost_count = FP_standard

Savings_early = early_warning_TP_count * 4000
Costs_early = FP_cost_count * 500
Net_early = Savings_early - Costs_early

print("\n--- Cost/Savings Analysis (Early Prediction) ---")
print(
    f"Number of valuable early warnings (predicted >2hrs early): {early_warning_TP_count}"
)
# print(f"Number of False Positives (cost incurred): {FP_cost_count}")
# print(f"Total Savings from early warnings: ${Savings_early}")
# print(f"Total Costs from False Positives: ${Costs_early}")
print(f"Net Savings (Early Warning Focused): ${Net_early}")

# standard net savings for comparison
Net_standard = (TP_standard * 4000) - (FP_standard * 500)
print(f"\nNet savings based on standard evaluation (all TPs): ${Net_standard}")

# show confusion matrix plot
plt.show()
