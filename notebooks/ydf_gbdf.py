import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import ydf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

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

# filter out near service stations
joined_pre_station_filter = joined
original_count = len(joined)
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
joined = joined[~joined["nearStation"]]
filtered_count = len(joined)
print("\nDone! \nFaults within 1km of service station labeled in 'joined'.")
print(f"When filtered, this removes {original_count - filtered_count} rows")
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

# Calculate the start of the n-hour window before the next trigger
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
print("===========================================")
print("\nSample rows where derate_window is True:")
print(
    joined[joined["derate_window"]][
        ["EquipmentID", "EventTimeStamp", "spn", "next_trigger_time", "derate_window"]
    ].head()
)
print("\n==========================================\n")

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
        joined[col] = joined[col].ffill().bfill() # remove bfill?
print(joined.isna().sum())

### separate data into pre and post 2019
joined_pre_2019 = joined[joined["EventTimeStamp"].dt.year < 2019]
joined_post_2019 = joined[joined["EventTimeStamp"].dt.year >= 2019]
### check to see how many derates happen after 2019
derates_2019 = joined[
    (joined["spn"] == 5246) & (joined["EventTimeStamp"] > "12-31-2018")
].copy()  # & (joined['active'] == True)
derates_2019["derate_gap"] = (
    derates_2019.sort_values(by=["EquipmentID", "EventTimeStamp"])
    .groupby("EquipmentID")["EventTimeStamp"]
    .diff()
)
gap = pd.to_timedelta("24 hours")
confirmed_derates_2019 = derates_2019[
    (derates_2019.derate_gap.isnull()) | (derates_2019["derate_gap"] > gap)
]
print(
    f"Goal for predicting derates: {len(confirmed_derates_2019)} derates to predict"
)
### --- label predictors ---
predictors = [
    col
    for col in joined_pre_2019.columns
    if col
    not in [
        # "spn",
        "EquipmentID",
        "EventTimeStamp",
        "derate_window",
        "next_trigger_time",
        "window_start_time",
        "nearStation",
        "Latitude",
        "Longitude",
        "active",
    ]
]
target = "derate_window"
### I need to set training data to joined_pre_2019 and test data to joined_post_2019

# --- Prepare Training and Testing Data ---
print("Preparing training (pre-2019) and testing (post-2019) data...")
X_train = joined_pre_2019[predictors]
y_train = joined_pre_2019[target]

X_test = joined_post_2019[predictors]
y_test = joined_post_2019[target]

# ----CORRELATION MATRIX----
correlation_matrix = X_train.corr()
# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, cmap="coolwarm", annot=False
)  # annot=True is too dense for 80 features
plt.title("Feature Correlation Matrix")
plt.savefig("../assets/corr_matrix.png")
plt.close()

# Find highly correlated pairs
upper_tri = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
highly_correlated = [
    column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.9)
]
print("Features with high correlation (>0.9) with others:", highly_correlated)
# Storing necessary info from the original TEST set for evaluation
original_test_info = joined_post_2019[
    [
        "EquipmentID",
        "EventTimeStamp",
        "spn",
        "next_trigger_time",  # Time of the SPN 5246 event
        "derate_window",  # "Actual" label
    ]
].copy()
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
# --- Train the Model ---
# Concatenate features and target for ydf training
train_df = pd.DataFrame(X_train)
train_df["derate_window"] = y_train
# test_df for prediction only needs features initially
test_df_predict = X_test.copy()

# Adjustments for model improvement
print("\nStarting model training with tuned hyperparameters...")
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

# Model Description:
print("----Model Description----")
print(model.describe())
print("-------------------------\n")

# --- SHAP Analysis ---
print("\n--- Running SHAP Analysis ---")
background = (
    X_train.sample(n=500, random_state=42) if len(X_train) > 500 else X_train
)
explainer = shap.Explainer(model.predict, background)
X_test_sample = (
    X_test.sample(n=500, random_state=42) if len(X_test) > 500 else X_test
)
shap_values = explainer(X_test_sample)

# SHAP summary plot
shap.summary_plot(shap_values, X_test_sample, show=False)
fig = plt.gcf()  # Get current figure
fig.suptitle("SHAP Summary Plot (Test Sample)")
fig.tight_layout()
fig.savefig("../assets/shap_summary.png")
plt.close(fig)
print("SHAP summary plot saved to ../assets/shap_summary.png")

# SHAP bar plot
shap_bar = shap.plots.bar(shap_values, show=False)  # Returns Axes

# don't show mean absolute shap value since they're all 0 bc they're so small.
for txt in shap_bar.texts:
    txt.set_visible(False)

fig_bar = shap_bar.get_figure()
shap_bar.set_title("SHAP Bar")
fig_bar.tight_layout()
fig_bar.savefig("../assets/shap_bar.png")
plt.close(fig_bar)
print("SHAP bar plot saved to ../assets/shap_bar.png")

# SHAP beeswarm plot
shap_beeswarm = shap.plots.beeswarm(shap_values, show=False)  # Returns Axes
fig_beeswarm = shap_beeswarm.get_figure()
shap_beeswarm.set_title("SHAP Beeswarm")
fig_beeswarm.tight_layout()
fig_beeswarm.savefig("../assets/shap_beeswarm.png")
plt.close(fig_beeswarm)
print("SHAP beeswarm plot saved to ../assets/shap_beeswarm.png")

# --- Make Predictions ---
print("\nMaking predictions on the test set...")
# Get probability predictions
y_pred_proba = model.predict(test_df_predict)
# Convert probabilities to class predictions using a n.n threshold
y_pred_class = (y_pred_proba > 0.95).astype(int)

# --- Prepare Results Dataframe for Analysis ---
print("Preparing results dataframe for detailed analysis...")
# Create a dataframe for analysis using the original test set info
# Ensure index alignment is correct
test_results = original_test_info.copy()
test_results["predicted_derate"] = pd.Series(y_pred_class, index=X_test.index)

# Calculate time until the actual trigger event
test_results["time_until_trigger"] = (
    test_results["next_trigger_time"] - test_results["EventTimeStamp"]
)

# --- Calculate Derate Gaps (Time Since Last Actual Derate) ---
print("Calculating time gaps between actual derate events...")
# Identify actual derate trigger events in the test set
actual_triggers = joined_post_2019[joined_post_2019["spn"] == target_spn].copy()
actual_triggers = actual_triggers.sort_values(
    by=["EquipmentID", "EventTimeStamp"]
)

# Calculate time since the previous trigger for the same equipment
actual_triggers["derate_gap"] = actual_triggers.groupby("EquipmentID")[
    "EventTimeStamp"
].diff()

# Define the 24-hour threshold
derate_reset_period = pd.Timedelta(hours=24)

# Add the derate_gap to the test_results dataframe
# We merge based on the actual trigger time ('next_trigger_time' in test_results
# corresponds to 'EventTimeStamp' in actual_triggers)
test_results = pd.merge(
    test_results,
    actual_triggers[["EquipmentID", "EventTimeStamp", "derate_gap"]],
    left_on=["EquipmentID", "next_trigger_time"],
    right_on=["EquipmentID", "EventTimeStamp"],
    how="left",
    suffixes=("", "_trigger"),  # Add suffix to avoid column name clash
)
# Drop the redundant EventTimeStamp_trigger column from the merge
test_results = test_results.drop(columns=["EventTimeStamp_trigger"])

print("Derate gap calculation complete.")

# --- Identify Valuable True Positives (for Savings) ---
print("\n--- Identifying Valuable True Positives (Savings Calculation) ---")
two_hours = pd.Timedelta(hours=2)

# Conditions for a valuable TP:
# - Model predicted derate (predicted_derate == 1)
# - It's actually a derate window (derate_window == 1) - Standard TP definition
# - The prediction was made more than 2 hours before the trigger (time_until_trigger > 2 hours)
# - The actual trigger event ('next_trigger_time') occurred > 24 hours after the previous one
#    (derate_gap > 24 hours OR derate_gap is NaT, meaning it's the first one)

valuable_TPs = test_results[
    (test_results["predicted_derate"] == 1)
    & (test_results["derate_window"] == 1)  # Ensure it's a true positive
    & (test_results["time_until_trigger"].notna())
    & (test_results["time_until_trigger"] > two_hours)
    & (
        (test_results["derate_gap"].isna())  # First derate for equipment
        | (test_results["derate_gap"] > derate_reset_period)
    )
].copy()

# count unique actual derate events that were successfully predicted early
# Group by the actual trigger event and check if any prediction within that group met the criteria
valuable_TP_events = valuable_TPs.drop_duplicates(
    subset=["EquipmentID", "next_trigger_time"]
)
valuable_TP_count = len(valuable_TP_events)

print(
    f"Found {valuable_TP_count} unique actual derate events predicted >2 hours early with >{derate_reset_period} gap."
)
# --- Identify Costly False Positives (for Costs) ---
print("\n--- Identifying Costly False Positives (Cost Calculation) ---")

# all false positives
false_positives = test_results[
    (test_results["predicted_derate"] == 1)
    & (test_results["derate_window"] == 0)
].copy()

# trigger times
actual_trigger_times_map = (
    actual_triggers.groupby("EquipmentID")["EventTimeStamp"]
    .apply(list)
    .to_dict()
)


# 3. Define the function to find time to nearest actual trigger
def time_to_nearest_trigger(row, trigger_map):
    equipment_id = row["EquipmentID"]
    fp_timestamp = row["EventTimeStamp"]
    # Handle cases where equipment ID might not be in the map
    if equipment_id not in trigger_map or not trigger_map[equipment_id]:
        # Return a large timedelta if no triggers exist for this equipment
        return pd.Timedelta(days=999)

    trigger_times = trigger_map[equipment_id]
    # Calculate absolute time differences
    time_diffs = [
        abs(fp_timestamp - trigger_time) for trigger_time in trigger_times
    ]
    # Return the minimum difference
    return min(
        time_diffs
    )  # time_diffs will not be empty here due to check above


# Initialize count and dataframe for initial costly FPs
costly_FP_count = 0
initial_costly_FPs = pd.DataFrame()

# 4. Calculate time to nearest actual trigger and filter
if len(false_positives.index) > 0:
    print(
        "Calculating time difference between false positives and nearest actual derate..."
    )
    false_positives["time_to_nearest_actual"] = false_positives.apply(
        time_to_nearest_trigger, args=(actual_trigger_times_map,), axis=1
    )

    # Filter FPs that are more than 24 hours away from ANY actual trigger
    initial_costly_FPs = false_positives[
        false_positives["time_to_nearest_actual"] > derate_reset_period
    ].copy()
    print(
        f"Found {len(initial_costly_FPs)} individual FP rows > {derate_reset_period} from any actual derate."
    )

else:
    print("No false positives found.")
    # initial_costly_FPs remains empty


# --- NEW STEP: Filter clustered costly FPs ---
if len(initial_costly_FPs.index) > 0:
    print(
        f"Filtering clustered costly FPs (keeping only those > {derate_reset_period} apart)..."
    )
    # Ensure sorting for the diff calculation
    initial_costly_FPs = initial_costly_FPs.sort_values(
        by=["EquipmentID", "EventTimeStamp"]
    )

    # Calculate time since the *previous costly FP* for the same equipment
    initial_costly_FPs["time_since_last_costly_fp"] = (
        initial_costly_FPs.groupby("EquipmentID")["EventTimeStamp"].diff()
    )

    # Keep a costly FP if it's the first one for the equipment (NaT)
    # OR if it occurred more than 24 hours after the previous costly FP
    final_costly_FPs = initial_costly_FPs[
        (initial_costly_FPs["time_since_last_costly_fp"].isna())
        | (
            initial_costly_FPs["time_since_last_costly_fp"]
            > derate_reset_period
        )
    ]

    costly_FP_count = len(final_costly_FPs)
else:
    # If initial_costly_FPs was empty, the count remains 0
    print("No initial costly FPs found to filter for clustering.")
    costly_FP_count = 0


print(
    f"Found {costly_FP_count} final costly False Positive events (separated by > {derate_reset_period})."
)

# --- Standard Evaluation Metrics (for comparison) ---
print("\n--- Standard Evaluation Metrics ---")
# Note: ydf evaluate needs the target column in the test_df
test_df_eval = X_test.copy()  # Start with features
test_df_eval["derate_window"] = y_test  # Add actual labels

# Evaluate using sklearn's f1_score (using y_test and y_pred_class)
print("\nCalculating macro F1 score (sklearn)...")
# Ensure y_test and y_pred_class are aligned if indices were shuffled (shouldn't be here)
macro_f1 = f1_score(y_test, y_pred_class, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# Print classification report and confusion matrix for more detail
print("\nClassification Report (sklearn):")
# Use the actual labels from the test set and the predicted classes
print(classification_report(y_test, y_pred_class, zero_division=0))

# Shap analysis

# get feature importances
print("\n--- Feature Importances (YDF Model) ---")
try:
    importances_dict = model.variable_importances()
    # print("Variable importances dict:", importances_dict)
    # Prefer SUM_SCORE if available, else NUM_AS_ROOT, else first available
    if "SUM_SCORE" in importances_dict:
        importances = importances_dict["SUM_SCORE"]
    elif "NUM_AS_ROOT" in importances_dict:
        importances = importances_dict["NUM_AS_ROOT"]
    else:
        key = list(importances_dict.keys())[0]
        importances = importances_dict[key]
        print(f"Using variable importance type: {key}")

    # print("importances:", importances)
    # If importances is a list of tuples, convert accordingly
    if importances and isinstance(importances[0], tuple):
        importances_df = pd.DataFrame(
            importances, columns=["importance", "attribute"]
        )
    else:
        importances_df = pd.DataFrame(importances)
    importances_df = importances_df.sort_values(
        by="importance", ascending=False
    )
    print(importances_df.head(20))
except Exception as e:
    print(f"Could not retrieve feature importances: {e}")

print("\nCreating Confusion Matrix (sklearn)...")
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Derate", "Derate"]
)
disp.plot()
plt.title("Confusion Matrix (Standard)")
plt.savefig("../assets/confusion.png")
plt.close()

# Standard TP/FP from confusion matrix for comparison
TN_standard = cm[0, 0]
FP_standard = cm[0, 1]
FN_standard = cm[1, 0]
TP_standard = cm[
    1, 1
]  # All TPs within the 2hr window, regardless of gap/timing

print("Standard Confusion Matrix Counts:")
print(f"  True Negatives (TN): {TN_standard}")
print(f"  False Positives (FP): {FP_standard}")
print(f"  False Negatives (FN): {FN_standard}")
print(
    f"  True Positives (TP): {TP_standard}  <-- target for this is derate window"
)

Net_standard = (TP_standard * 4000) - (FP_standard * 500)
print(
    f"\nNet savings based on standard evaluation (all TPs, all FPs): ${Net_standard}"
)
print(
    f"\nGoal for predicting derates (from previous script): {len(confirmed_derates_2019)} derates to predict"
)  # Print the goal again
print(
    f"Actual correctly predicted derates: {len(valuable_TP_events)} :("
)  # Print the actual number of correctly predicted derates

# --- Calculate Final Cost/Savings ---
print("\n--- Final Cost/Savings Analysis ---")
Savings = valuable_TP_count * 4000
Costs = costly_FP_count * 500
Net_Savings = Savings - Costs

print(f"Valuable True Positives (Savings): {valuable_TP_count}")
print(f"Costly False Positives (Costs): {costly_FP_count}")
print(f"Total Savings: ${Savings}")
print(f"Total Costs: ${Costs}")
print(f"Net Savings (Custom Definition): ${Net_Savings}")
