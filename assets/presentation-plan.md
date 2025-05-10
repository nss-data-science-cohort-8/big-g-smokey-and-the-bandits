---
id: presentation-plan
title: "presentation-plan"
author: "andrew richard"
---
# Vehicle Derate Prediction Analysis: Code Summary

This document outlines the steps performed by the Python script to predict vehicle derate events, focusing on data preparation, model training, and a custom evaluation methodology.

## 1. Data Loading and Initial Preparation

*   **Load Datasets:**
    *   `J1939Faults.csv`: Contains information about vehicle faults, including timestamps, equipment IDs, SPN (Suspect Parameter Number), and FMI (Failure Mode Identifier).
    *   `vehiclediagnosticonboarddata.csv`: Contains diagnostic parameter readings associated with fault events.
*   **Prepare Faults Data:**
    *   A predefined list of columns (`faults_drop_cols`) is dropped from the `faults_raw` DataFrame. These columns are likely deemed irrelevant for the analysis (e.g., `actionDescription`, `ecuSoftwareVersion`).
        - "actionDescription",
        - "activeTransitionCount",
        - "eventDescription",
        - "ecuSource",
        - "ecuSoftwareVersion",
        - "ecuModel",
        - "ecuMake",
        - "faultValue",
        - "MCTNumber",
        - "LocationTimeStamp",

*   **Prepare Diagnostics Data:**
    *   Checks for NaN values in `diagnostics_raw`.
    *   Counts total and unique `Id` values, and unique `FaultId` values.
    *   Converts string "TRUE"/"FALSE" values in the `Value` column to Python boolean `True`/`False`.
    *   **Pivot Diagnostics Data:** The `diagnostics_raw` DataFrame is pivoted. `FaultId` becomes the index, `Name` (diagnostic parameter names) become the new columns, and `Value` (diagnostic readings) populate the cells. This transforms the data from a long to a wide format, where each row represents a unique fault event and columns represent different diagnostic readings at the time of that fault.

## 2. Data Merging

*   The prepared `faults` DataFrame is merged with the pivoted `diagnostics` DataFrame.
*   The merge is an `inner` join, performed on `faults.RecordID` and `diagnostics.FaultId` (which was the index of the pivoted `diagnostics` DataFrame).
*   The result is a `joined` DataFrame containing fault information and associated diagnostic parameters for each event.

## 3. Geospatial Filtering (Service Stations)

*   **Objective:** To identify and potentially filter out fault events that occur very close to predefined service station locations. The assumption might be that faults near service stations are immediately addressed and not representative of typical operational failures.
*   **Define Stations:** A DataFrame `stations` is created with latitude and longitude coordinates for three service stations.
*   **Set Threshold:** A distance threshold of 0.5 miles is set (converted to meters).
*   **Create GeoDataFrames:**
    *   `gdf_joined`: The `joined` DataFrame is converted into a GeoDataFrame, with point geometries created from `Latitude` and `Longitude` columns.
    *   `gdf_stations`: The `stations` DataFrame is converted into a GeoDataFrame.
    *   Both are initially set to `CRS="EPSG:4326"` (WGS84, standard lat/lon).
*   **Reproject:** Both GeoDataFrames are reprojected to `CRS="EPSG:9311"`, a projected coordinate system suitable for accurate distance measurements in meters for the likely geographic area.
*   **Create Buffers:** A buffer (circular area) is created around each station point in `gdf_stations_proj` using the `threshold_meters`. These buffers are then combined into a single multi-polygon geometry (`combined_buffer`).
*   **Identify Faults within Buffers:** The script checks which fault points in `gdf_joined_proj` fall `within` the `combined_buffer`.
*   **Label and Filter:**
    *   A new boolean column `nearStation` is added to the `joined` DataFrame, indicating if a fault is within the buffer.
    *   The `joined` DataFrame is then filtered to *exclude* rows where `nearStation` is `True`.

## 4. Filter by 'active' Status

*   The `joined` DataFrame is further filtered to keep only rows where the `active` column is `True`. This focuses the analysis on faults that were active at the time of recording.

## 5. Target Event and Window Definition (Derate Prediction Setup)

*   **Define Target SPN:** `target_spn` is set to `5246`. This SPN likely signifies a specific critical fault event that leads to an engine derate.
*   **Timestamp Conversion:** The `EventTimeStamp` column is converted to pandas datetime objects.
*   **Sort Data:** The `joined` DataFrame is sorted by `EquipmentID` and then chronologically by `EventTimeStamp`. This is crucial for time-series-based calculations.
*   **Identify Next Trigger Time:**
    *   A Series `trigger_timestamps_only` is created, containing `EventTimeStamp` values only for rows where `spn == target_spn`. Other rows get `NaT` (Not a Time).
    *   This Series is grouped by `EquipmentID`, and a backward fill (`bfill()`) is applied. This means for each fault event, `next_trigger_time` will be the timestamp of the *next* occurrence of `spn == target_spn` for that same `EquipmentID`. If no subsequent target SPN occurs for an equipment, `next_trigger_time` will remain `NaT`.
*   **Define Derate Window:**
    *   `window_start_time` is calculated as 2 hours before `next_trigger_time`.
    *   A new boolean column `derate_window` is created. It's `True` if:
        1.  The fault's `EventTimeStamp` is on or after `window_start_time`.
        2.  The fault's `EventTimeStamp` is on or before `next_trigger_time`.
        3.  `next_trigger_time` is not `NaT` (i.e., a target derate event actually follows).
    *   This `derate_window` column will serve as the target variable for the machine learning model: the goal is to predict if a fault falls within this 2-hour window preceding a derate.

## 6. Feature Engineering and Selection

*   **Drop Columns:** Several columns are dropped from the `joined` DataFrame (e.g., `CruiseControlActive`, `RecordID`, location-related columns after filtering).
*   **Create New Features:**
    *   `time_since_last_fault`: For each `EquipmentID`, this calculates the time difference (in seconds) between the current fault event and the previous fault event.
    *   `fault_frequency`: For each `EquipmentID`, this counts the total number of faults recorded for that equipment in the dataset.
*   **Reorder Columns:** Columns are reordered for clarity.
*   **Data Type Correction:**
    *   For a list of columns (`comma_sub_cols`), string commas (`,`) are replaced with periods (`.`) to ensure correct parsing as decimal numbers.
    *   The DataFrame is then explicitly cast to specified `dtypes` (e.g., `float`, `int`, `datetime64[ns]`).
    *   Boolean columns are converted to integer type (0 or 1).
*   **Handle Remaining NaNs:** For columns with `int64` or `float64` dtypes, any remaining NaN values are filled using a backward fill (`bfill()`) followed by a forward fill (`ffill()`). This ensures no NaNs in numeric features fed to the model.

## 7. Data Splitting (Temporal)

*   **Split by Year:** The `joined` DataFrame is split into two:
    *   `joined_pre_2019`: Data where `EventTimeStamp` is before the year 2019 (used for training).
    *   `joined_post_2019`: Data where `EventTimeStamp` is 2019 or later (used for testing).
*   **Estimate Derate Prediction Goal:**
    *   `derates_2019`: A subset of post-2019 data containing only actual derate events (`spn == 5246`).
    *   `derate_gap`: For these derate events, the time difference since the *previous* derate event for the same `EquipmentID` is calculated.
    *   `confirmed_derates_2019`: Filters `derates_2019` to count unique derate events. An event is considered unique if it's the first for an equipment or if its `derate_gap` is greater than 24 hours. This count (`len(confirmed_derates_2019)`) serves as a benchmark for how many distinct derate situations the model aims to predict.

## 8. Predictor and Target Definition

*   **Predictors (`predictors`):** A list of column names is created, excluding `EquipmentID`, `EventTimeStamp`, the target `derate_window`, helper time columns (`next_trigger_time`, `window_start_time`), and columns already filtered out or not intended as features (like `nearStation`, `Latitude`, `Longitude`, `active`).
*   **Target (`target`):** The target variable is set to `derate_window`.

## 9. Model Training (Yggdrasil Decision Forests - Gradient Boosted Trees)

*   **Prepare Data for Model:**
    *   `X_train`, `y_train`: Features and target from `joined_pre_2019`.
    *   `X_test`, `y_test`: Features and target from `joined_post_2019`.
    *   `original_test_info`: A copy of key columns from `joined_post_2019` is stored for later detailed evaluation (EquipmentID, EventTimeStamp, spn, next_trigger_time, derate_window).
*   **YDF DataFrames:**
    *   `train_df`: `X_train` and `y_train` are combined into a single DataFrame as required by YDF.
    *   `test_df_predict`: `X_test` is prepared for prediction.
*   **Train Model:**
    *   A `ydf.GradientBoostedTreesLearner` is initialized.
    *   `label`: "derate_window" (the target column).
    *   `task`: `ydf.Task.CLASSIFICATION`.
    *   Hyperparameters are set: `num_trees=500`, `max_depth=10`, `shrinkage=0.1`, `l2_regularization=0.01`, `subsample=0.8`.
    *   The model is trained using `model.train(train_df)`.

| Hyperparameter       | Purpose                                         | Typical Effect                        |
|----------------------|-------------------------------------------------|---------------------------------------|
| `label`              | Target column to predict                        | Defines what the model learns         |
| `task`               | Type of problem (classification/regression)     | Sets loss function/objective          |
| `num_trees`          | Number of trees in the ensemble                 | More trees = better fit, more time    |
| `max_depth`          | Maximum depth of each tree                      | Deeper = more complex, risk overfit   |
| `shrinkage`          | Learning rate (step size for each tree)         | Lower = slower, more robust           |
| `l2_regularization`  | L2 penalty on weights                           | Reduces overfitting                   |
| `subsample`          | Fraction of data used per tree                  | Adds randomness, reduces overfitting  |

## 10. Prediction

*   **Predict Probabilities:** The trained model (`model.predict(test_df_predict)`) is used to get probability predictions for the positive class on the test set.
*   **Convert to Classes:** These probabilities are converted to binary class predictions (`y_pred_class`) using a threshold of 0.5 (probability > 0.5 means predicted class 1, else 0).

## 11. Results Analysis Preparation

*   **`test_results` DataFrame:**
    *   Starts as a copy of `original_test_info`.
    *   The `predicted_derate` (from `y_pred_class`) is added as a new column, ensuring index alignment with `X_test`.
*   **`time_until_trigger`:** Calculates the time difference between the `EventTimeStamp` of a fault in the test set and its corresponding `next_trigger_time` (the actual time of the derate event).

## 12. Custom Evaluation Metrics - Derate Gaps

*   **Identify Actual Triggers in Test Set:**
    *   `actual_triggers`: A DataFrame containing only the rows from `joined_post_2019` where `spn == target_spn` (i.e., actual derate events).
    *   It's sorted by `EquipmentID` and `EventTimeStamp`.
*   **Calculate `derate_gap`:** For these `actual_triggers`, the time difference (`EventTimeStamp.diff()`) since the previous actual trigger for the same `EquipmentID` is calculated.
*   **Merge `derate_gap` into `test_results`:** The `derate_gap` information is merged into the `test_results` DataFrame. The merge is based on `EquipmentID` and matching `test_results.next_trigger_time` with `actual_triggers.EventTimeStamp`. This associates each predicted window with the gap preceding the actual derate event it's supposed to predict.

## 13. Custom Evaluation Metrics - Identifying Valuable True Positives (Savings)

*   **Objective:** To count True Positives (TPs) that provide actual value (e.g., allow for proactive maintenance).
*   **Criteria for a "Valuable TP":**
    1.  `predicted_derate == 1` (model predicted a derate).
    2.  `derate_window == 1` (it was actually a derate window; standard TP).
    3.  `time_until_trigger` is not NaT and is greater than 2 hours (prediction was made more than 2 hours before the actual derate).
    4.  The actual derate event (`next_trigger_time`) either:
        *   Had a `derate_gap` that is `NaT` (it's the first recorded derate for that equipment in the test set).
        *   OR had a `derate_gap` greater than `derate_reset_period` (24 hours), meaning it's a distinct derate event not closely following a previous one.
*   **Count Unique Valuable TP Events:**
    *   `valuable_TPs`: A DataFrame containing all rows from `test_results` that meet the above criteria.
    *   `valuable_TP_events`: `valuable_TPs` is further processed to count only unique *actual derate events* that were successfully predicted early. This is done by `drop_duplicates` based on `EquipmentID` and `next_trigger_time`.
    *   `valuable_TP_count`: The number of such unique, valuably predicted derate events.

## 14. Custom Evaluation Metrics - Identifying Costly False Positives (Costs)

*   **Objective:** To count False Positives (FPs) that would likely lead to unnecessary costs (e.g., premature service).
*   **Initial False Positives:**
    *   `false_positives`: Rows from `test_results` where `predicted_derate == 1` but `derate_window == 0`.
*   **Time to Nearest Actual Trigger:**
    *   `actual_trigger_times_map`: A dictionary mapping each `EquipmentID` to a list of its actual derate `EventTimeStamp`s from the `actual_triggers` DataFrame.
    *   `time_to_nearest_trigger` function: For a given FP row, this function calculates the minimum absolute time difference between the FP's `EventTimeStamp` and any of the actual derate trigger times for that same `EquipmentID`.
    *   If FPs exist, `false_positives['time_to_nearest_actual']` is calculated.
*   **Initial Costly FPs:**
    *   `initial_costly_FPs`: Filters `false_positives` to keep only those where `time_to_nearest_actual` is greater than `derate_reset_period` (24 hours). These are FPs that are not close to any actual derate event.
*   **Filter Clustered Costly FPs:**
    *   If `initial_costly_FPs` is not empty:
        *   It's sorted by `EquipmentID` and `EventTimeStamp`.
        *   `time_since_last_costly_fp`: Calculates the time difference since the *previous costly FP* for the same `EquipmentID`.
        *   `final_costly_FPs`: Keeps an initial costly FP if it's the first one for the equipment (`time_since_last_costly_fp` is NaT) OR if it occurred more than 24 hours after the previous costly FP for that equipment. This avoids over-penalizing clusters of FPs that might relate to a single underlying issue or period of misprediction.
    *   `costly_FP_count`: The number of these `final_costly_FPs`.

## 15. Standard Evaluation Metrics

*   **Prepare Data for YDF Evaluate (Optional but good practice):**
    *   `test_df_eval`: `X_test` is combined with `y_test` (actual labels).
*   **Macro F1 Score (sklearn):** Calculated using `f1_score(y_test, y_pred_class, average="macro")`.
*   **Classification Report (sklearn):** Printed using `classification_report(y_test, y_pred_class)`. This provides precision, recall, F1-score, and support for each class.
*   **Confusion Matrix (sklearn):**
    *   Calculated using `confusion_matrix(y_test, y_pred_class)`.
    *   Displayed using `ConfusionMatrixDisplay`.
    *   Counts for standard True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) are extracted. Note that this `TP_standard` refers to *all* predictions correctly identifying the `derate_window`, without the additional "valuable" or "timing" criteria.
*   **Net Savings (Standard):** A simple net savings calculation based on `TP_standard` (worth $4000) and `FP_standard` (costing $500).

## 16. Final Cost/Savings Analysis (Custom)

*   **Calculate Savings:** `valuable_TP_count * 4000`.
*   **Calculate Costs:** `costly_FP_count * 500`.
*   **Calculate Net Savings:** `Savings - Costs`.
*   These values, based on the custom definitions of valuable TPs and costly FPs, are printed.

## 17. Display Plot

*   `plt.show()`: Displays the confusion matrix plot generated earlier.

This script performs a comprehensive workflow from data ingestion and cleaning to model training and a nuanced, business-logic-driven evaluation of the model's performance in predicting vehicle derate events.

## output from model: 

```
--------SHAPE OF FAULTS--------
(1187335, 10)
--------NaNs--------
Id         0
Name       0
Value      0
FaultId    0
dtype: int64

len(Id): 12821626
N unique_Id: 12821626

--------RECORD ID vs FAULT ID--------
n_unique FaultID: 1187335
n_unique RecordID: 1187335
Labeling faults near service stations...

Done!
Faults within 1km of service station labeled in 'joined'.
When filtered, this removes 129284 rows

Number of rows after filtering active=False out: 549008
Rows removed: 638327
Sorting data by EquipmentID and EventTimeStamp...
Sorting complete.
Calculating next trigger time...
Labeling derate window...

Verification:
Value counts for 'derate_window':
 derate_window
False    548064
True        944
Name: count, dtype: int64

Value counts for 'spn' (to confirm target SPN exists):
 spn
111       174649
929       115784
96         44398
829        43350
639        16172
           ...
521032         1
4380           1
781            1
442            1
54478          1
Name: count, Length: 420, dtype: int64

Sample rows where derate_window is True:
       EquipmentID      EventTimeStamp   spn   next_trigger_time
996835   105349576 2018-07-06 09:42:48  5246 2018-07-06 09:42:48
972882   105427203 2018-04-27 06:07:55  5246 2018-04-27 06:07:55
5712          1329 2015-02-25 13:53:08  4344 2015-02-25 13:53:08
5713          1329 2015-02-25 13:53:08  5246 2015-02-25 13:53:08
83425         1339 2015-06-12 15:35:22  5246 2015-06-12 15:35:22
1001106          NaN
358800           NaN
927313           NaN
936445     1474291.0
936451         224.0
             ...
1157136        478.0
4245             NaN
4427          6371.0
6438        164454.0
4952             NaN
Name: time_since_last_fault, Length: 549008, dtype: float64
EquipmentID                         object
EventTimeStamp              datetime64[ns]
next_trigger_time           datetime64[ns]
window_start_time           datetime64[ns]
spn                                  int64
fmi                                  int64
active                                bool
derate_window                         bool
time_since_last_fault              float64
fault_frequency                      int64
Latitude                           float64
Longitude                          float64
nearStation                           bool
Speed                               object
BarometricPressure                  object
EngineCoolantTemperature            object
EngineLoad                          object
EngineOilPressure                   object
EngineOilTemperature                object
EngineRpm                           object
EngineTimeLtd                       object
FuelLtd                             object
FuelRate                            object
FuelTemperature                     object
Throttle                            object
TurboBoostPressure                  object
dtype: object
EquipmentID                         object
EventTimeStamp              datetime64[ns]
next_trigger_time           datetime64[ns]
window_start_time           datetime64[ns]
spn                                  int64
fmi                                  int64
active                               int64
derate_window                        int64
time_since_last_fault              float64
fault_frequency                      int64
Latitude                           float64
Longitude                          float64
nearStation                          int64
Speed                              float64
BarometricPressure                 float64
EngineCoolantTemperature           float64
EngineLoad                         float64
EngineOilPressure                  float64
EngineOilTemperature               float64
EngineRpm                          float64
EngineTimeLtd                      float64
FuelLtd                            float64
FuelRate                           float64
FuelTemperature                    float64
Throttle                           float64
TurboBoostPressure                 float64
dtype: object
EquipmentID                      0
EventTimeStamp                   0
next_trigger_time           418880
window_start_time           418880
spn                              0
fmi                              0
active                           0
derate_window                    0
time_since_last_fault         1062
fault_frequency                  0
Latitude                         0
Longitude                        0
nearStation                      0
Speed                        19342
BarometricPressure           17717
EngineCoolantTemperature     17730
EngineLoad                   18198
EngineOilPressure            17611
EngineOilTemperature         19249
EngineRpm                    17295
EngineTimeLtd                21574
FuelLtd                      18263
FuelRate                     18467
FuelTemperature             277724
Throttle                    173300
TurboBoostPressure           19814
dtype: int64
Throttle
100.0    254595
0.0      101021
38.8        226
38.4        225
37.6        221
          ...
87.6         16
87.2         16
82.4         16
84.4         15
88.0         13
Name: count, Length: 251, dtype: int64
EquipmentID                      0
EventTimeStamp                   0
next_trigger_time           418880
window_start_time           418880
spn                              0
fmi                              0
active                           0
derate_window                    0
time_since_last_fault            0
fault_frequency                  0
Latitude                         0
Longitude                        0
nearStation                      0
Speed                            0
BarometricPressure               0
EngineCoolantTemperature         0
EngineLoad                       0
EngineOilPressure                0
EngineOilTemperature             0
EngineRpm                        0
EngineTimeLtd                    0
FuelLtd                          0
FuelRate                         0
FuelTemperature                  0
Throttle                         0
TurboBoostPressure               0
dtype: int64
Goal for predicting derates: 43 derates to predict
Preparing training (pre-2019) and testing (post-2019) data...
Features with high correlation (>0.9) with others: ['FuelRate']
Training data shape: (492397, 17)
Test data shape: (56611, 17)

Starting model training with tuned hyperparameters...
Train model on 492397 examples
Model trained in 0:00:24.124655
Model training complete.

Making predictions on the test set...
Preparing results dataframe for detailed analysis...
Calculating time gaps between actual derate events...
Derate gap calculation complete.

--- Identifying Valuable True Positives (Savings Calculation) ---
Found 0 unique actual derate events predicted >2 hours early with >1 days 00:00:00 gap.

--- Identifying Costly False Positives (Cost Calculation) ---
Calculating time difference between false positives and nearest actual derate...
Found 8 individual FP rows > 1 days 00:00:00 from any actual derate.
Filtering clustered costly FPs (keeping only those > 1 days 00:00:00 apart)...
Found 3 final costly False Positive events (separated by > 1 days 00:00:00).

--- Standard Evaluation Metrics ---

Calculating macro F1 score (sklearn)...
Macro F1 Score: 0.8598

Classification Report (sklearn):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56523
           1       0.87      0.61      0.72        88

    accuracy                           1.00     56611
   macro avg       0.94      0.81      0.86     56611
weighted avg       1.00      1.00      1.00     56611


--- Feature Importances (YDF Model) ---
     importance                 attribute
0   1448.630960                       spn
1    142.768980             EngineTimeLtd
2     76.703217                   FuelLtd
3     71.113743      EngineOilTemperature
4     59.218517     time_since_last_fault
5     48.859996           fault_frequency
6     44.221195                       fmi
7     42.081274           FuelTemperature
8     39.203473         EngineOilPressure
9     25.106245                 EngineRpm
10    21.467151                  FuelRate
11    21.320740  EngineCoolantTemperature
12    20.353880                EngineLoad
13    19.430094                     Speed
14    17.301844        TurboBoostPressure
15    14.441224        BarometricPressure
16    12.348499                  Throttle

Creating Confusion Matrix (sklearn)...
Standard Confusion Matrix Counts:
  True Negatives (TN): 56515
  False Positives (FP): 8
  False Negatives (FN): 34
  True Positives (TP): 54  <-- target for this is derate window

Net savings based on standard evaluation (all TPs, all FPs): $212000

Goal for predicting derates (from previous script): 43 derates to predict
Actual correctly predicted derates: 0 :(

--- Final Cost/Savings Analysis ---
Valuable True Positives (Savings): 0
Costly False Positives (Costs): 3
Total Savings: $0
Total Costs: $1500
Net Savings (Custom Definition): $-1500
```
