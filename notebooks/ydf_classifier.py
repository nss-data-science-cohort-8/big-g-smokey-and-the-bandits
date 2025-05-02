# import libraries
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

# load data
column_dtypes = {
    "EquipmentID": object,
    # "EventTimeStamp": "datetime64[ns]",
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
data = pd.read_csv(
    "../data/model_data.csv",
    dtype=column_dtypes,
    parse_dates=["EventTimeStamp"],
)

predictors = [
    col
    for col in data.columns
    if col not in ["EquipmentID", "EventTimeStamp", "derate_window"]
]
target = "derate_window"
X = data[predictors]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Concatenate features and target for YDF training
train_df = pd.DataFrame(X_train).join(pd.Series(y_train, name="derate_window"))
test_df = pd.DataFrame(X_test).join(pd.Series(y_test, name="derate_window"))
# --- Adjustments for Model Improvement ---
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

# --- Adjustments for Evaluation ---
# Test the model
y_pred_proba = model.predict(test_df)  # Get probability predictions
# are any of these predictions occuring outside the 2 hour mark or derate window?
# need to find a way to match these up with test values to determine the probabilities matched with timestamps and filter out inside 2 hour window.
# Convert probabilities to class predictions using a 0.5 threshold
y_pred_class = y_pred_proba > 0.5

# Evaluate using YDF's built-in evaluation (optional but provides detailed report)
evaluation = model.evaluate(test_df)
print("Full evaluation report: ", evaluation)

# Evaluate using sklearn's f1_score
print("Calculating macro F1 score...")
macro_f1 = f1_score(y_test, y_pred_class, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# Optional: Print classification report and confusion matrix for more detail
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Derate", "Derate"]
)
disp.plot()
plt.title("Confusion Matrix")

cm = pd.DataFrame(cm)
TN = cm.iloc[0, 0].astype(int)
FP = cm.iloc[0, 1].astype(int)
FN = cm.iloc[1, 0].astype(int)
TP = cm.iloc[1, 1].astype(int)
Costs = FP * 500
Savings = TP * 4000
Net = Savings - Costs
print(f"Net savings: ${Net}")

# show confusion matrix plot
plt.show()
