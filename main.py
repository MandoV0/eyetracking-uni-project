from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
PIPELINE:
raw signals
=> rule-based labeling (pseudo ground truth as we are missing necessary labels for training)
=> windowed feature extraction
=> Random Forest (3-classes, Attentive, Inattentive, Aggresive)
"""

# CONFIG
EGO_DATA_DIR = Path("data/VTI/T3.2/Ego vehicle data").resolve()
WINDOW = 250  # 5 seconds @ 50 Hz
RANDOM_STATE = 42

# DATA LOADING
def read_data() -> pd.DataFrame:
    csv_files = list(EGO_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Ego vehicle data directory")
    return pd.read_csv(csv_files[0])

# CLEANING
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix sentinel values
    if "aheadTHW" in df.columns:
        df["aheadTHW"] = df["aheadTHW"].replace(-1, 100.0)

    # Drop rows only if critical signals are missing
    df = df.dropna(subset=[
        "oveBodyVelocityX",
        "oveBodyAccelerationLongitudinalX",
        "steeringWheelAngle",
        "aheadTHW"
    ])

    return df

# RULE-BASED LABELING
def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = "Attentive"

    # INATTENTION (based on duration)
    ON_ROAD = df["openxrGazeWorldModel"].isin(["Road", "Windshield"])
    OFF_ROAD = ~ON_ROAD # On Road is a boolean series, ~ to invert

    # Count consecutive off road samples
    df["off_road_run"] = (
        OFF_ROAD
        .astype(int)
        .groupby((~OFF_ROAD).cumsum())
        .cumsum()
    )

    # >= 2 seconds 50 Hz
    inattentive_mask = df["off_road_run"] >= 100 # How long are we looking of road? 2 seconds => Innatentive
    df.loc[inattentive_mask, "Label"] = "Inattentive"

    # AGGRESSION (physics/speed based)
    aggressive_mask = (
        (df["oveBodyAccelerationLongitudinalX"] < -3.5) | # Harsh breaking - 3.5m/s
        (df["oveBodyJerkLongitudinalX"].abs() > 5.0) |    # Jerk is sudden movement
        (df["aheadTHW"] < 0.6) |                          # Tailgaiting
        ((df["brakePedalActive"] == True) & (df["brakeForce"] > 0.7))
    )

    # Aggressive overrides inattentive behavior
    df.loc[aggressive_mask, "Label"] = "Aggressive"

    """
    Maybe add Indicator
    """

    return df


# MAIN PIPELINE

# 1. Load
df = read_data()

# 2. Clean
df = clean_data(df)

# 3. Create labels
df = create_labels(df)

# 4. Encode labels as numbers for Random Forest
label_map = {
    "Attentive": 0,
    "Inattentive": 1,
    "Aggressive": 2
}
df["label"] = df["Label"].map(label_map)

print("Label distribution (raw samples):")
print(df["Label"].value_counts(), "\n")

# FEATURE WINDOWING

features = df.rolling(WINDOW).agg({
    "oveBodyVelocityX": ["mean", "std"],
    "oveBodyAccelerationLongitudinalX": ["min", "max", "std"],
    "steeringWheelAngle": ["std"],
    "aheadTHW": ["min"]
})

# Flatten column names
features.columns = ["_".join(col) for col in features.columns]

# Window label = majority class
features["label"] = (
    df["label"]
    .rolling(WINDOW)
    .apply(lambda x: x.mode().iloc[0])
)

features.dropna(inplace=True)

print("Label distribution (windows):")
print(features["label"].value_counts(), "\n")

# TRAIN RANDOM FOREST

X = features.drop(columns="label")
y = features["label"]

# train_test_split()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1 # Parallel Processing for faster training, should use all cpu cores
)

rf.fit(X, y)

print("Random Forest trained")

# FEATURE IMPORTANCE

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)











