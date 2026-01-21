import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from FeatureEngineer import FeatureEngineer
from AbsoluteThresholdLabeler import AbsoluteThresholdLabeler

# Paths relative to this pipeline
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "../data/VTI/T3.2")
EGO_DIR = "/Ego vehicle data/"
PHYS_DIR = "/Physiology/"
SURROUND_DIR = "/Surrounding vehicle data/"

MERGE = 0
LINEAR = 1
SPLINE = 2

def load_csv(filepath: str, seperator=","):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=seperator)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def find_pairs(max_pairs=10):
    """
    Locate matching ego/phys/surround files by ID suffix and load them.
    Returns list of tuples: (df_ego, df_phys, df_surround).
    """

    def find_file_with_id(id: str, dir_path: str) -> str:
        files = os.listdir(dir_path)
        for file in files:
            file_id = file.split("_")[-1]
            if file_id == id:
                return dir_path + file
        return "ERROR! No matching files!"

    ego_files = os.listdir(DATA_DIR + EGO_DIR)
    found_pairs = 0
    data = []
    for file in ego_files:
        file_id = file.split("_")[-1]
        df_ego = load_csv(find_file_with_id(file_id, DATA_DIR + EGO_DIR))
        df_phys = load_csv(find_file_with_id(file_id, DATA_DIR + PHYS_DIR))
        # Surround uses ";" as separator for some reason?
        df_surround = parse_surrounding_vehicle_data(find_file_with_id(file_id, DATA_DIR + SURROUND_DIR))

        data.append((df_ego, df_phys, df_surround))
        found_pairs += 1
        if found_pairs == max_pairs:
            break

    print(f"\n\n======== Found {found_pairs} of pairs. ========\n\n")
    return data


def parse_surrounding_vehicle_data(surround_path):
    """
    Parse surrounding vehicle data from i4driving CSV with dynamic actor columns.
    Returns pd.DataFrame with columns: time, actor_id, type, x, y, z, vx, vy, speed
    """
    records = []
    num_vars_per_actor = 24  # number of variables per actor in i4driving format

    with open(surround_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(';')
            if len(fields) < 4:
                continue
            if fields[0] == "timer" or fields[0] == "":
                continue
            try:
                time = float(fields[0])
                actor_data = fields[3:]
                for i in range(0, len(actor_data), num_vars_per_actor):
                    chunk = actor_data[i:i + num_vars_per_actor]
                    if len(chunk) < num_vars_per_actor:
                        continue
                    records.append({ # This is a Guess
                        'time': time,
                        'actor_id': chunk[0],
                        'type': chunk[1],
                        'x': float(chunk[4]) if chunk[4] else 0.0,
                        'y': float(chunk[5]) if chunk[5] else 0.0,
                        'z': float(chunk[6]) if chunk[6] else 0.0,
                        'vx': float(chunk[12]) if chunk[12] else 0.0,
                        'vy': float(chunk[13]) if chunk[13] else 0.0,
                        'speed': float(chunk[14]) if chunk[14] else 0.0
                    })
            except ValueError:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"Parsed {len(df)} surrounding actor records.")
    return df


def process_physiology_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive heartRate and rrInterval from ECG by simple peak detection and forward-fill.
    """
    print("Processing physiological data...")

    threshold = df['ecg'].mean() + (df['ecg'].std() * 3)
    df['beats_raw'] = (df['ecg'] > threshold).astype(int)
    df['beats'] = ((df['beats_raw'] == 1) & (df['beats_raw'].shift(1) == 0)).astype(int)

    beat_indices = df[df['beats'] == 1].index
    beat_times = df.loc[beat_indices, 'time']
    rr_intervals_sec = beat_times.diff()
    bpm_values = 60 / rr_intervals_sec

    df.loc[beat_indices, 'rrInterval'] = rr_intervals_sec
    df.loc[beat_indices, 'heartRate'] = bpm_values

    df['heartRate'] = df['heartRate'].ffill().bfill()
    df['rrInterval'] = df['rrInterval'].ffill().bfill()

    print("Physiological data processed: 'heartRate' and 'rrInterval' columns added.")
    return df


def sync_all(df_ego: pd.DataFrame, df_phys: pd.DataFrame, df_surround: pd.DataFrame, method: int = MERGE, tolerance: float = 0.05) -> pd.DataFrame:
    """
    Sync EGO, Physiology, and Surrounding data to EGO timestamps.
    method:
        MERGE (0)  -> nearest merge_asof
        LINEAR (1) -> linear interpolation
        SPLINE (2) -> cubic spline interpolation
    """
    df_ego = df_ego.sort_values("time").reset_index(drop=True)
    merged = df_ego.copy()

    def interpolate(df: pd.DataFrame, interpolation_mode: str, name_prefix: str) -> pd.DataFrame:
        df = df.sort_values("time").reset_index(drop=True)
        interp_df = pd.DataFrame({"time": df_ego["time"]})
        for col in df.columns:
            if col == "time":
                continue
            try:
                func = interp1d(df["time"], df[col], kind=interpolation_mode, fill_value="extrapolate")
                interp_df[col] = func(df_ego["time"])
            except Exception as e:
                print(f"Error: Could not interpolate column '{col}' from {name_prefix}: {e}")
        return interp_df.drop(columns=["time"])

    # Merge or interpolate Phys
    if df_phys is not None:
        df_phys = process_physiology_data(df_phys)
        if method == MERGE:
            merged = pd.merge_asof(merged, df_phys, on="time", direction="nearest", tolerance=tolerance)
        elif method == LINEAR:
            phys_interp = interpolate(df_phys, interpolation_mode="linear", name_prefix="Phys")
            merged = pd.concat([merged, phys_interp], axis=1)
        elif method == SPLINE:
            phys_interp = interpolate(df_phys, interpolation_mode="cubic", name_prefix="Phys")
            merged = pd.concat([merged, phys_interp], axis=1)

    # Merge Surround data (aggregation)
    if df_surround is not None and not df_surround.empty:
        print("Processing surrounding vehicle data for sync...")

        surround_with_ego = pd.merge_asof(
            df_surround.sort_values("time"),
            df_ego[["time", "oveInertialPositionX", "oveInertialPositionY", "oveInertialPositionZ"]].sort_values("time"),
            on="time",
            direction="nearest"
        )

        surround_with_ego["dist"] = np.sqrt(
            (surround_with_ego["x"] - surround_with_ego["oveInertialPositionX"])**2 +
            (surround_with_ego["y"] - surround_with_ego["oveInertialPositionY"])**2 +
            (surround_with_ego["z"] - surround_with_ego["oveInertialPositionZ"])**2
        )

        surround_with_ego = surround_with_ego[surround_with_ego["dist"] > 0.1]

        surround_features = surround_with_ego.groupby("time").agg(
            surround_actor_count=("actor_id", "count"),
            surround_min_dist=("dist", "min"),
            surround_avg_speed=("speed", "mean")
        ).reset_index()

        merged = pd.merge_asof(merged, surround_features, on="time", direction="nearest", tolerance=tolerance)
        print("Surrounding vehicle features added: 'surround_actor_count', 'surround_min_dist', 'surround_avg_speed'.")

    return merged


def load_processed_csv(csv_path, max_rows=None):
    """
    Load preprocessed features+labels CSV.
    Expected columns: feature columns + 'session_id' + '_label'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"Loading processed dataset: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)

    if "_label" not in df.columns or "session_id" not in df.columns:
        raise ValueError("Processed CSV must contain columns: '_label' and 'session_id'.")

    labels = df["_label"].to_numpy()
    session_ids = df["session_id"].to_numpy()

    feature_cols = [c for c in df.columns if c not in ("_label",)]
    features = df[feature_cols].copy()  # keep session_id in features for grouping for example for Cross Validation

    print(f"Samples loaded: {len(df)}")
    print(f"Feature columns: {len([c for c in feature_cols if c != 'session_id'])}")

    return features, labels, session_ids


def build_processed_from_raw(max_files=None, interpolation_mode=LINEAR, window_size=50):
    """
    Build features+labels from raw simulator CSVs.
    Returns (features_df_with_session_id, labels_array, session_ids_array).
    """
    print("Building processed dataset from RAW data...")

    max_pairs = max_files if max_files is not None else 1000
    pairs = find_pairs(max_pairs=max_pairs)

    combined = []
    session_ids = []

    for idx, pair_data in enumerate(pairs):
        # Handle both formats: (ego, phys, surround) or extended tuple
        if len(pair_data) == 3:
            df_ego, df_phys, df_surround = pair_data
        elif len(pair_data) == 5:
            df_ego, df_phys, df_surround, _participant_id, _file_id = pair_data
        else:
            print(f"Warning: unexpected pair format len={len(pair_data)}; skipping.")
            continue

        print(f"\nProcessing RAW session {idx + 1}/{len(pairs)}...")
        df_synced = sync_all(df_ego, df_phys, df_surround, method=interpolation_mode)
        df_synced["session_id"] = idx

        combined.append(df_synced)
        session_ids.extend([idx] * len(df_synced))

    if not combined:
        raise ValueError("No raw sessions could be processed.")

    df_all = pd.concat(combined, ignore_index=True)
    session_ids = np.asarray(session_ids)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df_all, window_size=window_size)
    features["session_id"] = session_ids

    print("\n" + "=" * 70)
    print("LABEL GENERATION")
    print("=" * 70)
    labeler = AbsoluteThresholdLabeler(use_data_driven_thresholds=True)
    labels = labeler.generate_labels(features)

    return features, labels, session_ids


def save_processed_csv(features, labels, out_path):
    """
    Save features + labels to a single CSV.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out = features.copy()
    df_out["_label"] = labels
    df_out.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path}")