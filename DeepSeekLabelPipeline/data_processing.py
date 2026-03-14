import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from FeatureEngineer import FeatureEngineer


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "../data/VTI/T3.2")
EGO_DIR = "/Ego vehicle data/"
PHYS_DIR = "/Physiology/"
SURROUND_DIR = "/Surrounding vehicle data/"

MERGE = 0
LINEAR = 1
SPLINE = 2


def load_csv(filepath, separator=","):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=separator)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def parse_surrounding_vehicle_data(surround_path):
    records = []
    num_vars_per_actor = 24

    with open(surround_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split(";")
            if len(fields) < 4 or fields[0] in ("timer", ""):
                continue
            try:
                time = float(fields[0])
                actor_data = fields[3:]
                for idx in range(0, len(actor_data), num_vars_per_actor):
                    chunk = actor_data[idx : idx + num_vars_per_actor]
                    if len(chunk) < num_vars_per_actor:
                        continue
                    records.append(
                        {
                            "time": time,
                            "actor_id": chunk[0],
                            "type": chunk[1],
                            "x": float(chunk[4]) if chunk[4] else 0.0,
                            "y": float(chunk[5]) if chunk[5] else 0.0,
                            "z": float(chunk[6]) if chunk[6] else 0.0,
                            "vx": float(chunk[12]) if chunk[12] else 0.0,
                            "vy": float(chunk[13]) if chunk[13] else 0.0,
                            "speed": float(chunk[14]) if chunk[14] else 0.0,
                        }
                    )
            except ValueError:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"Parsed {len(df)} surrounding actor records.")
    return df


def find_pairs(max_pairs=10):
    def find_file_with_id(file_id, dir_path):
        for file_name in os.listdir(dir_path):
            if file_name.split("_")[-1] == file_id:
                return dir_path + file_name
        return None

    ego_dir = DATA_DIR + EGO_DIR
    phys_dir = DATA_DIR + PHYS_DIR
    surround_dir = DATA_DIR + SURROUND_DIR

    if not os.path.isdir(ego_dir):
        raise FileNotFoundError(f"Ego data directory not found: {ego_dir}")

    data = []
    for file_name in os.listdir(ego_dir):
        file_id = file_name.split("_")[-1]
        ego_path = find_file_with_id(file_id, ego_dir)
        phys_path = find_file_with_id(file_id, phys_dir)
        surround_path = find_file_with_id(file_id, surround_dir)

        if not ego_path or not phys_path or not surround_path:
            continue

        df_ego = load_csv(ego_path)
        df_phys = load_csv(phys_path)
        df_surround = parse_surrounding_vehicle_data(surround_path)
        data.append((df_ego, df_phys, df_surround))
        if len(data) >= max_pairs:
            break

    print(f"\n\n======== Found {len(data)} pairs. ========\n\n")
    return data


def process_physiology_data(df):
    print("Processing physiological data...")
    threshold = df["ecg"].mean() + (df["ecg"].std() * 3)
    df["beats_raw"] = (df["ecg"] > threshold).astype(int)
    df["beats"] = ((df["beats_raw"] == 1) & (df["beats_raw"].shift(1) == 0)).astype(int)

    beat_indices = df[df["beats"] == 1].index
    beat_times = df.loc[beat_indices, "time"]
    rr_intervals_sec = beat_times.diff()
    bpm_values = 60 / rr_intervals_sec

    df.loc[beat_indices, "rrInterval"] = rr_intervals_sec
    df.loc[beat_indices, "heartRate"] = bpm_values

    df["heartRate"] = df["heartRate"].ffill().bfill()
    df["rrInterval"] = df["rrInterval"].ffill().bfill()

    print("Physiological data processed: 'heartRate' and 'rrInterval' columns added.")
    return df


def sync_all(df_ego, df_phys, df_surround, method=MERGE, tolerance=0.05):
    df_ego = df_ego.sort_values("time").reset_index(drop=True)
    merged = df_ego.copy()

    def interpolate(df, interpolation_mode, name_prefix):
        df = df.sort_values("time").reset_index(drop=True)
        interp_df = pd.DataFrame({"time": df_ego["time"]})
        for col in df.columns:
            if col == "time":
                continue
            try:
                func = interp1d(df["time"], df[col], kind=interpolation_mode, fill_value="extrapolate")
                interp_df[col] = func(df_ego["time"])
            except Exception as exc:
                print(f"Error: Could not interpolate column '{col}' from {name_prefix}: {exc}")
        return interp_df.drop(columns=["time"])

    if df_phys is not None:
        df_phys = process_physiology_data(df_phys)
        if method == MERGE:
            merged = pd.merge_asof(merged, df_phys, on="time", direction="nearest", tolerance=tolerance)
        elif method == LINEAR:
            merged = pd.concat([merged, interpolate(df_phys, "linear", "Phys")], axis=1)
        elif method == SPLINE:
            merged = pd.concat([merged, interpolate(df_phys, "cubic", "Phys")], axis=1)

    if df_surround is not None and not df_surround.empty:
        print("Processing surrounding vehicle data for sync...")
        surround_with_ego = pd.merge_asof(
            df_surround.sort_values("time"),
            df_ego[["time", "oveInertialPositionX", "oveInertialPositionY", "oveInertialPositionZ"]].sort_values("time"),
            on="time",
            direction="nearest",
        )
        surround_with_ego["dist"] = np.sqrt(
            (surround_with_ego["x"] - surround_with_ego["oveInertialPositionX"]) ** 2
            + (surround_with_ego["y"] - surround_with_ego["oveInertialPositionY"]) ** 2
            + (surround_with_ego["z"] - surround_with_ego["oveInertialPositionZ"]) ** 2
        )
        surround_with_ego = surround_with_ego[surround_with_ego["dist"] > 0.1]

        surround_features = surround_with_ego.groupby("time").agg(
            surround_actor_count=("actor_id", "count"),
            surround_min_dist=("dist", "min"),
            surround_avg_speed=("speed", "mean"),
        ).reset_index()

        merged = pd.merge_asof(merged, surround_features, on="time", direction="nearest", tolerance=tolerance)
        print("Surrounding vehicle features added: 'surround_actor_count', 'surround_min_dist', 'surround_avg_speed'.")

    return merged


def build_features_from_raw(max_files=None, interpolation_mode=LINEAR, window_size=50):
    print("Building unlabeled feature dataset from RAW data...")
    max_pairs = max_files if max_files is not None else 1000
    pairs = find_pairs(max_pairs=max_pairs)

    combined_sessions = []
    session_ids = []
    for idx, pair in enumerate(pairs):
        df_ego, df_phys, df_surround = pair
        print(f"\nProcessing RAW session {idx + 1}/{len(pairs)}...")
        df_synced = sync_all(df_ego, df_phys, df_surround, method=interpolation_mode)
        combined_sessions.append(df_synced)
        session_ids.extend([idx] * len(df_synced))

    if not combined_sessions:
        raise ValueError("No raw sessions could be processed.")

    df_all = pd.concat(combined_sessions, ignore_index=True)
    session_ids = np.asarray(session_ids)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df_all, window_size=window_size)
    features["session_id"] = session_ids
    features.insert(0, "row_id", np.arange(len(features), dtype=int))
    return features


def bootstrap_unlabeled_from_processed_csv(processed_csv, out_path=None, max_rows=None):
    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    print(f"Bootstrapping unlabeled features from existing processed dataset: {processed_csv}")
    df = pd.read_csv(processed_csv, nrows=max_rows)

    drop_cols = [col for col in ["_label", "aggression_score", "inattention_score"] if col in df.columns]
    features = df.drop(columns=drop_cols).copy()

    if "row_id" not in features.columns:
        features.insert(0, "row_id", np.arange(len(features), dtype=int))

    if "session_id" not in features.columns:
        raise ValueError("Bootstrapped dataset must contain 'session_id'.")

    if out_path:
        save_unlabeled_features_csv(features, out_path)
    return features


def save_unlabeled_features_csv(features, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Saved unlabeled features to: {out_path}")


def load_unlabeled_features_csv(csv_path, max_rows=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Unlabeled feature CSV not found: {csv_path}")

    print(f"Loading unlabeled features from: {csv_path}")
    features = pd.read_csv(csv_path, nrows=max_rows)
    if "row_id" not in features.columns:
        features.insert(0, "row_id", np.arange(len(features), dtype=int))
    if "session_id" not in features.columns:
        raise ValueError("Unlabeled feature CSV must contain 'session_id'.")
    print(f"Loaded {len(features)} unlabeled rows with {len(features.columns)} columns")
    return features


def save_labeled_dataset_csv(features, labels, out_path, label_col="_label"):
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same number of rows.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out = features.copy()
    df_out[label_col] = np.asarray(labels, dtype=int)
    df_out.to_csv(out_path, index=False)
    print(f"Saved labeled dataset to: {out_path}")


def load_labeled_dataset_csv(csv_path, max_rows=None, label_col="_label"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled dataset not found: {csv_path}")

    print(f"Loading labeled dataset: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)
    if label_col not in df.columns:
        raise ValueError(f"Labeled CSV must contain '{label_col}'.")
    if "session_id" not in df.columns:
        raise ValueError("Labeled CSV must contain 'session_id'.")

    labels = df[label_col].to_numpy(dtype=int)
    session_ids = df["session_id"].to_numpy()
    features = df.drop(columns=[label_col]).copy()
    return features, labels, session_ids, df
