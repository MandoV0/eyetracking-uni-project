import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

"""
Relative data directory
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data/VTI/T3.2")
EGO_DIR = "/Ego vehicle data/"
PHYS_DIR = "/Physiology/"
SURROUND_DIR = "/Surrounding vehicle data/"

def load_csv(filepath: str, seperator=","):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=seperator)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df
"""
This function combines the pairs of data into one csv, assuming that the files have the same name.

Returns: List[ (ego, phys, surround) ] each is a panda dataframe.
"""
def find_pairs(max_pairs=10):

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
        # Surround uses ";" as a seperator while the others use ","
        df_surround = parse_surrounding_vehicle_data(find_file_with_id(file_id, DATA_DIR + SURROUND_DIR)) # TODO: Surround is really messy with a dynamic amount of actors.

        data.append((df_ego, df_phys, df_surround))
        found_pairs += 1
        if found_pairs == max_pairs:
            break

    print(f"\n\n======== Found {found_pairs} of pairs. ========\n\n")
    return data

def parse_surrounding_vehicle_data(surround_path):
    """
    Parse surrounding vehicle data from i4driving CSV with dynamic actor columns.
    
    Returns:
        pd.DataFrame with columns: time, actor_id, type, x, y, z, vx, vy, speed
    """
    records = []
    num_vars_per_actor = 24  # number of variables per actor in i4driving format

    with open(surround_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(';')
            if len(fields) < 4:
                continue

            if fields[0] == "timer" or fields[0] == "": # Skip First line/row
                continue
            
            try:
                time = float(fields[0])
                # fields[1] is system_timestamp
                # fields[2] is num_actors
                actor_data = fields[3:]

                # iterate over actors
                for i in range(0, len(actor_data), num_vars_per_actor):
                    chunk = actor_data[i:i + num_vars_per_actor]
                    if len(chunk) < num_vars_per_actor:
                        continue  # skip incomplete actor data

                    # These are guesses for what everything means.
                    records.append({
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

"""
Derives/Calculates Heart Rate and RR Interval from the csv
"""
def process_physiology_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Processing physiological data...")

    # Simple thresholding fallback (Mean + 3*STD)
    threshold = df['ecg'].mean() + (df['ecg'].std() * 3)
    df['beats_raw'] = (df['ecg'] > threshold).astype(int)
    # Only count the rising edge of a beat
    df['beats'] = ((df['beats_raw'] == 1) & (df['beats_raw'].shift(1) == 0)).astype(int)

    # Calculate RR Intervals (Time between heart beats)
    beat_indices = df[df['beats'] == 1].index # 0 FALSE : No Beat, 1 TRUE: Heart Beat

    # Calculate difference in 'time' between beats
    beat_times = df.loc[beat_indices, 'time']
    rr_intervals_sec = beat_times.diff() 
    
    # Calculate BPM (60 / seconds)
    bpm_values = 60 / rr_intervals_sec
    
    # 3. Map these values back to the original dataframe at the beat locations
    df.loc[beat_indices, 'rrInterval'] = rr_intervals_sec
    df.loc[beat_indices, 'heartRate'] = bpm_values
    
    # Forward Fill to create a continuous signal
    # Heart rate persists until the next beat changes it
    # Fill any leading NaNs (before first beat)
    df['heartRate'] = df['heartRate'].ffill().bfill()
    df['rrInterval'] = df['rrInterval'].ffill().bfill()
    
    print("Physiological data processed: 'heartRate' and 'rrInterval' columns added.")
    return df

MERGE = 0
LINEAR = 1
SPLINE = 2

"""
Sync EGO, Physiology, and Surrounding data to EGO timestamps.

method: MERGE (0) -> nearest merge_asof
    LINEAR (1) -> linear interpolation
    SPLINE (2) -> cubic spline interpolation
"""
def sync_all(df_ego: pd.DataFrame, df_phys: pd.DataFrame, df_surround: pd.DataFrame, method: int = MERGE, tolerance: float = 0.05) -> pd.DataFrame:
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

        # TODO: Improve the parsing
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
        
        # Filter out very small distances (likely ego itself or artifacts)
        # In esmini/i4driving, ego is sometimes in the actor list but often not.
        surround_with_ego = surround_with_ego[surround_with_ego["dist"] > 0.1]
        
        surround_features = surround_with_ego.groupby("time").agg(
            surround_actor_count=("actor_id", "count"),
            surround_min_dist=("dist", "min"),
            surround_avg_speed=("speed", "mean")
        ).reset_index()
        
        merged = pd.merge_asof(merged, surround_features, on="time", direction="nearest", tolerance=tolerance)
        print("Surrounding vehicle features added: 'surround_actor_count', 'surround_min_dist', 'surround_avg_speed'.")

    return merged

"""
method:
    MERGE (0) -> nearest merge_asof
    LINEAR (1) -> linear interpolation
    SPLINE (2) -> cubic spline interpolation
"""
def execute(max_pairs=1, interpolation_mode: int = MERGE) -> pd.DataFrame:
    data = find_pairs(max_pairs=max_pairs)
    return sync_all(data[0][0], data[0][1], data[0][2], interpolation_mode)

# TODO: Clean up Surrounding Vehicle data so it can be used.