import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

"""
Relative data directory
"""
DATA_DIR = "../data/VTI/T3.2"
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
        # df_surround = load_csv(find_file_with_id(file_id, DATA_DIR + SURROUND_DIR), seperator=";") # TODO: Surround is really messy with a dynamic amount of actors.
        # df_surround = df_surround.rename(columns={'timer': 'time'})

        data.append((df_ego, df_phys, None))
        found_pairs += 1
        if found_pairs == max_pairs:
            break

    print(f"\n\n======== Found {found_pairs} of pairs. ========\n\n")
    return data

def parse_road_actors(df_surround):
    print
    # timer(42.005), system_timestamp (240318 155930.123) , scene (17); Rest in this format repeating: bike_boy1;Road 358;358;357.000;-174.400;2.000;-4.400;-1;0.000;0.000;0.000;0.000;0.000;0.000;0.000;0;0;0;0;0;0;0;0;275; NEXT
    # 17 Seems to stand for the amount of variables per object?
    """
    
    """
    pass

"""
Derives/Calculates Heart Rate and RR Interval from the csv
"""
def process_physiology_data(df: pd.DataFrame) -> pd.DataFrame:

    print("Processing physiological data...")

    # Simple thresholding fallback (Mean + 3*STD)
    threshold = df['ecg'].mean() + (df['ecg'].std() * 3)
    df['beats'] = (df['ecg'] > threshold).astype(int)

    # Calculate RR Intervals (Time between heart beats)
    beat_indices = df[df['beats'] == 1].index # 0 FALSE : No Beat, 1 TRUE: Heart Beat

    # Calculate difference in 'time' between beats
    # ASSUMPTION: 'time' is in nanoseconds. If timestamps are seconds, remove / 1e9
    beat_times = df.loc[beat_indices, 'time']
    rr_intervals_ns = beat_times.diff() 
    
    # Convert to seconds (Adjust 1e9 if your time is already in ms or seconds)
    rr_intervals_sec = rr_intervals_ns / 1e9 
    
    # Calculate Instantaneous BPM (60 / seconds)
    bpm_values = 60 / rr_intervals_sec
    
    # 3. Map these values back to the original dataframe at the beat locations
    df.loc[beat_indices, 'rrInterval'] = rr_intervals_sec
    df.loc[beat_indices, 'heartRate'] = bpm_values
    
    # 4. Forward Fill to create a continuous signal
    # Heart rate persists until the next beat changes it
    df['heartRate'] = df['heartRate'].fillna(method='ffill')
    df['rrInterval'] = df['rrInterval'].fillna(method='ffill')
    
    # Fill any leading NaNs (before first beat)
    df['heartRate'] = df['heartRate'].fillna(method='bfill')
    df['rrInterval'] = df['rrInterval'].fillna(method='bfill')
    
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
                func = interp1d(df["time"], df[col], interpolation_mode=interpolation_mode, fill_value="extrapolate")
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
# TODO: Calculate the amount of beats for Physiology

if __name__ == "__main__":
    df_surround = load_csv("data/VTI/T3.2/Surrounding vehicle data/i4driving_roadA_scene_fp03_roadA_fp3_1710765373441.csv", seperator=";")
    print(df_surround.columns)