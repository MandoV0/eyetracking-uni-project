import pandas as pd
import os
from scipy.interpolate import interp1d

"""
Relative data directory
"""
DATA_DIR = "data/VTI/T3.2"
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