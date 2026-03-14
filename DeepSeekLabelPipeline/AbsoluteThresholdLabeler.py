import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


class AbsoluteThresholdLabeler:
    def __init__(self, use_data_driven_thresholds=True):
        self.use_data_driven = use_data_driven_thresholds
        self.thresholds = {}

    def _analyze_data_distribution(self, features):
        print("Analyzing data distribution to set thresholds...")

        thresholds = {}

        if "jerk_long_max" in features.columns:
            thresholds["jerk_high"] = np.percentile(features["jerk_long_max"].abs(), 90)
        if "accel_long_std" in features.columns:
            thresholds["accel_std_high"] = np.percentile(features["accel_long_std"], 85)
        if "steering_rate" in features.columns:
            thresholds["steering_rate_high"] = np.percentile(features["steering_rate"], 85)
        if "throttle_changes" in features.columns:
            thresholds["throttle_changes_high"] = np.percentile(features["throttle_changes"], 85)

        if "gaze_off_road_ratio" in features.columns:
            thresholds["gaze_off_road_high"] = np.percentile(features["gaze_off_road_ratio"], 80)
        if "lateral_pos_std" in features.columns:
            thresholds["lateral_std_high"] = np.percentile(features["lateral_pos_std"], 85)
        if "ndrt_error_rate" in features.columns:
            thresholds["ndrt_error_high"] = np.percentile(features["ndrt_error_rate"], 80)
        if "eyelid_low_ratio" in features.columns:
            thresholds["eyelid_low_high"] = np.percentile(features["eyelid_low_ratio"], 80)

        if "hr_mean" in features.columns:
            hr_mean = features["hr_mean"].mean()
            hr_std = features["hr_mean"].std()
            thresholds["hr_elevated"] = hr_mean + 1.5 * hr_std

        self.thresholds = thresholds
        print(f"Set {len(thresholds)} thresholds based on data distribution")
        return thresholds

    def generate_labels(self, features, event_persistence_seconds=2.0, sample_rate_hz=10):
        print("Generating labels using absolute thresholds...")

        if self.use_data_driven:
            self._analyze_data_distribution(features)

        labels = np.zeros(len(features), dtype=int)
        aggressive_events = np.zeros(len(features), dtype=bool)
        inattentive_events = np.zeros(len(features), dtype=bool)

        if "gaze_off_road_ratio" in features.columns and "gaze_off_road_high" in self.thresholds:
            inattentive_events |= features["gaze_off_road_ratio"] > self.thresholds["gaze_off_road_high"]

        if "eyelid_low_ratio" in features.columns and "eyelid_low_high" in self.thresholds:
            inattentive_events |= features["eyelid_low_ratio"] > self.thresholds["eyelid_low_high"]

        if "jerk_long_max" in features.columns and "jerk_high" in self.thresholds:
            aggressive_events |= features["jerk_long_max"].abs() > self.thresholds["jerk_high"]

        aggression_score = np.zeros(len(features))
        if "accel_long_std" in features.columns and "accel_std_high" in self.thresholds:
            aggression_score += (features["accel_long_std"] > self.thresholds["accel_std_high"]).astype(float) * 0.3
        if "steering_rate" in features.columns and "steering_rate_high" in self.thresholds:
            aggression_score += (features["steering_rate"] > self.thresholds["steering_rate_high"]).astype(float) * 0.3
        if "throttle_changes" in features.columns and "throttle_changes_high" in self.thresholds:
            aggression_score += (features["throttle_changes"] > self.thresholds["throttle_changes_high"]).astype(float) * 0.2
        if "hr_mean" in features.columns and "hr_elevated" in self.thresholds:
            aggression_score += (features["hr_mean"] > self.thresholds["hr_elevated"]).astype(float) * 0.2

        inattention_score = np.zeros(len(features))
        if "lateral_pos_std" in features.columns and "lateral_std_high" in self.thresholds:
            inattention_score += (features["lateral_pos_std"] > self.thresholds["lateral_std_high"]).astype(float) * 0.4
        if "ndrt_error_rate" in features.columns and "ndrt_error_high" in self.thresholds:
            inattention_score += (features["ndrt_error_rate"] > self.thresholds["ndrt_error_high"]).astype(float) * 0.3
        if "control_smoothness" in features.columns:
            inattention_score += ((1 - features["control_smoothness"]) > 0.5).astype(float) * 0.3

        aggressive_events |= aggression_score >= 0.5
        inattentive_events |= inattention_score >= 0.5

        persistence_samples = int(event_persistence_seconds * sample_rate_hz)
        if persistence_samples > 1:
            aggressive_events = pd.Series(aggressive_events).rolling(
                window=persistence_samples,
                min_periods=1,
                center=True,
            ).max().astype(bool).values
            inattentive_events = pd.Series(inattentive_events).rolling(
                window=persistence_samples,
                min_periods=1,
                center=True,
            ).max().astype(bool).values

        labels[aggressive_events] = 2
        labels[inattentive_events & ~aggressive_events] = 1

        features["aggression_score"] = aggression_score
        features["inattention_score"] = inattention_score

        self._print_distribution(labels)
        return labels

    def _print_distribution(self, labels):
        label_counts = pd.Series(labels).value_counts()
        print("\nLabel distribution:")
        print(f"  Attentive (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(labels) * 100:.1f}%)")
        print(f"  Inattentive (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(labels) * 100:.1f}%)")
        print(f"  Aggressive (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0) / len(labels) * 100:.1f}%)")
