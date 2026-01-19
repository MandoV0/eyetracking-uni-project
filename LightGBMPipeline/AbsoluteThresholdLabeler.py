"""
Labeler for the unlabeled driver data.
We have no ground-truth labels to train the ML models so we need to generate them ourselves.
We want to avoud circular logic where the model just learns the rules.

So to avoid this we use Global, absolute, physically interpretable thresholds
Basically crossing the threshold has the same meaning in everywhere.

For Example:
   high jerk is 5 m/s^3
   This will now mean the same thing at T = 10s, t = 2h
   Or if Driver A or B does it.
   Previously we used percentiles which means that the threshold is relative to the sample which was bad.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TODO: Save Data to avoid recalculating everything for every model run.
class AbsoluteThresholdLabeler:
    
    def __init__(self, use_data_driven_thresholds=True):
        self.use_data_driven = use_data_driven_thresholds
        self.thresholds = {}
    
    """
    Learns the global reference thresholds based on the data distribution.
    """
    def _analyze_data_distribution(self, features):
        print("Analyzing data distribution to set thresholds...")   
        
        thresholds = {}
        
        # Aggressive indicators. Look at high percentiles
        if 'jerk_long_max' in features.columns:
            thresholds['jerk_high'] = np.percentile(features['jerk_long_max'].abs(), 90)
        if 'accel_long_std' in features.columns:
            thresholds['accel_std_high'] = np.percentile(features['accel_long_std'], 85)
        if 'steering_rate' in features.columns:
            thresholds['steering_rate_high'] = np.percentile(features['steering_rate'], 85)
        if 'throttle_changes' in features.columns:
            thresholds['throttle_changes_high'] = np.percentile(features['throttle_changes'], 85)
        
        # Inattention indicators
        if 'gaze_off_road_ratio' in features.columns:
            thresholds['gaze_off_road_high'] = np.percentile(features['gaze_off_road_ratio'], 80)
        if 'lateral_pos_std' in features.columns:
            thresholds['lateral_std_high'] = np.percentile(features['lateral_pos_std'], 85)
        if 'ndrt_error_rate' in features.columns:
            thresholds['ndrt_error_high'] = np.percentile(features['ndrt_error_rate'], 80)
        if 'eyelid_low_ratio' in features.columns:
            thresholds['eyelid_low_high'] = np.percentile(features['eyelid_low_ratio'], 80)
        
        # Heart rate indicators
        if 'hr_mean' in features.columns:
            # Use z-score: HR > mean + 1.5*std is elevated
            hr_mean = features['hr_mean'].mean()
            hr_std = features['hr_mean'].std()
            thresholds['hr_elevated'] = hr_mean + 1.5 * hr_std
        
        self.thresholds = thresholds
        print(f"Set {len(thresholds)} thresholds based on data distribution")
        return thresholds

    """
    Generate labels using absolute thresholds with event persistence.
        
    Args:
        features: DataFrame with engineered features
        event_persistence_seconds: How long an event persists (smoothing)
        sample_rate_hz: Sampling rate to convert seconds to samples
    """
    def generate_labels(self, features, event_persistence_seconds=2.0, sample_rate_hz=10):
        print("Generating labels using absolute thresholds...")
        
        if self.use_data_driven:
            self._analyze_data_distribution(features)
        
        labels = np.zeros(len(features), dtype=int)  # Default 0 means Attentive
        
        # Initialize event/label flags
        aggressive_events = np.zeros(len(features), dtype=bool)
        inattentive_events = np.zeros(len(features), dtype=bool)
        
        # CRITICAL EVENTS
        # These are events that are so critical that they override everything else.
        # Critical bevause inattentive driving is more dangerous than aggressive driving.
        # Example: Gaze of road looking at your phone might cause you to crash into a group of people.
        # Critical Inattention

        # |= is a bitwise OR. 
        if 'gaze_off_road_ratio' in features.columns and 'gaze_off_road_high' in self.thresholds:
            critical_inatt = features['gaze_off_road_ratio'] > self.thresholds['gaze_off_road_high']
            inattentive_events |= critical_inatt
        
        if 'eyelid_low_ratio' in features.columns and 'eyelid_low_high' in self.thresholds:
            drowsy = features['eyelid_low_ratio'] > self.thresholds['eyelid_low_high']
            inattentive_events |= drowsy
        
        # Critical Aggression
        # 
        if 'jerk_long_max' in features.columns and 'jerk_high' in self.thresholds:
            harsh_maneuver = features['jerk_long_max'].abs() > self.thresholds['jerk_high']
            aggressive_events |= harsh_maneuver
        
        # ACCUMULATIVE SCORING (Lower Priority)
        # Basically instead of doing if X is aggresive, we build a Temporal score over time.
        # So it becomes: if enough aggresive events happen, then the driver is aggressive.
        # Sum of weights is 1.0
        # Build aggression score
        aggression_score = np.zeros(len(features))
        if 'accel_long_std' in features.columns and 'accel_std_high' in self.thresholds:
            aggression_score += (features['accel_long_std'] > self.thresholds['accel_std_high']).astype(float) * 0.3
        if 'steering_rate' in features.columns and 'steering_rate_high' in self.thresholds:
            aggression_score += (features['steering_rate'] > self.thresholds['steering_rate_high']).astype(float) * 0.3
        if 'throttle_changes' in features.columns and 'throttle_changes_high' in self.thresholds:
            aggression_score += (features['throttle_changes'] > self.thresholds['throttle_changes_high']).astype(float) * 0.2
        if 'hr_mean' in features.columns and 'hr_elevated' in self.thresholds:
            aggression_score += (features['hr_mean'] > self.thresholds['hr_elevated']).astype(float) * 0.2
        
        # Build inattention score
        inattention_score = np.zeros(len(features))
        if 'lateral_pos_std' in features.columns and 'lateral_std_high' in self.thresholds:
            inattention_score += (features['lateral_pos_std'] > self.thresholds['lateral_std_high']).astype(float) * 0.4
        if 'ndrt_error_rate' in features.columns and 'ndrt_error_high' in self.thresholds:
            inattention_score += (features['ndrt_error_rate'] > self.thresholds['ndrt_error_high']).astype(float) * 0.3
        if 'control_smoothness' in features.columns:
            inattention_score += ((1 - features['control_smoothness']) > 0.5).astype(float) * 0.3
        
        # Apply accumulative thresholds
        agg_threshold = 0.5  # Need at least 50% of indicators
        inatt_threshold = 0.5
        
        aggressive_events |= aggression_score >= agg_threshold
        inattentive_events |= inattention_score >= inatt_threshold
        
        # Apply event persistence for temporal smoothing
        # Smoothing out shortterm spikes and noise.
        persistence_samples = int(event_persistence_seconds * sample_rate_hz) # Example: 2.0 seconds at 10 Hz is 20 samples
        if persistence_samples > 1:
            # Use rolling max to extend events
            aggressive_events = pd.Series(aggressive_events).rolling(
                window=persistence_samples, min_periods=1, center=True
            ).max().astype(bool).values
            
            inattentive_events = pd.Series(inattentive_events).rolling(
                window=persistence_samples, min_periods=1, center=True
            ).max().astype(bool).values
        
        # Assign labels Aggressive takes priority for safety of others.
        labels[aggressive_events] = 2  # Aggressive
        labels[inattentive_events & ~aggressive_events] = 1  # Inattentive (if not aggressive)
        # inattentive_events & ~aggressive_events element wise boolean logic as both are boolean arrays.
        # So assign inattentive only if the driver is inattentive but not aggressive as aggresive has higher
        # priority.

        # Store scores for analysis/debugging
        features['aggression_score'] = aggression_score
        features['inattention_score'] = inattention_score
        
        self._print_distribution(labels)
        return labels
    
    def _print_distribution(self, labels):
        label_counts = pd.Series(labels).value_counts()
        print(f"\nLabel distribution:")
        print(f"  Attentive (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(labels) * 100:.1f}%)")
        print(f"  Inattentive (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(labels) * 100:.1f}%)")
        print(f"  Aggressive (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0) / len(labels) * 100:.1f}%)")