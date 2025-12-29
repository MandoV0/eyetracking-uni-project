import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from DataProcessor import load_csv, execute

class DriverBehaviorClassifier:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=1, random_state=42, use_phys=True):
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=random_state)
        self.feature_columns = []
        self.use_phys = use_phys
    
    def engineer_features(self, df, window_size=50, use_phys=True):
        """
        Engineer features from raw sensor data using rolling windows
        window_size: number of rows for rolling calculations. (Calculate the mean,min,max.... inside these windows as behaviour isnt instantanious)
        """
        print("Engineering features...")
        features = pd.DataFrame()
        
        features['time'] = df['time']
        
        # ---- AGGRESSIVE DRIVING ----
        # Acceleration metrics
        features['accel_long_mean'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).mean()
        features['accel_long_std'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).std()
        features['accel_long_max'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).max()
        features['accel_lat_std'] = df['oveBodyAccelerationLateralY'].rolling(window_size, min_periods=1).std()
        
        # Jerk metrics (how fast f acceleration changes)
        features['jerk_long_mean'] = df['oveBodyJerkLongitudinalX'].rolling(window_size, min_periods=1).mean().abs()
        features['jerk_long_max'] = df['oveBodyJerkLongitudinalX'].rolling(window_size, min_periods=1).max().abs()
        features['jerk_lat_std'] = df['oveBodyJerkLateralY'].rolling(window_size, min_periods=1).std()
        
        # Speed and throttle
        features['speed'] = np.sqrt(df['oveBodyVelocityX'] ** 2 + df['oveBodyVelocityY']**2)
        features['speed_std'] = features['speed'].rolling(window_size, min_periods=1).std()
        features['throttle_mean'] = df['throttle'].rolling(window_size, min_periods=1).mean()
        features['throttle_changes'] = df['throttle'].diff().abs().rolling(window_size, min_periods=1).sum()
        
        # Steering aggressiveness
        features['steering_angle_std'] = df['steeringWheelAngle'].rolling(window_size, min_periods=1).std()
        features['steering_rate'] = df['steeringWheelAngle'].diff().abs().rolling(window_size, min_periods=1).mean()
        features['yaw_velocity_std'] = df['oveYawVelocity'].rolling(window_size, min_periods=1).std()
        
        # Headway (distance to car in front)
        if 'aheadTHW' in df.columns:
            features['thw_mean'] = df['aheadTHW'].replace(-1, np.nan).rolling(window_size, min_periods=1).mean()
            features['short_headway_ratio'] = (df['aheadTHW'].rolling(window_size, min_periods=1).apply(
                lambda x: (((x > 0) & (x < 2)).sum() / len(x)) if len(x) > 0 else 0
            ))
        
        # Brake usage
        if 'brakePedalActive' in df.columns:
            features['brake_frequency'] = df['brakePedalActive'].rolling(window_size, min_periods=1).sum()
        
        # ----- INATTENTIVE DRIVING -----
        # Lane position variability
        features['lateral_pos_std'] = df['ovePositionLateralR'].rolling(window_size, min_periods=1).std()
        features['lateral_pos_mean'] = df['ovePositionLateralR'].rolling(window_size, min_periods=1).mean().abs()
        
        # NDRT (Non driving related tasks) performance
        if 'arrowsWrongCount' in df.columns:
            features['ndrt_error_rate'] = (df['arrowsWrongCount'] + df['arrowsTimeoutCount']).rolling(window_size, min_periods=1).mean()
            features['ndrt_total_attempts'] = (df['arrowsCorrectCount'] + df['arrowsWrongCount'] + df['arrowsTimeoutCount']).rolling(window_size, min_periods=1).sum()
        
        # Eye gaze features
        # TODO: Fine tune the angles
        if 'openxrGazeHeading' in df.columns:
            # Off-road
            features['gaze_heading_abs'] = df['openxrGazeHeading'].abs()
            features['gaze_pitch_abs'] = df['openxrGazePitch'].abs()
            features['gaze_off_road'] = ((df['openxrGazeHeading'].abs() > 30) | 
                                          (df['openxrGazePitch'].abs() > 20)).astype(int)
            features['gaze_off_road_ratio'] = features['gaze_off_road'].rolling(window_size, min_periods=1).mean()
        
        # Eyelid opening to detect if the driver is drowsiness (sleepy), could maybe pair this with the hearbeat later? TODO
        if 'varjoEyelidOpening' in df.columns:
            features['eyelid_mean'] = df['varjoEyelidOpening'].rolling(window_size, min_periods=1).mean()
            features['eyelid_low_ratio'] = (df['varjoEyelidOpening'] < 0.3).rolling(window_size, min_periods=1).mean()
        
        # If the driver is Attentive he should be driving pretty smoothly.
        features['control_smoothness'] = 1 / (1 + features['steering_rate'] + features['throttle_changes'] / 10)

        if use_phys:
            if 'heartRate' in df.columns:
                # 1. Stress Level (Mean HR)
                features['hr_mean'] = df['heartRate'].rolling(window_size, min_periods=1).mean()
                
                # 2. Stress Response (HR Change)
                # How much the heart rate is jumping up/down
                features['hr_change'] = df['heartRate'].diff().abs().rolling(window_size, min_periods=1).mean()

            if 'rrInterval' in df.columns:
                # 3. HRV (Standard Deviation of RR intervals) - SDNN
                # LOW value = High Stress/Aggression
                # HIGH value = Drowsy/Relaxed
                features['hrv_sdnn'] = df['rrInterval'].rolling(window_size, min_periods=1).std()
                
                # 4. RMSSD (Root Mean Square of Successive Differences)
                # Better for short windows than SDNN
                diff_rr = df['rrInterval'].diff()
                features['hrv_rmssd'] = (diff_rr ** 2).rolling(window_size, min_periods=1).mean() ** 0.5
        
        # Fill NaN values with next or previous valid value.
        # Example: [NaN, 3, 5] -> [3, 3, 5]
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        print(f"All Features: {features.columns}")

        return features
    
    def generate_labels(self, features):
        """
        Generate pseudo-labels based on heuristic rules
        Returns: labels (Attentive=0, Inattentive=1, Aggressive=2)
        """
        print("Generating rule-based labels...")
        
        # Normalize features for scoring (0 - 1 scale) so we can compare them to eachother
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-10) # 1e-10 to avoid possible division by zero, so we are just adding a tiny number.
        
        # TODO: Fine tune the weights
        # AGGRESSION SCORE
        aggression_score = (
            normalize(features['accel_long_std']) * 0.15 +
            normalize(features['jerk_long_max']) * 0.20 +
            normalize(features['jerk_lat_std']) * 0.15 +
            normalize(features['speed_std']) * 0.10 +
            normalize(features['steering_angle_std']) * 0.15 +
            normalize(features['steering_rate']) * 0.15 +
            normalize(features['throttle_changes']) * 0.10
        )

        if self.use_phys:
            if 'hr_mean' in features.columns:
                # High Heart Rate + High Variability (erratic) = Aggressive/Stress
                aggression_score += normalize(features['hr_mean']) * 0.15
                aggression_score += normalize(features['hr_change']) * 0.10

        if 'short_headway_ratio' in features.columns:
            aggression_score += normalize(features['short_headway_ratio']) * 0.10
        
        # Add headway if available
        if 'short_headway_ratio' in features.columns:
            print("Added Headway")
            aggression_score += normalize(features['short_headway_ratio']) * 0.10
        
        # INATTENTION SCORE
        inattention_score = (
            normalize(features['lateral_pos_std']) * 0.20
        )
        
        if 'ndrt_error_rate' in features.columns:
            inattention_score += normalize(features['ndrt_error_rate']) * 0.25
        
        if 'gaze_off_road_ratio' in features.columns:
            inattention_score += normalize(features['gaze_off_road_ratio']) * 0.30
        
        if 'eyelid_low_ratio' in features.columns:
            inattention_score += normalize(features['eyelid_low_ratio']) * 0.15
        
        if self.use_phys:
            if 'hrv_rmssd' in features.columns:
                # High HRV (Relaxed/Drowsy) indicates Inattention
                inattention_score += normalize(features['hrv_rmssd']) * 0.20
        
        # Low control smoothness indicates inattention.
        inattention_score += normalize(1 - features['control_smoothness']) * 0.10
        
        # CLASSIFICATION LOGIC
        # Use percentile based thresholds.
        # TODO: Fine tune the treshold
        agg_threshold_high = np.percentile(aggression_score, 70)
        inatt_threshold_high = np.percentile(inattention_score, 70)
        
        labels = np.zeros(len(features), dtype=int)  # Default: Attentive
        
        # If both scores are high, aggressive is choosen as aggressive driving is more dangerous
        labels[(aggression_score > agg_threshold_high) & (inattention_score <= inatt_threshold_high)] = 2   # Aggressive
        labels[(inattention_score > inatt_threshold_high) & (aggression_score <= agg_threshold_high)] = 1   # Inattentive
        labels[(aggression_score > agg_threshold_high) & (inattention_score > inatt_threshold_high)] = 2    # Aggressive
        
        # Store scores for analysis
        features['aggression_score'] = aggression_score
        features['inattention_score'] = inattention_score
        
        label_counts = pd.Series(labels).value_counts()
        print(f"\nLabel distribution:")
        print(f"  Attentive (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(labels) * 100:.1f}%)")
        print(f"  Inattentive (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(labels) * 100:.1f}%)")
        print(f"  Aggressive (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0) / len(labels) * 100:.1f}%)")
        
        # TODO: Use a Hidden Markov Model to create the labels using the weights

        return labels
    
    """
    Removes columns not used for training
    """
    def prepare_training_data(self, features, labels):
        exclude_cols = ['time', 'aggression_score', 'inattention_score']
        X = features.drop(columns=[col for col in exclude_cols if col in features.columns])
        self.feature_columns = X.columns.tolist()
        return X, labels
    
    def train(self, X, y):
        print("\n\n Training Random Forest classifier...")
        
        # Split data into train/test samples.
        # TODO: Use all available files for training/testing 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify preservers propotions of the labels for the split.
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy   :      {train_score:.3f}")
        print(f"Testing accuracy    :      {test_score:.3f}")

        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        y_pred = self.model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Attentive', 'Inattentive', 'Aggressive']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return X_train, X_test, y_train, y_test, y_pred
    
    """
    Predict models performance on testing/new data.
    """
    def predict(self, features):
        X = features[self.feature_columns]
        return self.model.predict(X)
    
    def run_pipeline(self, df: pd.DataFrame, window_size=50):
        features = self.engineer_features(df, window_size)
        labels = self.generate_labels(features)
        X, y = self.prepare_training_data(features, labels)
        
        results = self.train(X, y)
        print ("\n\n----- Done -----\n\n")
        return features, labels, results

class DataCleaner:
    """
    The purpose of this class is:
    - Entfernen fehlerhafter Messwerte (Outlier Detection, Sensorfehler)
    - Glättung und Noise-Reduction mittels Filterverfahren
    """

if __name__ == "__main__":
    

    # TODO: Use all data for training/testing instead of a single csv file.
        
    print("\n\n\nModel training complete")
    # TODO: Visualizations of data and model performance. (Also for the confusion matrix)
    # TODO: Add Physiology data to the algorithm. High heart beat could indicate risky/aggresive driving
    # TODO: Test with different estimators and depth.

    """
    Currently it seems like the model is not learning any patters it seems like its learning the labeling rules making the accuracy "too" high.
    So a HMM will probably perform better.
    """

    # default_merge   = execute(interpolation_mode=0)
    linear_sync     = execute(interpolation_mode=1, max_pairs=3)
    # spline_sync     = execute(interpolation_mode=1)

    """
    print("Merge with Phys")
    classifier = DriverBehaviorClassifier(max_depth=5, min_samples_leaf=5)
    features, labels, results = classifier.run_pipeline(default_merge, window_size=50)

    print("Merge without Phys")
    classifier = DriverBehaviorClassifier(max_depth=5, min_samples_leaf=5)
    features, labels, results = classifier.run_pipeline(default_merge, window_size=50, use_pyhs=False)
    """

    print("Linear with Phys")
    classifier = DriverBehaviorClassifier(max_depth=5, min_samples_leaf=5)
    features, labels, results = classifier.run_pipeline(linear_sync, window_size=50)