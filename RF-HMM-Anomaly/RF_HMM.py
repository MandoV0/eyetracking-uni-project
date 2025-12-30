"""
Simplified Anomaly-Based Driver Behavior Classifier
Classifies driving behavior into: Attentive, Inattentive, Aggressive
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, log_loss,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

class AnomalyBasedClassifier:
    """
    Anomaly-Based Driver Behavior Classifier
    
    Idea:
    1. Anomaly Detection (Isolation Forest) - 85% Attentive, 15% anomalies
    2. Cluster Anomalies (K-Means) - Separate into Aggressive vs Inattentive
    3. HMM Smoothing (Temporal consistency)
    4. Random Forest Training
    """
    
    def __init__(self, contamination=0.15, n_hmm_states=5, random_state=42):
        self.contamination = contamination
        self.n_hmm_states = n_hmm_states
        self.random_state = random_state
        
        self.scaler = StandardScaler()          # Normalizes/Scales the features 
        self.iso_forest = IsolationForest(      # Detects Anomalies for example a driver suddenly breaking very hard with extreme steering
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced'         # Gives more importance to rarer classes so we dont just predict the majority class.
        )
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
    
    def prepare_features(self, features):
        feature_cols = [
            # Core dynamics
            'accel_long_mean', 'accel_long_std', 'accel_lat_std',
            'jerk_long_mean', 'jerk_long_max', 'jerk_lat_std',
            'speed', 'speed_std', 'throttle_mean', 'throttle_changes',
            'steering_angle_std', 'steering_rate',
            'lateral_pos_std', 'control_smoothness'
        ]
        
        # Optional features
        optional = ['gaze_off_road_ratio', 'ndrt_error_rate', 
                   'short_headway_ratio', 'hr_mean', 'hrv_rmssd']
        
        for feat in optional:
            if feat in features.columns:
                feature_cols.append(feat)
        
        X = features[feature_cols].fillna(0)
        self.feature_columns = feature_cols
        return X
    
    """
    Identify Anomalies using Isolation Forest.
    """
    def __anomaly_detection(self, features):
        print("\n--- Start: Anomaly Detection ---")
        
        X = self.prepare_features(features)
        X_scaled = self.scaler.fit_transform(X)
        
        anomaly_labels = self.iso_forest.fit_predict(X_scaled)
        
        n_anomalies = (anomaly_labels == -1).sum()
        n_normal = (anomaly_labels == 1).sum()
        print(f"Detected {n_anomalies:,} anomalies ({n_anomalies/len(anomaly_labels)*100:.1f}%)")
        print(f"Normal behavior: {n_normal:,} ({n_normal/len(anomaly_labels)*100:.1f}%)")
        
        return anomaly_labels, X_scaled
    
    """
    Cluster anomalies into Aggressive vs Inattentive
    """
    def _cluster_anomalies(self, X_scaled, anomaly_labels, features):
        print("\n--- Start: Clustering Anomalies ---")
        
        anomaly_mask = anomaly_labels == -1
        X_anomalies = X_scaled[anomaly_mask]
        
        if len(X_anomalies) < 10:
            print("!!! Too few anomalies, labeling all as Aggressive !!!")
            initial_labels = np.zeros(len(anomaly_labels), dtype=int)
            initial_labels[anomaly_mask] = 2
            return initial_labels
        
        # Cluster into 2 groups
        kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        anomaly_clusters = kmeans.fit_predict(X_anomalies)
        
        # Score each cluster
        feature_df = features[self.feature_columns].iloc[anomaly_mask]
        cluster_profiles = []
        
        for cluster in [0, 1]:
            mask = anomaly_clusters == cluster
            means = feature_df[mask].mean()
            
            # Aggression: harsh dynamics
            agg_score = (
                means.get('jerk_long_max', 0) * 4.0 +
                means.get('accel_long_std', 0) * 3.0 +
                means.get('steering_rate', 0) * 3.0 +
                means.get('throttle_changes', 0) * 2.0
            )
            
            # Inattention: poor control & distraction
            inatt_score = (
                means.get('lateral_pos_std', 0) * 4.0 +
                means.get('gaze_off_road_ratio', 0) * 5.0 +
                means.get('ndrt_error_rate', 0) * 4.0 +
                (1 - means.get('control_smoothness', 1.0)) * 3.0
            )
            
            if 'hrv_rmssd' in means.index:
                inatt_score += means['hrv_rmssd'] * 0.5
            
            cluster_profiles.append({
                'cluster': cluster,
                'count': mask.sum(),
                'agg_score': agg_score,
                'inatt_score': inatt_score
            })
            
            print(f"Cluster {cluster}: {mask.sum():,} samples - "
                  f"Agg={agg_score:.2f}, Inatt={inatt_score:.2f}")
        
        # Assign labels based on score ratios
        ratio0 = cluster_profiles[0]['agg_score'] / (cluster_profiles[0]['inatt_score'] + 1e-10) # 1e-10 is just a tiny number to avoid possible errors by dividng by zero
        ratio1 = cluster_profiles[1]['agg_score'] / (cluster_profiles[1]['inatt_score'] + 1e-10)
        
        print(f"Cluster 0 ratio (Agg/Inatt): {ratio0:.2f}")
        print(f"Cluster 1 ratio (Agg/Inatt): {ratio1:.2f}")
        
        aggressive_cluster = 0 if ratio0 > ratio1 else 1
        inattentive_cluster = 1 - aggressive_cluster
        
        print(f"Cluster {aggressive_cluster} -> Aggressive")
        print(f"Cluster {inattentive_cluster} -> Inattentive")
        
        # Apply labels
        initial_labels = np.zeros(len(anomaly_labels), dtype=int)
        anomaly_indices = np.where(anomaly_mask)[0]
        
        for i, idx in enumerate(anomaly_indices):
            initial_labels[idx] = 2 if anomaly_clusters[i] == aggressive_cluster else 1
        
        self._print_distribution(initial_labels, "Initial")
        return initial_labels
    
    """
    Tempral Smoothing with HMM.
    Driver behavior does not change in an instant with every row/timestep.
    So the HMM smoothes it out and stops the Random Forest from learning Noise/Random Feature Spikes.
    """
    def _hmm_smoothing(self, features, initial_labels):
        """Stage 3: Temporal smoothing with HMM"""
        print("\n--- Start: HMM Temporal Smoothing ---")
        
        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        hmm_model = hmm.GaussianHMM(
            n_components=self.n_hmm_states,
            covariance_type="diag",
            n_iter=50,
            random_state=self.random_state
        )
        
        hmm_model.fit(X_scaled)
        hmm_states = hmm_model.predict(X_scaled)
        
        # Map HMM states to behavior labels with minority protection
        state_mapping = {}
        
        for state in range(self.n_hmm_states):
            mask = hmm_states == state
            votes = initial_labels[mask]
            
            if len(votes) == 0:
                state_mapping[state] = 0
                continue
            
            vote_counts = np.bincount(votes, minlength=3)
            percentages = vote_counts / len(votes)
            
            # Protect minority classes
            # Basically stops rare classes from being smoothed away by the HMM
            if percentages[1] > 0.15:  # Inattentive
                state_mapping[state] = 1
            elif percentages[2] > 0.30:  # Aggressive
                state_mapping[state] = 2
            else:
                state_mapping[state] = vote_counts.argmax()
        
        smoothed_labels = np.array([state_mapping[s] for s in hmm_states])
        
        # Fallback if classes eliminated
        if len(np.unique(smoothed_labels)) < 3:
            print("!!! Fallback: Class eliminated, using conservative smoothing...")
            smoothed_labels = initial_labels.copy()
            
            for state in range(self.n_hmm_states):
                mask = hmm_states == state
                votes = initial_labels[mask]
                
                if len(votes) > 0:
                    vote_counts = np.bincount(votes, minlength=3)
                    if vote_counts.max() / len(votes) > 0.80:
                        smoothed_labels[mask] = vote_counts.argmax()
        
        if len(np.unique(smoothed_labels)) < 3:
            print("!!! Using initial labels")
            smoothed_labels = initial_labels
        
        self._print_distribution(smoothed_labels, "Smoothed")
        
        changes = (initial_labels != smoothed_labels).sum()
        print(f"Labels changed: {changes:,} ({changes/len(initial_labels)*100:.1f}%)")
        
        return smoothed_labels
    
    """
    Trains the Random Forest Classifier
    """
    def _train_rf(self, features, labels):
        print("\n--- Start: Training Random Forest ---")
        
        X = self.prepare_features(features)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print("!!! Need at least 2 classes!")
            return None
        
        print(f"Training with {len(unique_labels)} classes")
        
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        self.rf_model.fit(self.X_train, y_train)
        
        y_train_pred = self.rf_model.predict(self.X_train)
        y_test_pred = self.rf_model.predict(self.X_test)
        
        self._print_metrics(y_train, y_train_pred, y_test, y_test_pred, unique_labels)
        self._print_feature_importance()
        
        return self.X_train, self.X_test, y_train, y_test, y_test_pred
    
    def _print_distribution(self, labels, stage_name):
        print(f"\n{stage_name} distribution:")
        for i, name in enumerate(['Attentive', 'Inattentive', 'Aggressive']):
            count = (labels == i).sum()
            print(f"  {name}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    def _print_metrics(self, y_train, y_train_pred, y_test, y_test_pred, unique_labels):
        print("\n\n ------ MODEL PERFORMANCE METRICS ------ \n\n")
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\n--- MODEL ACCURACY ---")
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Testing Accuracy:    {test_acc:.4f}")
        print(f"Train-Test Gap:      {(train_acc - test_acc):.4f}")
        
        print(f"\n--- WEIGHTED METRICS ---")
        print(f"{'Metric':<20} {'Training':<12} {'Testing':<12}")
        print(f"{'-'*44}")
        
        for metric_name, metric_func in [
            ('Precision', precision_score),
            ('Recall', recall_score),
            ('F1-Score', f1_score)
        ]:
            train_val = metric_func(y_train, y_train_pred, average='weighted', zero_division=0)
            test_val = metric_func(y_test, y_test_pred, average='weighted', zero_division=0)
            print(f"{metric_name:<20} {train_val:<12.4f} {test_val:<12.4f}")
        
        # Per-class metrics
        print(f"\n--- PER-CLASS METRICS (Test Set) ---")
        label_names = ['Attentive', 'Inattentive', 'Aggressive']
        test_labels_present = np.unique(np.concatenate([y_test, y_test_pred]))
        
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print(f"{'-'*61}")
        
        for label in test_labels_present:
            if label < len(label_names):
                tp = np.sum((y_test == label) & (y_test_pred == label))
                fp = np.sum((y_test != label) & (y_test_pred == label))
                fn = np.sum((y_test == label) & (y_test_pred != label))
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                support = (y_test == label).sum()
                
                print(f"{label_names[label]:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {support:<10}")
        
        # Error metrics
        print(f"\n--- ERROR METRICS ---")
        print(f"{'Metric':<30} {'Training':<12} {'Testing':<12}")
        print(f"{'-'*54}")
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        print(f"{'MSE':<30} {train_mse:<12.4f} {test_mse:<12.4f}")
        print(f"{'RMSE':<30} {np.sqrt(train_mse):<12.4f} {np.sqrt(test_mse):<12.4f}")
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        print(f"{'MAE':<30} {train_mae:<12.4f} {test_mae:<12.4f}")
        
        # Log loss
        print(f"\n--- COST FUNCTION ---")
        y_train_proba = self.rf_model.predict_proba(self.X_train)
        y_test_proba = self.rf_model.predict_proba(self.X_test)
            
        train_proba_full = np.zeros((len(y_train), 3))
        test_proba_full = np.zeros((len(y_test), 3))
            
        for i, label in enumerate(self.rf_model.classes_):
            train_proba_full[:, label] = y_train_proba[:, i]
            test_proba_full[:, label] = y_test_proba[:, i]
            
        print(f"Training Log Loss:   {log_loss(y_train, train_proba_full, labels=[0, 1, 2]):.4f}")
        print(f"Testing Log Loss:    {log_loss(y_test, test_proba_full, labels=[0, 1, 2]):.4f}")
        
        # Statistical metrics
        print(f"\n--- AGREEMENT METRICS ---")
        print(f"Cohen's Kappa (Train):  {cohen_kappa_score(y_train, y_train_pred):.4f}")
        print(f"Cohen's Kappa (Test):   {cohen_kappa_score(y_test, y_test_pred):.4f}")
        
        try:
            print(f"Matthews Corr (Train):  {matthews_corrcoef(y_train, y_train_pred):.4f}")
            print(f"Matthews Corr (Test):   {matthews_corrcoef(y_test, y_test_pred):.4f}")
        except:
            pass
        
        # Confusion matrix
        print(f"\n--- CONFUSION MATRIX (Test Set) ---")
        cm = confusion_matrix(y_test, y_test_pred, labels=test_labels_present)
        
        print(f"\n{'':>15}", end="")
        for label in test_labels_present:
            print(f"{label_names[label]:>12}", end="")
        print()
        
        for i, label in enumerate(test_labels_present):
            print(f"{label_names[label]:>15}", end="")
            for val in cm[i]:
                print(f"{val:>12}", end="")
            print()
        
        # Cross-validation
        print(f"\n--- CROSS-VALIDATION (5-Fold) ---")
        try:
            X_full = np.vstack([self.X_train, self.X_test])
            y_full = np.concatenate([y_train, y_test])
            
            cv_scores = cross_val_score(self.rf_model, X_full, y_full, cv=5)
            print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"Could not perform CV: {e}")
        
        print(f"\n{'='*70}\n")
    
    def _print_feature_importance(self):
        """Print feature importance"""
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n--- FEATURE IMPORTANCE ---")
        print("Top 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
    
    def fit(self, features):
        anomaly_labels, X_scaled = self.__anomaly_detection(features)
        initial_labels = self._cluster_anomalies(X_scaled, anomaly_labels, features)
        smoothed_labels = self._hmm_smoothing(features, initial_labels)
        results = self._train_rf(features, smoothed_labels)
        
        print("\n\n----- TRAINING COMPLETE -----\n\n")
        
        return initial_labels, smoothed_labels, results
    
    def predict(self, features):
        X = self.prepare_features(features)
        return self.rf_model.predict(X)
    
    def predict_proba(self, features):
        X = self.prepare_features(features)
        return self.rf_model.predict_proba(X)

if __name__ == "__main__":
    from DataProcessor import execute
    from LabelGenerator import DriverBehaviorClassifier
    
    print("Loading data...")
    df = execute(interpolation_mode=1, max_pairs=15, combine_all=True)
    
    print("Engineering features...")
    feature_engineer = DriverBehaviorClassifier()
    features = feature_engineer.engineer_features(df, window_size=200)
    
    model = AnomalyBasedClassifier(
        contamination=0.2,    # Expect 20% anomalies
        n_hmm_states=5,        # HMM complexity
        random_state=42
    )
    
    initial_labels, final_labels, results = model.fit(features)
    
    print("\n Model trained successfully!")

    # TODO:
    # Current Problems:
    # - Still seems like the model is just memorizing the labeling logic so the train test accuracy is basially the same.