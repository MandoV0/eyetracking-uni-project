"""
Driver State Classifier with LightGBM and Model Saving Functionality

This module provides a complete pipeline for training and saving a driver state classifier
using absolute threshold-based labeling and LightGBM (best performing model).
"""

"""
Trains a LightGBM Model with Cross-Validation.
This prevents data leakage across sessions.
Each Fold leaves out all data from one driving session as a test
as some Drivers may have unique patterns, this prevents the model from memorizing specific drivers or sessions.


"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


class DriverStateClassifier:
    def __init__(self, random_state=42, verbose=-1, device='cpu', gpu_platform_id=0, gpu_device_id=0):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random seed for reproducibility
            verbose: Verbosity level (-1 for silent)
            device: 'cpu' or 'gpu' (GPU works on both NVIDIA and AMD via OpenCL)
            gpu_platform_id: OpenCL platform ID (0 = default)
            gpu_device_id: OpenCL device ID (0 = default)
        """
        self.random_state = random_state
        
        # Set up LightGBM parameters
        lgbm_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'class_weight': 'balanced',
            'random_state': random_state,
            'verbose': verbose,
            'device': device
        }
        
        # Add GPU-specific parameters if using GPU
        if device == 'gpu':
            lgbm_params['gpu_platform_id'] = gpu_platform_id
            lgbm_params['gpu_device_id'] = gpu_device_id
        
        self.model = LGBMClassifier(**lgbm_params)
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.labeler = None
    
    """
    Prepare features, excluding metadata columns that are not relevant for trainig
    """
    def prepare_features_for_training(self, features):
        
        exclude_cols = ['time', 'aggression_score', 'inattention_score', 'session_id']
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        
        if self.feature_cols is None:
            self.feature_cols = feature_cols
        
        X = features[self.feature_cols].fillna(0)
        return X
    
    def train_with_cross_validation(self, all_features, all_labels, session_ids):
        """
        Train with Leave-One-Session-Out cross-validation.
        This prevents overfitting to specific driving sessions/drivers.
        """
        print("\n" + "="*70)
        print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
        print("="*70)
        
        X = self.prepare_features_for_training(all_features)
        X_scaled = self.scaler.fit_transform(X)
        
        logo = LeaveOneGroupOut()
        cv_scores = []
        cv_predictions = []
        cv_true_labels = []
        train_accs = []
        test_accs = []
        gaps = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, all_labels, session_ids)):
            print(f"\n--- Fold {fold_idx + 1}: Testing session {session_ids[test_idx[0]]} ---")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = all_labels[train_idx], all_labels[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Predict on both train and test
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            gaps.append(train_acc - test_acc)
            cv_scores.append(test_acc)
            cv_predictions.extend(y_test_pred)
            cv_true_labels.extend(y_test)
            
            gap_status = '!!!!!! LIKELY OVERFITTING !!!!!!' if abs(train_acc - test_acc) > 0.15 else ' OK'
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Train-Test Gap: {train_acc - test_acc:+.4f} ({gap_status})")
            print(f"  Train set size: {len(y_train)}")
            print(f"  Test set size:  {len(y_test)}")
            print(f"  Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        # Overall metrics
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"Mean Train Accuracy: {np.mean(train_accs):.4f}")
        print(f"Mean Test Accuracy:  {np.mean(test_accs):.4f} +- {np.std(test_accs):.4f}")
        print(f"Mean Train-Test Gap: {np.mean(gaps):+.4f} ({'!!!! LIKELY OVERFITTING !!!!' if abs(np.mean(gaps)) > 0.15 else ' OK'})")
        print(f"Min Test Accuracy:  {np.min(test_accs):.4f}")
        print(f"Max Test Accuracy:  {np.max(test_accs):.4f}")
        
        # Overall classification report
        print("\nOverall Classification Report:")
        print(classification_report(cv_true_labels, cv_predictions, target_names=['Attentive', 'Inattentive', 'Aggressive']))
        
        # Overall confusion matrix
        print("\nOverall Confusion Matrix:")
        cm = confusion_matrix(cv_true_labels, cv_predictions)
        print(f"{'':>15} {'Attentive':>12} {'Inattentive':>12} {'Aggressive':>12}")
        for i, name in enumerate(['Attentive', 'Inattentive', 'Aggressive']):
            print(f"{name:>15}", end="")
            for val in cm[i]:
                print(f"{val:>12}", end="")
            print()
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for i, name in enumerate(['Attentive', 'Inattentive', 'Aggressive']):
            mask = np.array(cv_true_labels) == i
            if mask.sum() > 0:
                prec = precision_score(cv_true_labels, cv_predictions, labels=[i], average='micro', zero_division=0)
                rec = recall_score(cv_true_labels, cv_predictions, labels=[i], average='micro', zero_division=0)
                f1 = f1_score(cv_true_labels, cv_predictions, labels=[i], average='micro', zero_division=0)
                print(f"  {name}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        
        print(f"\nCohen's Kappa: {cohen_kappa_score(cv_true_labels, cv_predictions):.4f}")
        
        return {
            'cv_scores': cv_scores,
            'mean_train_accuracy': np.mean(train_accs),
            'mean_test_accuracy': np.mean(test_accs),
            'mean_gap': np.mean(gaps),
            'std_accuracy': np.std(cv_scores),
            'predictions': cv_predictions,
            'true_labels': cv_true_labels
        }
    
    def train_final_model(self, features, labels):
        """
        Train the final model on all available data (no cross-validation).
        
        Args:
            features: DataFrame with engineered features
            labels: Array of labels
            
        Returns:
            Training accuracy
        """
        X = self.prepare_features_for_training(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model on all data
        self.model.fit(X_scaled, labels)
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(labels, train_pred)
        
        return train_acc
    
    def predict(self, features):
        """
        Predict driver states for new data.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Array of predictions (0=Attentive, 1=Inattentive, 2=Aggressive)
        """
        if self.feature_cols is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        X = features[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, features):
        """
        Predict driver state probabilities for new data.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Array of probability predictions [Attentive, Inattentive, Aggressive]
        """
        if self.feature_cols is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        X = features[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet or doesn't support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train on training split and evaluate on held-out test split.
        Prints BOTH train and test accuracy.
        """

        # Prepare features
        X_train_prep = self.prepare_features_for_training(X_train)
        X_test_prep = self.prepare_features_for_training(X_test)

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_prep)
        X_test_scaled = self.scaler.transform(X_test_prep)

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Predict
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # PRINT RESULTS
        print("\nTRAIN / TEST RESULTS")
        print("=" * 60)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test  Accuracy: {test_acc:.4f}")
        print(f"Train-Test Gap: {train_acc - test_acc:+.4f}")

        print("\nTEST SET CLASSIFICATION REPORT:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=['Attentive', 'Inattentive', 'Aggressive']
        ))

        print("TEST SET CONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"{'':>15} {'Attentive':>12} {'Inattentive':>12} {'Aggressive':>12}")
        for i, name in enumerate(['Attentive', 'Inattentive', 'Aggressive']):
            print(f"{name:>15}", end="")
            for val in cm[i]:
                print(f"{val:>12}", end="")
            print()

        print(f"\nTest Cohen's Kappa: {cohen_kappa_score(y_test, y_test_pred):.4f}")