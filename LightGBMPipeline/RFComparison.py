import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)


def compare_with_random_forest(features, labels, session_ids, classifier, random_state=42, test_data=None, test_size=0.2):
    """
    Compare Random Forest with LightGBM using a simple train/test split (no CV).
    
    Args:
        features: DataFrame with features (train set)
        labels: Array with labels (train set)
        session_ids: Array with session IDs (for session-based split if test_data not provided)
        classifier: DriverStateClassifier instance (for feature preparation/scaling)
        random_state: Random seed
        test_data: Optional tuple (X_test_df, y_test) - if provided, uses this test set
        test_size: Test set size ratio (used if test_data is None)
    """
    print("\n" + "="*70)
    print(" === START: RANDOM FOREST TRAINING ===")
    print("="*70)
    
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=30, min_samples_leaf=25,
        max_features='sqrt', random_state=random_state, class_weight='balanced', n_jobs=16)
    
    # Prepare features using the same feature selection as LightGBM
    X = classifier.prepare_features_for_training(features)

    # If explicit test data provided, use it
    if test_data is not None:
        X_test_df, y_test = test_data
        X_test = classifier.prepare_features_for_training(X_test_df)
        
        # Fit
        X_train_scaled = classifier.scaler.fit_transform(X)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        X_train_final = X_train_scaled
        y_train_final = labels
        X_test_final = X_test_scaled
        y_test_final = y_test
    else:
        # Do a session-based train/test split
        if session_ids is None:
            raise ValueError("RFComparison: Either provide test_data or session_ids for splitting.")
        
        unique_sessions = np.unique(session_ids)
        train_sessions, test_sessions = train_test_split(
            unique_sessions, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        train_mask = np.isin(session_ids, train_sessions)
        test_mask = np.isin(session_ids, test_sessions)
        
        X_train_df = features.loc[train_mask].reset_index(drop=True)
        X_test_df = features.loc[test_mask].reset_index(drop=True)
        y_train_final = labels[train_mask]
        y_test_final = labels[test_mask]
        
        X_train = classifier.prepare_features_for_training(X_train_df)
        X_test = classifier.prepare_features_for_training(X_test_df)
        
        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train and evaluate RF
    rf_model.fit(X_train_final, y_train_final)
    train_pred = rf_model.predict(X_train_final)
    test_pred = rf_model.predict(X_test_final)

    train_acc = accuracy_score(y_train_final, train_pred)
    test_acc = accuracy_score(y_test_final, test_pred)

    print(f"RF Train Accuracy: {train_acc:.4f}")
    print(f"RF Test Accuracy:  {test_acc:.4f}")

    print("\nRF CLASSIFICATION REPORT (test set)")
    print("-----------------------------------")
    print(classification_report(y_test_final, test_pred, digits=4))

    print("\nRF CONFUSION MATRIX (test set)")
    print("------------------------------")
    cm = confusion_matrix(y_test_final, test_pred)
    print(cm)

    kappa = cohen_kappa_score(y_test_final, test_pred)
    print(f"\nRF Cohen's Kappa (test): {kappa:.4f}")