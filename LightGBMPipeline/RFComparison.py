import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

def compare_with_random_forest(features, labels, session_ids, classifier, cv_results=None, random_state=42):
    print("\n" + "="*70)
    print(" === START: RANDOM FOREST COMPARISON ===")
    print("="*70)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=30, min_samples_leaf=25,
        max_features='sqrt', random_state=random_state, class_weight='balanced') # Class weight Balance to make sure Model doesnt train on "biased" data.
    
    # Prepare features using the same scaler as LightGBM
    X = classifier.prepare_features_for_training(features)
    
    # Fit scaler if not already fitted
    if not hasattr(classifier.scaler, 'mean_') or classifier.scaler.mean_ is None:
        X_scaled = classifier.scaler.fit_transform(X)
    else:
        X_scaled = classifier.scaler.transform(X)
    
    # Cross-validation
    logo = LeaveOneGroupOut()
    rf_train_accs = []
    rf_test_accs = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, labels, session_ids)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        rf_model.fit(X_train, y_train)
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)
        
        rf_train_accs.append(accuracy_score(y_train, train_pred))
        rf_test_accs.append(accuracy_score(y_test, test_pred))
    
    print(f"RF Train Accuracy: {np.mean(rf_train_accs):.4f}")
    print(f"RF Test Accuracy:  {np.mean(rf_test_accs):.4f} ± {np.std(rf_test_accs):.4f}")
    
    if cv_results is not None:
        print(f"\nComparison:")
        print(f"  LightGBM Train:       {cv_results['mean_train_accuracy']:.4f}")
        print(f"  LightGBM Test:        {cv_results['mean_test_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"  Random Forest Train:  {np.mean(rf_train_accs):.4f}")
        print(f"  Random Forest Test:   {np.mean(rf_test_accs):.4f} ± {np.std(rf_test_accs):.4f}")
        print(f"  Test Difference:      {cv_results['mean_test_accuracy'] - np.mean(rf_test_accs):+.4f}")
    
    return {
        'mean_train_accuracy': np.mean(rf_train_accs),
        'mean_test_accuracy': np.mean(rf_test_accs),
        'std_test_accuracy': np.std(rf_test_accs)
    }