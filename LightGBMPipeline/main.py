import time

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split
from DriverStateClassifier import DriverStateClassifier
from AbsoluteThresholdLabeler import AbsoluteThresholdLabeler
from RFComparison import compare_with_random_forest

CACHE_PATH = "cache/features_labels_6fb5efcc.csv"

def load_dataset(csv_path, max_rows=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)

    labels = df["_label"].values
    session_ids = df["session_id"].values

    feature_cols = [c for c in df.columns if c not in ("_label", "session_id")]
    features = df[feature_cols].copy()
    features["session_id"] = session_ids

    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")

    return features, labels, session_ids


def session_train_test_split(features, labels, session_ids, test_size, random_state):
    unique_sessions = np.unique(session_ids)

    train_sess, test_sess = train_test_split(unique_sessions, test_size=test_size, random_state=random_state, shuffle=True)

    train_mask = np.isin(session_ids, train_sess)
    test_mask = np.isin(session_ids, test_sess)

    X_train = features.loc[train_mask].reset_index(drop=True)
    y_train = labels[train_mask]

    X_test = features.loc[test_mask].reset_index(drop=True)
    y_test = labels[test_mask]

    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser("Driver State Classification (CSV-based)")

    # Evaluation
    parser.add_argument("--eval-mode", choices=["cv", "split"], default="cv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # Misc
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--compare-rf", action="store_true")
    parser.add_argument("--feature-importance", action="store_true")

    # GPU
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-platform-id", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)

    # Experimental
    parser.add_argument("--lstm", action="store_true", default=False)

    args = parser.parse_args()

    print("=" * 70)
    print("--- DRIVER STATE CLASSIFICATION ---")
    print(f"=> EVALUATION MODE: {args.eval_mode}")
    print("=" * 70)

    # LOAD DATA
    features, labels, session_ids = load_dataset(CACHE_PATH, max_rows=args.max_rows)

    labeler = AbsoluteThresholdLabeler(use_data_driven_thresholds=True)

    # My initial thought was to add GPU support so the Model training is fast which is important for quick iterations of the pipeline.
    # At 5.000.000 rows of data the GPU is approxametly 15% faster. At 1 Million Rows they are around the same speed.
    device = "gpu" if args.use_gpu else "cpu"
    print(f"Device: {device.upper()}")

    classifier = DriverStateClassifier(
        device=device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
    )
    classifier.labeler = labeler

    # EVALUATION
    if args.eval_mode == "cv":
        print("\nRunning cross-validation...")
        cv_results = classifier.train_with_cross_validation(features, labels, session_ids)

        if args.compare_rf: compare_with_random_forest(features, labels, session_ids, classifier, cv_results)

    else:
        print("\nRunning session-based train/test split...")
        X_train, X_test, y_train, y_test = session_train_test_split(features, labels, session_ids, args.test_size, args.random_state)

        print(f"Train samples: {len(X_train)}")
        print(f"Test samples : {len(X_test)}")

        classifier.train_and_evaluate(X_train, y_train, X_test, y_test)

        if args.compare_rf:
            compare_with_random_forest(X_train, y_train, None, classifier, test_data=(X_test, y_test))

    if args.feature_importance:
        print("\nFeature importance:")
        classifier.get_feature_importance(top_n=20)

    print("\n === !!! DONE !!! ===")
    print("=" * 70)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n\n\nTotal runtime: {elapsed:.2f} seconds")

"""
Notes:
Need at least 50.000 Rows for training or model doesnt have enough data.

Benchmarks GPU vs CPU.
GPU with 100.000 Rows takes roughly 6.05 Seconds. (Excluding Feature Engineering)

1.000.000 Rows CPU : 16.56s
1.000.000 Rows GPU : 16.40s

5.000.000 Rows GPU : 80.41 Seconds
5.000.000 Rows CPU : 95.09 Seconds

95.09 - 80.41 = 14.68s faster
Which is 14.68 / 95.09 = 15% Faster

CPU: Ryzen 5 5600x
GPU: AMD RX 6700

Total Lines: 5037535

Benchmarks Model:
Train Test Split: 0.8/0.2
On all Data.
======================================================================
DRIVER STATE CLASSIFICATION
EVALUATION MODE: SPLIT
======================================================================
Loading dataset: cache/features_labels_6fb5efcc.csv
Samples: 5037534
Features: 37
Device: CPU

Running session-based train/test split...
Train samples: 4010219
Test samples : 1027315

TRAIN / TEST RESULTS
============================================================
Train Accuracy: 0.9246
Test  Accuracy: 0.9197
Train-Test Gap: +0.0049

TEST SET CLASSIFICATION REPORT:
              precision    recall  f1-score   support

   Attentive       0.93      0.95      0.94    524109
 Inattentive       0.91      0.89      0.90    246694
  Aggressive       0.91      0.89      0.90    256512

    accuracy                           0.92   1027315
   macro avg       0.92      0.91      0.91   1027315
weighted avg       0.92      0.92      0.92   1027315

TEST SET CONFUSION MATRIX:
                   Attentive  Inattentive   Aggressive
      Attentive      497937       10009       16163
    Inattentive       22882      218734        5078
     Aggressive       15490       12865      228157

Test Cohen's Kappa: 0.8698

 DONE
======================================================================

"""