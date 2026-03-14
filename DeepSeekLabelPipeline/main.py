import argparse
import json
import math
import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from AbsoluteThresholdLabeler import AbsoluteThresholdLabeler
from DeepSeekLabeler import DeepSeekLabeler
from DriverStateClassifier import DriverStateClassifier
from data_processing import (
    LINEAR,
    bootstrap_unlabeled_from_processed_csv,
    build_features_from_raw,
    load_labeled_dataset_csv,
    load_unlabeled_features_csv,
    save_labeled_dataset_csv,
    save_unlabeled_features_csv,
)
from utils import session_train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UNLABELED_CSV = os.path.join(SCRIPT_DIR, "cache", "features_unlabeled.csv")
DEFAULT_ABS_CSV = os.path.join(SCRIPT_DIR, "cache", "features_labels_abs.csv")
DEFAULT_DEEPSEEK_CSV = os.path.join(SCRIPT_DIR, "cache", "features_labels_deepseek.csv")
DEFAULT_DEEPSEEK_CACHE_CSV = os.path.join(SCRIPT_DIR, "cache", "deepseek_labels.csv")
DEFAULT_DEEPSEEK_USAGE_JSON = os.path.join(SCRIPT_DIR, "cache", "deepseek_usage.json")
DEFAULT_DEEPSEEK_LOG = os.path.join(SCRIPT_DIR, "cache", "deepseek.log")
DEFAULT_DEEPSEEK_PROGRESS_JSON = os.path.join(SCRIPT_DIR, "cache", "deepseek_progress.json")
DEFAULT_RESULTS_JSON = os.path.join(SCRIPT_DIR, "results", "compare_summary.json")
DEFAULT_ORIGINAL_PROCESSED = os.path.join(SCRIPT_DIR, "..", "LightGBMPipeline", "cache", "features_labels_full.csv")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _json_safe(payload):
    safe = {}
    for key, value in payload.items():
        if hasattr(value, "tolist"):
            safe[key] = value.tolist()
        elif isinstance(value, float) and not math.isfinite(value):
            safe[key] = None
        elif isinstance(value, (int, float, str, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = value
    return safe


def _save_json(payload, out_path):
    _ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON summary to: {out_path}")


def _auto_prepare_unlabeled_if_missing(features_csv, max_rows):
    if os.path.exists(features_csv):
        return

    if not os.path.exists(DEFAULT_ORIGINAL_PROCESSED):
        raise FileNotFoundError(
            f"Unlabeled feature CSV not found: {features_csv}\n"
            f"Auto-bootstrap source also missing: {DEFAULT_ORIGINAL_PROCESSED}"
        )

    print(f"Unlabeled feature CSV missing. Auto-bootstrapping from: {DEFAULT_ORIGINAL_PROCESSED}")
    bootstrap_unlabeled_from_processed_csv(
        processed_csv=DEFAULT_ORIGINAL_PROCESSED,
        out_path=features_csv,
        max_rows=max_rows,
    )


def build_features_command(args):
    if args.bootstrap_from_processed:
        return bootstrap_unlabeled_from_processed_csv(
            processed_csv=args.bootstrap_from_processed,
            out_path=args.output_csv,
            max_rows=args.max_rows,
        )

    features = build_features_from_raw(
        max_files=args.max_files,
        interpolation_mode=args.interpolation,
        window_size=args.window_size,
    )
    if args.max_rows is not None and len(features) > args.max_rows:
        features = features.head(args.max_rows).copy()
    save_unlabeled_features_csv(features, args.output_csv)
    return features


def label_abs_command(args):
    _auto_prepare_unlabeled_if_missing(args.features_csv, args.max_rows)
    features = load_unlabeled_features_csv(args.features_csv, max_rows=args.max_rows)
    labeler = AbsoluteThresholdLabeler(use_data_driven_thresholds=True)
    features_for_save = features.copy()
    labels = labeler.generate_labels(features_for_save)
    save_labeled_dataset_csv(features_for_save, labels, args.output_csv)
    return labels


def label_deepseek_command(args):
    _auto_prepare_unlabeled_if_missing(args.features_csv, args.max_rows)
    features = load_unlabeled_features_csv(args.features_csv, max_rows=args.max_rows)
    if args.row_start is not None or args.row_end is not None:
        start = args.row_start or 0
        end = args.row_end if args.row_end is not None else len(features)
        features = features.iloc[start:end].reset_index(drop=True)
        print(f"Selected row window: start={start}, end={end}, rows={len(features)}")
    labeler = DeepSeekLabeler(
        model_name=args.model_name,
        api_key_env=args.api_key_env,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        base_url=args.base_url,
        min_batch_size=args.min_batch_size,
        log_path=args.log_path,
        progress_path=args.progress_json,
        worker_name=args.worker_name,
    )
    labels = labeler.label_dataframe(
        features,
        cache_csv=args.cache_csv,
        usage_json=args.usage_json,
        resume=not args.no_resume,
    )
    save_labeled_dataset_csv(features, labels, args.output_csv)
    return labels


def _evaluate_branch(input_csv, eval_mode, test_size, random_state, use_gpu, artifact_path=None):
    features, labels, session_ids, _ = load_labeled_dataset_csv(input_csv)
    classifier = DriverStateClassifier(device="gpu" if use_gpu else "cpu")

    if eval_mode == "cv":
        if len(pd.unique(session_ids)) < 2:
            print("Only one session available. Falling back from CV to split.")
        else:
            metrics = classifier.train_with_cross_validation(features, labels, session_ids)
            if artifact_path:
                classifier.train_final_model(features, labels)
                classifier.save_artifacts(artifact_path)
            return metrics

    X_train, X_test, y_train, y_test = session_train_test_split(
        features,
        labels,
        session_ids,
        test_size=test_size,
        random_state=random_state,
    )
    metrics = classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
    if artifact_path:
        classifier.train_final_model(features, labels)
        classifier.save_artifacts(artifact_path)
    return metrics


def train_command(args):
    if args.input_csv:
        input_csv = args.input_csv
        suffix = args.model_suffix or "custom"
    elif args.label_source == "abs":
        input_csv = DEFAULT_ABS_CSV
        suffix = "abs"
    else:
        input_csv = DEFAULT_DEEPSEEK_CSV
        suffix = "deepseek"

    artifact_path = None
    if not args.no_save_model:
        artifact_path = os.path.join(DEFAULT_MODEL_DIR, f"driver_state_lgbm_{suffix}.pkl")

    return _evaluate_branch(
        input_csv=input_csv,
        eval_mode=args.eval_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        artifact_path=artifact_path,
    )


def compare_command(args):
    _, _, _, abs_df = load_labeled_dataset_csv(args.abs_csv)
    _, _, _, deepseek_df = load_labeled_dataset_csv(args.deepseek_csv)

    abs_labels = abs_df[["row_id", "_label"]].rename(columns={"_label": "abs_label"})
    deepseek_labels = deepseek_df[["row_id", "_label"]].rename(columns={"_label": "deepseek_label"})
    joined = abs_labels.merge(deepseek_labels, on="row_id", how="inner")
    if joined.empty:
        raise ValueError("No overlapping row_id values between ABS and DeepSeek labeled datasets.")

    label_summary = {
        "rows_compared": int(len(joined)),
        "agreement_accuracy": float(accuracy_score(joined["abs_label"], joined["deepseek_label"])),
        "cohen_kappa": _finite_or_none(cohen_kappa_score(joined["abs_label"], joined["deepseek_label"])),
        "abs_distribution": {str(k): int(v) for k, v in joined["abs_label"].value_counts().sort_index().items()},
        "deepseek_distribution": {str(k): int(v) for k, v in joined["deepseek_label"].value_counts().sort_index().items()},
    }

    abs_metrics = _evaluate_branch(
        input_csv=args.abs_csv,
        eval_mode=args.eval_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        artifact_path=None,
    )
    deepseek_metrics = _evaluate_branch(
        input_csv=args.deepseek_csv,
        eval_mode=args.eval_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        artifact_path=None,
    )

    summary = {
        "label_agreement": label_summary,
        "abs_lgbm_metrics": _json_safe(abs_metrics),
        "deepseek_lgbm_metrics": _json_safe(deepseek_metrics),
    }
    _save_json(summary, args.output_json)
    return summary


def merge_deepseek_parts_command(args):
    part_dir = args.parts_dir
    if not os.path.isdir(part_dir):
        raise FileNotFoundError(f"Parts directory not found: {part_dir}")

    part_files = sorted(
        os.path.join(part_dir, name)
        for name in os.listdir(part_dir)
        if name.endswith(".csv") and "deepseek_labels_part_" in name
    )
    if not part_files:
        raise ValueError(f"No part CSV files found in: {part_dir}")

    merged_parts = [pd.read_csv(path) for path in part_files]
    labels_df = pd.concat(merged_parts, ignore_index=True)
    labels_df = labels_df.drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id").reset_index(drop=True)

    features = load_unlabeled_features_csv(args.features_csv, max_rows=args.max_rows)
    merged = features.merge(labels_df, on="row_id", how="inner")
    if merged.empty:
        raise ValueError("Merged DeepSeek parts contained no matching row_id values.")

    merged = merged.rename(columns={"deepseek_label": "_label"})
    _ensure_parent(args.output_csv)
    merged.to_csv(args.output_csv, index=False)
    print(f"Saved merged DeepSeek labeled dataset to: {args.output_csv}")

    if args.output_usage_json:
        usage_files = sorted(
            os.path.join(part_dir, name)
            for name in os.listdir(part_dir)
            if name.endswith(".json") and "deepseek_usage_part_" in name
        )
        usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
        for path in usage_files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in usage_sum:
                usage_sum[key] += int(data.get(key, 0) or 0)
        _save_json(usage_sum, args.output_usage_json)
    return merged


def build_parser():
    parser = argparse.ArgumentParser("DeepSeek-assisted driver-state labeling pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-features", help="Build and save unlabeled feature CSV")
    build_cmd.add_argument("--output-csv", default=DEFAULT_UNLABELED_CSV)
    build_cmd.add_argument("--bootstrap-from-processed", default=None)
    build_cmd.add_argument("--max-files", type=int, default=None)
    build_cmd.add_argument("--window-size", type=int, default=50)
    build_cmd.add_argument("--interpolation", type=int, default=LINEAR, choices=[0, 1, 2])
    build_cmd.add_argument("--max-rows", type=int, default=None)

    abs_cmd = subparsers.add_parser("label-abs", help="Generate ABS labels from unlabeled feature CSV")
    abs_cmd.add_argument("--features-csv", default=DEFAULT_UNLABELED_CSV)
    abs_cmd.add_argument("--output-csv", default=DEFAULT_ABS_CSV)
    abs_cmd.add_argument("--max-rows", type=int, default=None)

    deepseek_cmd = subparsers.add_parser("label-deepseek", help="Generate DeepSeek labels from unlabeled feature CSV")
    deepseek_cmd.add_argument("--features-csv", default=DEFAULT_UNLABELED_CSV)
    deepseek_cmd.add_argument("--output-csv", default=DEFAULT_DEEPSEEK_CSV)
    deepseek_cmd.add_argument("--cache-csv", default=DEFAULT_DEEPSEEK_CACHE_CSV)
    deepseek_cmd.add_argument("--usage-json", default=DEFAULT_DEEPSEEK_USAGE_JSON)
    deepseek_cmd.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    deepseek_cmd.add_argument("--base-url", default="https://api.deepseek.com")
    deepseek_cmd.add_argument("--model-name", default="deepseek-chat")
    deepseek_cmd.add_argument("--batch-size", type=int, default=64)
    deepseek_cmd.add_argument("--timeout-seconds", type=int, default=90)
    deepseek_cmd.add_argument("--max-retries", type=int, default=5)
    deepseek_cmd.add_argument("--backoff-seconds", type=float, default=2.0)
    deepseek_cmd.add_argument("--min-batch-size", type=int, default=4)
    deepseek_cmd.add_argument("--max-rows", type=int, default=None)
    deepseek_cmd.add_argument("--row-start", type=int, default=None)
    deepseek_cmd.add_argument("--row-end", type=int, default=None)
    deepseek_cmd.add_argument("--worker-name", default="worker")
    deepseek_cmd.add_argument("--log-path", default=DEFAULT_DEEPSEEK_LOG)
    deepseek_cmd.add_argument("--progress-json", default=DEFAULT_DEEPSEEK_PROGRESS_JSON)
    deepseek_cmd.add_argument("--no-resume", action="store_true")

    merge_cmd = subparsers.add_parser("merge-deepseek-parts", help="Merge parallel DeepSeek part files")
    merge_cmd.add_argument("--features-csv", default=DEFAULT_UNLABELED_CSV)
    merge_cmd.add_argument("--parts-dir", default=os.path.join(SCRIPT_DIR, "cache", "parts"))
    merge_cmd.add_argument("--output-csv", default=DEFAULT_DEEPSEEK_CSV)
    merge_cmd.add_argument("--output-usage-json", default=DEFAULT_DEEPSEEK_USAGE_JSON)
    merge_cmd.add_argument("--max-rows", type=int, default=None)

    train_cmd = subparsers.add_parser("train", help="Train LGBM on ABS or DeepSeek labeled dataset")
    train_cmd.add_argument("--input-csv", default=None)
    train_cmd.add_argument("--label-source", choices=["abs", "deepseek"], default="abs")
    train_cmd.add_argument("--model-suffix", default=None)
    train_cmd.add_argument("--eval-mode", choices=["split", "cv"], default="split")
    train_cmd.add_argument("--test-size", type=float, default=0.2)
    train_cmd.add_argument("--random-state", type=int, default=42)
    train_cmd.add_argument("--use-gpu", action="store_true")
    train_cmd.add_argument("--no-save-model", action="store_true")

    compare_cmd = subparsers.add_parser("compare", help="Compare ABS and DeepSeek labels plus downstream LGBM")
    compare_cmd.add_argument("--abs-csv", default=DEFAULT_ABS_CSV)
    compare_cmd.add_argument("--deepseek-csv", default=DEFAULT_DEEPSEEK_CSV)
    compare_cmd.add_argument("--output-json", default=DEFAULT_RESULTS_JSON)
    compare_cmd.add_argument("--eval-mode", choices=["split", "cv"], default="split")
    compare_cmd.add_argument("--test-size", type=float, default=0.2)
    compare_cmd.add_argument("--random-state", type=int, default=42)
    compare_cmd.add_argument("--use-gpu", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-features" and args.bootstrap_from_processed is None and os.path.exists(DEFAULT_ORIGINAL_PROCESSED):
        print(
            "Hint: original processed cache exists. For quick iteration you can use "
            f"--bootstrap-from-processed \"{DEFAULT_ORIGINAL_PROCESSED}\""
        )

    start = time.perf_counter()
    if args.command == "build-features":
        build_features_command(args)
    elif args.command == "label-abs":
        label_abs_command(args)
    elif args.command == "label-deepseek":
        label_deepseek_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "compare":
        compare_command(args)
    elif args.command == "merge-deepseek-parts":
        merge_deepseek_parts_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    elapsed = time.perf_counter() - start
    print(f"Finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
