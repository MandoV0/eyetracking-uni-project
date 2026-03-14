import argparse
import json
import math
import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(SCRIPT_DIR, "main.py")
DEFAULT_PARTS_DIR = os.path.join(SCRIPT_DIR, "cache", "parts")
DEFAULT_PROGRESS_DIR = os.path.join(DEFAULT_PARTS_DIR, "progress")
DEFAULT_LOG_DIR = os.path.join(DEFAULT_PARTS_DIR, "logs")
DEFAULT_ORIGINAL_PROCESSED = os.path.join(SCRIPT_DIR, "..", "LightGBMPipeline", "cache", "features_labels_full.csv")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_worker_command(args, worker_idx, row_start, row_end):
    part_csv = os.path.join(args.parts_dir, f"deepseek_labels_part_{worker_idx:02d}.csv")
    usage_json = os.path.join(args.parts_dir, f"deepseek_usage_part_{worker_idx:02d}.json")
    log_path = os.path.join(args.log_dir, f"worker_{worker_idx:02d}.log")
    progress_json = os.path.join(args.progress_dir, f"worker_{worker_idx:02d}.json")

    cmd = [
        sys.executable,
        MAIN_PY,
        "label-deepseek",
        "--model-name",
        args.model_name,
        "--features-csv",
        args.features_csv,
        "--cache-csv",
        part_csv,
        "--usage-json",
        usage_json,
        "--batch-size",
        str(args.batch_size),
        "--max-rows",
        str(args.max_rows),
        "--row-start",
        str(row_start),
        "--row-end",
        str(row_end),
        "--worker-name",
        f"worker-{worker_idx:02d}",
        "--log-path",
        log_path,
        "--progress-json",
        progress_json,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-retries",
        str(args.max_retries),
        "--backoff-seconds",
        str(args.backoff_seconds),
        "--min-batch-size",
        str(args.min_batch_size),
    ]
    if args.no_resume:
        cmd.append("--no-resume")
    return cmd


def summarize_progress(progress_dir):
    completed = 0
    total = 0
    workers = []
    if not os.path.isdir(progress_dir):
        return {"completed": 0, "total": 0, "workers": []}

    for name in sorted(os.listdir(progress_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(progress_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        completed += int(data.get("completed_rows", 0) or 0)
        total += int(data.get("total_rows", 0) or 0)
        workers.append(data)
    return {"completed": completed, "total": total, "workers": workers}


def ensure_unlabeled_features(args):
    if os.path.exists(args.features_csv):
        return

    if not os.path.exists(DEFAULT_ORIGINAL_PROCESSED):
        raise FileNotFoundError(
            f"Unlabeled feature CSV missing: {args.features_csv}\n"
            f"Auto-bootstrap source also missing: {DEFAULT_ORIGINAL_PROCESSED}"
        )

    cmd = [
        sys.executable,
        MAIN_PY,
        "build-features",
        "--bootstrap-from-processed",
        DEFAULT_ORIGINAL_PROCESSED,
        "--output-csv",
        args.features_csv,
        "--max-rows",
        str(args.max_rows),
    ]
    print(f"Auto-building unlabeled features into: {args.features_csv}")
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser("Parallel DeepSeek label runner")
    parser.add_argument("--features-csv", default=os.path.join(SCRIPT_DIR, "cache", "features_unlabeled.csv"))
    parser.add_argument("--output-csv", default=os.path.join(SCRIPT_DIR, "cache", "features_labels_deepseek.csv"))
    parser.add_argument("--output-usage-json", default=os.path.join(SCRIPT_DIR, "cache", "deepseek_usage.json"))
    parser.add_argument("--parts-dir", default=DEFAULT_PARTS_DIR)
    parser.add_argument("--progress-dir", default=DEFAULT_PROGRESS_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.parts_dir)
    ensure_dir(args.progress_dir)
    ensure_dir(args.log_dir)
    ensure_unlabeled_features(args)

    chunk_size = math.ceil(args.max_rows / args.workers)
    processes = []
    print(f"Launching {args.workers} workers with batch_size={args.batch_size}, chunk_size={chunk_size}")

    for worker_idx in range(args.workers):
        row_start = worker_idx * chunk_size
        row_end = min(args.max_rows, row_start + chunk_size)
        if row_start >= row_end:
            continue
        cmd = build_worker_command(args, worker_idx, row_start, row_end)
        proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
        processes.append((worker_idx, row_start, row_end, proc))
        print(f"Started worker-{worker_idx:02d}: rows {row_start}-{row_end}")

    while True:
        alive = 0
        for _, _, _, proc in processes:
            if proc.poll() is None:
                alive += 1

        progress = summarize_progress(args.progress_dir)
        total = progress["total"] or args.max_rows
        completed = progress["completed"]
        pct = (completed / total * 100) if total else 0
        print(f"Parallel progress: {completed}/{total} rows ({pct:.1f}%), alive_workers={alive}")

        if alive == 0:
            break
        time.sleep(args.poll_seconds)

    failed = [f"worker-{idx:02d}" for idx, _, _, proc in processes if proc.returncode != 0]
    if failed:
        raise SystemExit(f"Some workers failed: {', '.join(failed)}")

    merge_cmd = [
        sys.executable,
        MAIN_PY,
        "merge-deepseek-parts",
        "--features-csv",
        args.features_csv,
        "--parts-dir",
        args.parts_dir,
        "--output-csv",
        args.output_csv,
        "--output-usage-json",
        args.output_usage_json,
        "--max-rows",
        str(args.max_rows),
    ]
    subprocess.run(merge_cmd, cwd=SCRIPT_DIR, check=True)
    print("Parallel DeepSeek labeling completed and merged.")


if __name__ == "__main__":
    main()
