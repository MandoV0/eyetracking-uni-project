import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from utils import extract_json_payload


class DeepSeekLabeler:
    def __init__(
        self,
        model_name="deepseek-chat",
        api_key=None,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        batch_size=64,
        timeout_seconds=90,
        max_retries=5,
        backoff_seconds=2.0,
        min_batch_size=4,
        log_path=None,
        progress_path=None,
        worker_name="worker",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.min_batch_size = min_batch_size
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        self.log_path = log_path
        self.progress_path = progress_path
        self.worker_name = worker_name
        self.aliases = {
            "gor": "gaze_off_road_ratio",
            "elr": "eyelid_low_ratio",
            "lps": "lateral_pos_std",
            "ner": "ndrt_error_rate",
            "csm": "control_smoothness",
            "jlm": "jerk_long_max",
            "als": "accel_long_std",
            "str": "steering_rate",
            "thc": "throttle_changes",
            "thw": "thw_mean",
            "shr": "short_headway_ratio",
            "hrm": "hr_mean",
        }
        self.usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self.stats = {
            "rows_completed": 0,
            "batch_failures": 0,
            "split_events": 0,
            "started_at": datetime.utcnow().isoformat() + "Z",
        }

    def label_dataframe(self, features, cache_csv=None, usage_json=None, resume=True):
        if not self.api_key:
            raise ValueError(f"DeepSeek API key missing. Set environment variable '{self.api_key_env}'.")

        work_df = features.copy().reset_index(drop=True)
        if "row_id" not in work_df.columns:
            work_df.insert(0, "row_id", np.arange(len(work_df), dtype=int))

        completed = {}
        if cache_csv and resume and os.path.exists(cache_csv):
            cached = pd.read_csv(cache_csv)
            if {"row_id", "deepseek_label"}.issubset(cached.columns):
                cached = cached.drop_duplicates(subset=["row_id"], keep="last")
                completed = dict(zip(cached["row_id"].astype(int), cached["deepseek_label"].astype(int)))
                self._log(f"Resuming DeepSeek labeling with {len(completed)} cached labels from: {cache_csv}")

        pending_rows = work_df[~work_df["row_id"].isin(completed.keys())].copy()
        if pending_rows.empty:
            self._log("No pending rows left for DeepSeek labeling.")
            return work_df["row_id"].map(completed).to_numpy(dtype=int)

        write_mode = "a" if cache_csv and resume and os.path.exists(cache_csv) else "w"
        append_header = write_mode == "w"
        start_time = time.perf_counter()

        for start in range(0, len(pending_rows), self.batch_size):
            batch_df = pending_rows.iloc[start : start + self.batch_size].copy()
            labels = self._label_batch_recursive(batch_df)
            batch_result = pd.DataFrame(
                {
                    "row_id": batch_df["row_id"].astype(int).to_numpy(),
                    "deepseek_label": np.asarray(labels, dtype=int),
                }
            )

            if cache_csv:
                os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
                batch_result.to_csv(cache_csv, mode=write_mode, index=False, header=append_header)
                write_mode = "a"
                append_header = False

            completed.update(dict(zip(batch_result["row_id"], batch_result["deepseek_label"])))
            self.stats["rows_completed"] = len(completed)
            elapsed = time.perf_counter() - start_time
            self._log(
                f"DeepSeek labeled batch rows {start + 1}-{start + len(batch_df)} / {len(pending_rows)} "
                f"in {elapsed:.1f}s"
            )

            if usage_json:
                self.save_usage(usage_json)
            self._save_progress(len(work_df), len(completed), elapsed, batch_df)

        labels_series = work_df["row_id"].map(completed)
        if labels_series.isna().any():
            missing = work_df.loc[labels_series.isna(), "row_id"].tolist()[:10]
            raise ValueError(f"Missing DeepSeek labels for rows: {missing}")

        if usage_json:
            self.save_usage(usage_json)

        return labels_series.to_numpy(dtype=int)

    def save_usage(self, usage_json):
        os.makedirs(os.path.dirname(usage_json) or ".", exist_ok=True)
        with open(usage_json, "w", encoding="utf-8") as f:
            json.dump(self.usage_totals, f, indent=2)

    def _label_batch_recursive(self, batch_df):
        try:
            return self._query_with_retries(batch_df)
        except Exception as exc:
            self.stats["batch_failures"] += 1
            if len(batch_df) <= self.min_batch_size:
                row_id = int(batch_df.iloc[0]["row_id"])
                raise RuntimeError(f"DeepSeek labeling failed for row_id={row_id}: {exc}") from exc

            mid = len(batch_df) // 2
            self.stats["split_events"] += 1
            self._log(f"Batch failed for {len(batch_df)} rows. Splitting into {mid} and {len(batch_df) - mid}.")
            left = self._label_batch_recursive(batch_df.iloc[:mid].copy())
            right = self._label_batch_recursive(batch_df.iloc[mid:].copy())
            return list(left) + list(right)

    def _query_with_retries(self, batch_df):
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return self._query_batch(batch_df)
            except Exception as exc:
                last_error = exc
                sleep_seconds = self.backoff_seconds * attempt
                self._log(f"DeepSeek batch attempt {attempt} failed: {exc}")
                if attempt <= self.max_retries:
                    time.sleep(sleep_seconds)
        raise last_error

    def _query_batch(self, batch_df):
        system_prompt = (
            "You classify driving behaviour from engineered driving and eye-tracking features. "
            "For each row, decide the most plausible class independently. "
            "Classes: 0=Attentive, 1=Inattentive, 2=Aggressive. "
            "Use the feature meanings and overall pattern, not a rigid rule list. "
            "Return JSON only with exactly this schema: {\"labels\":[...]}. "
            "No explanation, no markdown, no extra keys."
        )
        user_prompt = self._build_prompt(batch_df)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max(48, len(batch_df) * 2 + 8),
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if exc.code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Transient DeepSeek API error {exc.code}: {body[-500:]}")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {body[-500:]}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error while calling DeepSeek API: {exc}") from exc

        self._record_usage(response_json)
        text = self._extract_text(response_json)
        payload_json = extract_json_payload(text)
        if isinstance(payload_json, dict):
            labels = payload_json.get("labels")
            if labels is None:
                labels = payload_json.get("predictions")
        else:
            labels = payload_json

        if not isinstance(labels, list):
            raise ValueError("Expected JSON list under 'labels'.")
        if len(labels) != len(batch_df):
            raise ValueError(f"Expected {len(batch_df)} labels, got {len(labels)}.")

        normalized = []
        for value in labels:
            if isinstance(value, dict):
                for key in ("label", "class", "state", "_label"):
                    if key in value:
                        value = value[key]
                        break
            label = int(value)
            if label not in (0, 1, 2):
                raise ValueError(f"Invalid label returned by DeepSeek: {label}")
            normalized.append(label)
        return normalized

    def _record_usage(self, response_json):
        usage = response_json.get("usage", {})
        self.usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self.usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        self.usage_totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        self.usage_totals["requests"] += 1

    def _extract_text(self, response_json):
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError(f"No choices returned by DeepSeek: {response_json}")

        message = choices[0].get("message", {})
        text = (message.get("content") or "").strip()
        if not text:
            raise ValueError(f"DeepSeek choice contained no text payload: {response_json}")
        return text

    def _build_prompt(self, batch_df):
        available = {alias: col for alias, col in self.aliases.items() if col in batch_df.columns}
        if not available:
            raise ValueError("No DeepSeek prompt features found in dataset.")

        rows = []
        for _, row in batch_df.iterrows():
            compact_row = {}
            for alias, col in available.items():
                value = row[col]
                compact_row[alias] = round(float(value), 4) if pd.notna(value) else 0.0
            rows.append(compact_row)

        legend = ", ".join(f"{alias}={col}" for alias, col in available.items())
        return (
            f"Feature map: {legend}\n"
            "Interpretation guide:\n"
            "- gor: ratio of gaze directed away from the road; higher often means visual distraction or inattention.\n"
            "- elr: ratio of low eyelid opening; higher can indicate drowsiness or reduced alertness.\n"
            "- lps: lateral position variability; higher suggests lane-keeping instability.\n"
            "- ner: non-driving-related task error rate; higher suggests distraction or reduced attention.\n"
            "- csm: control smoothness; lower values suggest unstable or jerky control.\n"
            "- jlm: maximum longitudinal jerk; high values suggest abrupt manoeuvres.\n"
            "- als: longitudinal acceleration variability; high values suggest inconsistent acceleration behaviour.\n"
            "- str: steering rate; high values suggest abrupt steering corrections.\n"
            "- thc: throttle changes; high values suggest unstable pedal behaviour.\n"
            "- thw: time headway to vehicle ahead; very low values can indicate risky following.\n"
            "- shr: ratio of short headway events; higher values suggest risky proximity driving.\n"
            "- hrm: mean heart rate; elevated values may support stress or workload but are not decisive alone.\n"
            "Labeling goal:\n"
            "- Attentive: stable control, road-focused gaze, no strong distraction or risky manoeuvre pattern.\n"
            "- Inattentive: distraction, off-road gaze, drowsiness, unstable lane keeping, or poor NDRT-related attention pattern.\n"
            "- Aggressive: abrupt control, risky headway, jerky steering/acceleration, forceful manoeuvre pattern.\n"
            "Use holistic judgment across the row. Do not force a class from one feature alone unless it is very strong.\n"
            f"Rows: {rows}"
        )

    def _log(self, message):
        line = f"[{datetime.utcnow().isoformat()}Z][{self.worker_name}] {message}"
        print(line)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _save_progress(self, total_rows, completed_rows, elapsed_seconds, batch_df):
        if not self.progress_path:
            return
        payload = {
            "worker": self.worker_name,
            "total_rows": int(total_rows),
            "completed_rows": int(completed_rows),
            "remaining_rows": int(max(0, total_rows - completed_rows)),
            "elapsed_seconds": round(float(elapsed_seconds), 2),
            "rows_per_second": round(completed_rows / elapsed_seconds, 4) if elapsed_seconds > 0 else None,
            "last_batch_size": int(len(batch_df)),
            "last_row_id": int(batch_df["row_id"].max()),
            "usage": self.usage_totals,
            "stats": self.stats,
        }
        os.makedirs(os.path.dirname(self.progress_path) or ".", exist_ok=True)
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
