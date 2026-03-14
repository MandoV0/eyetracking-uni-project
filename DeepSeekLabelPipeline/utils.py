import json
import re

import numpy as np
from sklearn.model_selection import train_test_split


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text):
    if text is None:
        return ""
    return ANSI_ESCAPE_RE.sub("", text)


def _try_json(candidate):
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_balanced_json(text):
    text = text.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        payload = _try_json(text[start : idx + 1])
                        if payload is not None:
                            return payload
                        break
            start = text.find(opener, start + 1)
    return None


def extract_json_payload(text):
    cleaned = strip_ansi(text).strip()
    direct = _try_json(cleaned)
    if direct is not None:
        return direct

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        payload = _try_json(block)
        if payload is not None:
            return payload

    payload = _extract_balanced_json(cleaned)
    if payload is not None:
        return payload

    raise ValueError("No valid JSON payload found in Ollama response.")


def session_train_test_split(features, labels, session_ids, test_size=0.2, random_state=42):
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        print("Warning: Only one session in data. Falling back to row-wise train/test split.")
        X_train, X_test, y_train, y_test = train_test_split(
            features.reset_index(drop=True),
            labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        return X_train, X_test, y_train, y_test

    train_sessions, test_sessions = train_test_split(
        unique_sessions,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_mask = np.isin(session_ids, train_sessions)
    test_mask = np.isin(session_ids, test_sessions)

    X_train = features.loc[train_mask].reset_index(drop=True)
    y_train = labels[train_mask]
    X_test = features.loc[test_mask].reset_index(drop=True)
    y_test = labels[test_mask]
    return X_train, X_test, y_train, y_test
