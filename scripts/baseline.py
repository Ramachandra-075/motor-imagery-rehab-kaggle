"""Test baseline-relative and spatially referenced EEG power features."""
import itertools
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from train import CHANNELS, FS, balanced_labels, discover
from search import model

BANDS = [(4, 8), (8, 12), (12, 16), (16, 22), (22, 30), (30, 40)]
MODES = ("post", "ratio", "post_ratio", "post_halves_ratio")
REFERENCES = ("raw", "car", "laplacian")
MODELS = ("logreg03", "logreg3", "lda")


def trials(z, name, labeled):
    with z.open(name) as stream:
        frame = pd.read_csv(stream, low_memory=False)
    marks = frame.Marker_val.fillna("").astype(str).to_numpy()
    labels = ("move", "rest") if labeled else ("cue_start",)
    indices = np.flatnonzero(np.isin(marks, labels))
    signal = frame[CHANNELS].to_numpy(dtype=np.float32).T
    x, y = [], []
    for i in indices:
        window = signal[:, i-FS:i+1125]  # 1 sec before, 4.5 sec after onset.
        if window.shape != (8, 1375):
            raise ValueError(f"Invalid epoch at {name}: {i}")
        window = np.nan_to_num(window)
        x.append(window)
        if labeled:
            y.append(int(marks[i] == "move"))
    return np.stack(x), np.array(y)


def rereference(x, reference):
    if reference == "raw":
        return x
    if reference == "car":
        return x - np.mean(x, axis=1, keepdims=True)
    # Local motor cortex derivations, plus original channels for context.
    extra = np.stack((
        x[:, 1] - (x[:, 0] + x[:, 2] + x[:, 4]) / 3,
        x[:, 2] - (x[:, 0] + x[:, 1] + x[:, 3] + x[:, 5]) / 4,
        x[:, 3] - (x[:, 0] + x[:, 2] + x[:, 6]) / 3,
    ), axis=1)
    return np.concatenate((x, extra), axis=1)


def build_features(x, reference, mode):
    x = rereference(x, reference)
    parts = []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype="bandpass", fs=FS, output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1)
        baseline = np.log(np.var(filtered[:, :, 0:250], axis=-1) + 1e-10)
        post = np.log(np.var(filtered[:, :, 375:1375], axis=-1) + 1e-10)
        if mode in ("post", "post_ratio", "post_halves_ratio"):
            parts.append(post)
        if mode in ("ratio", "post_ratio", "post_halves_ratio"):
            parts.append(post - baseline)
        if mode == "post_halves_ratio":
            parts.append(np.log(np.var(filtered[:, :, 375:875], axis=-1) + 1e-10))
            parts.append(np.log(np.var(filtered[:, :, 875:1375], axis=-1) + 1e-10))
    return np.hstack(parts)


def main():
    results = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                   "other_hits": 0, "other_trials": 0})
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = trials(z, path, True)
                sessions.append((sid, x, y))
            x = np.concatenate([item[1] for item in sessions])
            y = np.concatenate([item[2] for item in sessions])
            group = np.concatenate([np.full(len(item[2]), item[0]) for item in sessions])
            for reference, mode in itertools.product(REFERENCES, MODES):
                feat = build_features(x, reference, mode)
                for name in MODELS:
                    key = "/".join((reference, mode, name))
                    checks = [sessions[-1][0]]
                    if len(sessions) >= 3:
                        checks.append(sessions[1][0])
                    for split_i, sid in enumerate(checks):
                        val = group == sid
                        clf = model(name)
                        clf.fit(feat[~val], y[~val])
                        prob = clf.predict_proba(feat[val])[:, 1]
                        pred = balanced_labels(prob, len(prob)//2)
                        split = "recent" if split_i == 0 else "other"
                        results[key][split+"_hits"] += int(np.sum(pred == y[val]))
                        results[key][split+"_trials"] += len(pred)
            print(subject, flush=True)
    rows = []
    for key, val in results.items():
        rows.append({"config": key, **val,
                     "recent_accuracy": val["recent_hits"]/val["recent_trials"],
                     "other_accuracy": val["other_hits"]/val["other_trials"]})
    rows.sort(key=lambda item: (item["other_accuracy"], item["recent_accuracy"]), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/baseline.json").write_text(json.dumps(rows, indent=2))
    print("TOP", json.dumps(rows[:15], indent=2))


if __name__ == "__main__":
    main()
