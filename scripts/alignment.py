"""Evaluate unsupervised session alignment for fixed V2 EEG model."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from train import balanced_labels, discover, read_epochs
from search import BAND_SETS, features, model
from submit_v2 import make_bank

MODES = ("none", "mean25", "mean50", "mean75", "mean100",
         "std50", "std100")


def align(xtrain, xval, mode):
    if mode == "none":
        return xval
    train_mean = xtrain.mean(0)
    val_mean = xval.mean(0)
    amount = int(mode[-2:]) / 100 if mode[-2:].isdigit() else 1
    if mode.startswith("mean"):
        return xval + amount * (train_mean - val_mean)
    train_std = np.maximum(xtrain.std(0), 1e-5)
    val_std = np.maximum(xval.std(0), 1e-5)
    standardized = (xval - val_mean) * train_std / val_std + train_mean
    return xval + amount * (standardized - xval)


def main():
    scores = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                 "other_hits": 0, "other_trials": 0})
    archive = next(Path("data").glob("*.zip"))
    with zipfile.ZipFile(archive) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(z, path, True)
                sessions.append((sid, x, y))
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            ids = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            feat = features(make_bank(x), BAND_SETS["all"], "full", "one_two")
            checks = [sessions[-1][0]]
            if len(sessions) >= 3:
                checks.append(sessions[1][0])
            for check_i, sid in enumerate(checks):
                val = ids == sid
                fitted = model("logreg03")
                fitted.fit(feat[~val], y[~val])
                for mode in MODES:
                    adjusted = align(feat[~val], feat[val], mode)
                    probabilities = fitted.predict_proba(adjusted)[:, 1]
                    pred = balanced_labels(probabilities, len(probabilities) // 2)
                    key = "recent" if check_i == 0 else "other"
                    scores[mode][key + "_hits"] += int(np.sum(pred == y[val]))
                    scores[mode][key + "_trials"] += len(pred)
            print(subject, flush=True)
    result = {mode: {**s, "recent_accuracy": s["recent_hits"] / s["recent_trials"],
                     "other_accuracy": s["other_hits"] / s["other_trials"]}
              for mode, s in scores.items()}
    Path("output").mkdir(exist_ok=True)
    Path("output/alignment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
