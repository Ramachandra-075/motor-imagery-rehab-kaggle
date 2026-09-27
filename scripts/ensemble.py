"""Evaluate ensembles of motor-band and slow-wave EEG models."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from train import balanced_labels, discover, read_epochs
from search import BAND_SETS as OLD_SETS, features as old_features, model
from search2 import filter_epochs, feature_matrix
from submit_v2 import make_bank

WEIGHTS = (0, .25, .5, .75, 1)
scores = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                              "other_hits": 0, "other_trials": 0})


def normalized(proba):
    std = np.std(proba) + 1e-8
    return (proba - np.mean(proba)) / std


def main():
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(z, path, True)
                sessions.append((sid, x, y))
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            f_old = old_features(make_bank(x), OLD_SETS["all"], "full", "one_two")
            f_slow = feature_matrix(filter_epochs(x), "all", "thirds", "eight")
            for split_i, sid in enumerate([sessions[-1][0], *([sessions[1][0]] if len(sessions) >= 3 else [])]):
                val = groups == sid
                a = model("logreg03")
                b = model("logreg3")
                a.fit(f_old[~val], y[~val])
                b.fit(f_slow[~val], y[~val])
                p1, p2 = a.predict_proba(f_old[val])[:, 1], b.predict_proba(f_slow[val])[:, 1]
                for weight in WEIGHTS:
                    for scale in ("direct", "normalized"):
                        one, two = (p1, p2) if scale == "direct" else (normalized(p1), normalized(p2))
                        p = (1-weight)*one + weight*two
                        pred = balanced_labels(p, len(p)//2)
                        key = f"{weight:.2f}/{scale}"
                        split = "recent" if split_i == 0 else "other"
                        scores[key][split + "_hits"] += int(np.sum(pred == y[val]))
                        scores[key][split + "_trials"] += len(pred)
            print(subject, flush=True)
    result = {key: {**s, "recent_accuracy": s["recent_hits"]/s["recent_trials"],
                    "other_accuracy": s["other_hits"]/s["other_trials"]}
              for key, s in scores.items()}
    Path("output").mkdir(exist_ok=True)
    Path("output/ensemble.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
