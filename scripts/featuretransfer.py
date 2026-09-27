"""Validate interpretable feature subsets for pooled EEG transfer."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover

REFS = (("car", "logreg03"), ("laplacian", "lda"))
SELECTIONS = ("all", "ratio", "motor", "motor_ratio", "post")


def subset(x, ref, selection):
    if selection == "all":
        return x
    width = 8 if ref == "car" else 11
    indices = []
    for band in range(6):
        for part in range(4):
            if selection in ("ratio", "motor_ratio") and part != 1:
                continue
            if selection == "post" and part != 0:
                continue
            channels = (1, 2, 3) if selection.startswith("motor") else range(width)
            indices.extend(band*4*width + part*width + c for c in channels)
    return x[:, indices]


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            sessions = [(sid, *trials(archive, name, True)) for sid, name in sorted(trains[subject])]
            x = np.concatenate([s[1] for s in sessions])
            records[subject] = dict(
                y=np.concatenate([s[2] for s in sessions]),
                groups=np.concatenate([np.full(len(s[2]), s[0]) for s in sessions]),
                sessions=sessions,
                features={ref: build_features(x, ref, "post_halves_ratio") for ref, _ in REFS})
            print("features", subject, flush=True)
    totals = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                  "other_hits": 0, "other_trials": 0})
    for subject, record in records.items():
        splits = [("recent", record["sessions"][-1][0])]
        if len(record["sessions"]) >= 3:
            splits.append(("other", record["sessions"][1][0]))
        for split, sid in splits:
            valid = record["groups"] == sid
            baseline = []
            for ref, method in REFS:
                x = record["features"][ref]
                clf = model(method)
                clf.fit(x[~valid], record["y"][~valid])
                baseline.append(clf.predict_proba(x[valid])[:, 1])
            local = np.mean(baseline, axis=0)
            pooled = {}
            for selection in SELECTIONS:
                ps = []
                for ref, _ in REFS:
                    matrix, labels = [], []
                    for other_subject, other in records.items():
                        x = subset(other["features"][ref], ref, selection)
                        use = (~valid if other_subject == subject else
                               np.ones(len(other["y"]), dtype=bool))
                        scaler = StandardScaler().fit(x[use])
                        matrix.append(scaler.transform(x[use]))
                        labels.append(other["y"][use])
                        if other_subject == subject:
                            query = scaler.transform(x[valid])
                    clf = LogisticRegression(C=.03, max_iter=1500)
                    clf.fit(np.concatenate(matrix), np.concatenate(labels))
                    ps.append(clf.predict_proba(query)[:, 1])
                pooled[selection] = np.mean(ps, axis=0)
            for selection, probs in pooled.items():
                for weight in (.25, .5, .75):
                    blended = (1-weight)*local+weight*probs
                    pred = balanced_labels(blended, len(blended)//2)
                    key = f"{selection}_{weight:g}"
                    totals[key][split+"_hits"] += int(np.sum(pred == record["y"][valid]))
                    totals[key][split+"_trials"] += len(pred)
        print("validated", subject, flush=True)
    result = [{"config": key, **v,
               "recent_accuracy": v["recent_hits"]/v["recent_trials"],
               "other_accuracy": v["other_hits"]/v["other_trials"]}
              for key,v in totals.items()]
    result.sort(key=lambda row: (row["other_accuracy"], row["recent_accuracy"]), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/featuretransfer.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
