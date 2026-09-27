"""Evaluate nonlinear subject transfer against the validated V4 baseline.

No Kaggle submission is performed.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover

REFS = (("car", "logreg03"), ("laplacian", "lda"))


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            sessions = [(sid, *trials(archive, name, True)) for sid, name in sorted(trains[subject])]
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            feats = {ref: build_features(x, ref, "post_halves_ratio") for ref, _ in REFS}
            records[subject] = dict(y=y, groups=groups, feats=feats,
                                    sessions=sessions)
            print("features", subject, flush=True)
    scores = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                  "other_hits": 0, "other_trials": 0})
    for subject, record in records.items():
        splits = [("recent", record["sessions"][-1][0])]
        if len(record["sessions"]) >= 3:
            splits.append(("other", record["sessions"][1][0]))
        for split, sid in splits:
            holdout = record["groups"] == sid
            truth = record["y"][holdout]
            local_prob, cat_prob = [], []
            for ref, method in REFS:
                x = record["feats"][ref]
                local = model(method)
                local.fit(x[~holdout], record["y"][~holdout])
                local_prob.append(local.predict_proba(x[holdout])[:, 1])
                features, targets = [], []
                for other_subject, other in records.items():
                    other_x = other["feats"][ref]
                    train_mask = (~holdout if other_subject == subject else
                                  np.ones(len(other["y"]), dtype=bool))
                    scaler = StandardScaler().fit(other_x[train_mask])
                    features.append(scaler.transform(other_x[train_mask]))
                    targets.append(other["y"][train_mask])
                    if other_subject == subject:
                        valid_x = scaler.transform(x[holdout])
                cat = CatBoostClassifier(iterations=250, depth=4, learning_rate=.04,
                                         l2_leaf_reg=10, loss_function="Logloss",
                                         random_seed=42, thread_count=1, verbose=False)
                cat.fit(np.concatenate(features), np.concatenate(targets))
                cat_prob.append(cat.predict_proba(valid_x)[:, 1])
            base, pooled = np.mean(local_prob, axis=0), np.mean(cat_prob, axis=0)
            for weight in (0, .25, .5, .75, 1):
                key = f"cat_{weight:g}"
                probs = (1-weight)*base + weight*pooled
                pred = balanced_labels(probs, len(probs)//2)
                scores[key][split+"_hits"] += int(np.sum(pred == truth))
                scores[key][split+"_trials"] += len(truth)
        print("validated", subject, flush=True)
    results = [{"config": k, **v, "recent_accuracy": v["recent_hits"]/v["recent_trials"],
                "other_accuracy": v["other_hits"]/v["other_trials"]}
               for k, v in scores.items()]
    Path("output").mkdir(exist_ok=True)
    Path("output/cattransfer.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
