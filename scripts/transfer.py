"""Evaluate cross-subject transfer against V4 without submitting to Kaggle."""
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

REFERENCES = (("car", "logreg03"), ("laplacian", "lda"))
BLENDS = (0, .25, .5, .75, 1)


def score(p, truth):
    return int(np.sum(balanced_labels(p, len(p) // 2) == truth))


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            sessions = [(sid, *trials(archive, name, True)) for sid, name in sorted(trains[subject])]
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            query, _ = trials(archive, tests[subject], False)
            features = {ref: build_features(x, ref, "post_halves_ratio") for ref, _ in REFERENCES}
            query_features = {ref: build_features(query, ref, "post_halves_ratio") for ref, _ in REFERENCES}
            records[subject] = dict(y=y, groups=groups, features=features,
                                    query_features=query_features, sessions=sessions)
            print("features", subject, flush=True)

    results = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                   "other_hits": 0, "other_trials": 0})
    details = []
    for subject, item in records.items():
        valid_sessions = [("recent", item["sessions"][-1][0])]
        if len(item["sessions"]) >= 3:
            valid_sessions.append(("other", item["sessions"][1][0]))
        for split, sid in valid_sessions:
            mask = item["groups"] == sid
            local_probs = []
            transfer_probs = []
            for ref, method in REFERENCES:
                x = item["features"][ref]
                local = model(method)
                local.fit(x[~mask], item["y"][~mask])
                local_probs.append(local.predict_proba(x[mask])[:, 1])
                # Normalize every subject using its own labeled recordings.
                xs, ys = [], []
                for other_subject, other in records.items():
                    other_x = other["features"][ref]
                    use = ~mask if other_subject == subject else np.ones(len(other["y"]), bool)
                    scaler = StandardScaler().fit(other_x[use])
                    xs.append(scaler.transform(other_x[use]))
                    ys.append(other["y"][use])
                    if other_subject == subject:
                        query_x = scaler.transform(x[mask])
                classifier = LogisticRegression(C=.03, max_iter=1500)
                classifier.fit(np.concatenate(xs), np.concatenate(ys))
                transfer_probs.append(classifier.predict_proba(query_x)[:, 1])
            local_p = np.mean(local_probs, axis=0)
            pooled_p = np.mean(transfer_probs, axis=0)
            for w in BLENDS:
                p = (1-w)*local_p + w*pooled_p
                key = f"pooled_{w:g}"
                results[key][split+"_hits"] += score(p, item["y"][mask])
                results[key][split+"_trials"] += mask.sum()
            details.append({"subject": subject, "split": split, "local_hits": score(local_p, item["y"][mask]),
                            "pooled_hits": score(pooled_p, item["y"][mask])})
        print("validated", subject, flush=True)
    summary = [{"config": key, **v,
                "recent_accuracy": v["recent_hits"]/v["recent_trials"],
                "other_accuracy": v["other_hits"]/v["other_trials"]} for key, v in results.items()]
    Path("output").mkdir(exist_ok=True)
    Path("output/transfer.json").write_text(json.dumps({"summary":summary,"details":details},indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
