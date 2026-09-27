"""Create the V5 cross-subject EEG submission. Does not submit to Kaggle."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
            labeled = [trials(archive, path, True) for _, path in sorted(trains[subject])]
            x = np.concatenate([pair[0] for pair in labeled])
            y = np.concatenate([pair[1] for pair in labeled])
            query, _ = trials(archive, tests[subject], False)
            if len(query) != 40:
                raise ValueError(f"{subject}: expected 40 test epochs, found {len(query)}")
            records[subject] = dict(y=y, train={
                ref: build_features(x, ref, "post_halves_ratio") for ref, _ in REFS
            }, test={
                ref: build_features(query, ref, "post_halves_ratio") for ref, _ in REFS
            })
            print("features", subject, flush=True)
    pooled_probs = {subject: [] for subject in records}
    for ref, _ in REFS:
        pooled_x, pooled_y, queries = [], [], {}
        for subject, record in records.items():
            scaler = StandardScaler().fit(record["train"][ref])
            pooled_x.append(scaler.transform(record["train"][ref]))
            pooled_y.append(record["y"])
            queries[subject] = scaler.transform(record["test"][ref])
        clf = LogisticRegression(C=.03, max_iter=1500)
        clf.fit(np.concatenate(pooled_x), np.concatenate(pooled_y))
        for subject in records:
            pooled_probs[subject].append(clf.predict_proba(queries[subject])[:, 1])
    predictions, details = [], []
    for subject, record in records.items():
        local_probs = []
        for ref, method in REFS:
            clf = model(method)
            clf.fit(record["train"][ref], record["y"])
            local_probs.append(clf.predict_proba(record["test"][ref])[:, 1])
        probs = .5*np.mean(local_probs, axis=0)+.5*np.mean(pooled_probs[subject], axis=0)
        labels = balanced_labels(probs, 20)
        predictions.extend(np.where(labels, "move", "rest").tolist())
        details.append({"subject":subject,"train_epochs":len(record["y"]),
                        "test_epochs":len(labels),"move_predictions":int(labels.sum())})
    if len(predictions) != 680:
        raise ValueError(f"Expected 680 rows, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method":"50% V4 local spatial blend + 50% pooled logistic spatial blend",
        "recent_validation":{"correct":123,"total":170,"accuracy":123/170},
        "other_validation":{"correct":616,"total":800,"accuracy":616/800},
        "kaggle_score":"unsubmitted",
        "subjects":details
    },indent=2))


if __name__ == "__main__":
    main()
