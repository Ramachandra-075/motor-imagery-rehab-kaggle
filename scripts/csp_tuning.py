"""Validate CSP component counts and classifier regularization on held-out sessions.

Research only: never submits to Kaggle.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import discover, read_epochs, filter_bank, fit_csp, csp_features, balanced_labels

COMPONENTS = (2, 4, 6, 8)
CLASSIFIERS = ("lda_auto", "logreg_003", "logreg_03")


def classifier(name):
    if name == "lda_auto":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    c = 0.03 if name == "logreg_003" else 0.3
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def main():
    counts = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, _ = discover(archive)
        for subject in sorted(trains):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(archive, path, training=True)
                sessions.append((sid, filter_bank(x), y))
            if len(sessions) < 2:
                continue
            train_bank = [np.concatenate([s[1][b] for s in sessions[:-1]]) for b in range(3)]
            train_y = np.concatenate([s[2] for s in sessions[:-1]])
            valid_bank, valid_y = sessions[-1][1], sessions[-1][2]
            for ncomp in COMPONENTS:
                spatial = [fit_csp(band, train_y, n_components=ncomp) for band in train_bank]
                xtr = csp_features(train_bank, spatial)
                xva = csp_features(valid_bank, spatial)
                for clf_name in CLASSIFIERS:
                    clf = classifier(clf_name)
                    clf.fit(xtr, train_y)
                    p = clf.predict_proba(xva)[:, list(clf.classes_).index(1)]
                    pred = balanced_labels(p, len(valid_y)//2)
                    key = f"csp_{ncomp}/{clf_name}"
                    row = counts.setdefault(key, {"correct": 0, "total": 0, "subjects": 0})
                    row["correct"] += int((pred == valid_y).sum())
                    row["total"] += len(valid_y)
                    row["subjects"] += 1
            print("CSP experiment", subject, flush=True)
    results = []
    for key, row in counts.items():
        results.append({"config": key, **row, "accuracy": row["correct"]/row["total"]})
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    report = {
        "validation": "last recording session held out per subject; class-balanced predictions",
        "warning": "Exploratory model selection on held-out sessions; not an untouched final estimate.",
        "results": results,
        "best_candidate": results[0] if results else None,
        "target_accuracy": 0.90,
        "target_reached": bool(results and results[0]["accuracy"] > 0.90),
    }
    Path("output").mkdir(exist_ok=True)
    Path("output/csp_tuning.json").write_text(json.dumps(report, indent=2))
    print("CSP_TUNING_RESULTS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
