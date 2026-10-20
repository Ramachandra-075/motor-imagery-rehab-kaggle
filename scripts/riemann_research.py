"""Research a distinct filter-bank Riemannian tangent-space EEG model.

Uses the same per-subject stratified 5-fold protocol as the current grid.
It writes metrics only; candidate CSV selection and Kaggle submission are unchanged.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from baseline import trials
from train import balanced_labels, discover, FS

BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (8, 30))


def filter_bank(epochs):
    bank = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
        bank.append(sosfiltfilt(sos, epochs, axis=-1).astype(np.float64))
    return bank


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, _ = discover(archive)
        for subject in sorted(trains):
            sessions = [(sid, *trials(archive, path, True))
                        for sid, path in sorted(trains[subject])]
            x = np.concatenate([item[1] for item in sessions])
            y = np.concatenate([item[2] for item in sessions]).astype(int)
            records[subject] = (filter_bank(x), y)
            print("riemann features", subject, x.shape, flush=True)

    truth_all, pred_all = [], []
    subject_results = {}
    for subject, (banks, y) in records.items():
        oof = np.full(len(y), np.nan, dtype=float)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260927)
        for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
            band_probabilities = []
            for x in banks:
                # Covariance and tangent-space transforms are fitted inside each fold
                # to prevent validation data influencing learned transforms.
                cov = Covariances(estimator="lwf")
                train_cov = cov.fit_transform(x[tr])
                valid_cov = cov.transform(x[va])
                ts = TangentSpace(metric="riemann")
                train_ts = ts.fit_transform(train_cov, y[tr])
                valid_ts = ts.transform(valid_cov)
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
                )
                clf.fit(train_ts, y[tr])
                band_probabilities.append(clf.predict_proba(valid_ts)[:, 1])
            oof[va] = np.mean(band_probabilities, axis=0)
            print("riemann fold", subject, fold, flush=True)
        pred = balanced_labels(oof, int(np.sum(y == 1)))
        subject_results[str(subject)] = {
            "accuracy": float(np.mean(pred == y)),
            "correct": int(np.sum(pred == y)),
            "total": int(len(y)),
        }
        truth_all.extend(y.tolist())
        pred_all.extend(pred.astype(int).tolist())
        print("riemann subject result", subject, subject_results[str(subject)], flush=True)

    cm = confusion_matrix(truth_all, pred_all, labels=[0, 1])
    report = {
        "method": "filter-bank Riemannian covariance tangent space + logistic regression",
        "bands_hz": [list(x) for x in BANDS],
        "validation": "per-subject stratified 5-fold OOF; random_state=20260927",
        "correct": int(np.sum(np.asarray(truth_all) == np.asarray(pred_all))),
        "total": len(truth_all),
        "accuracy": float(np.mean(np.asarray(truth_all) == np.asarray(pred_all))),
        "labels": ["rest", "move"],
        "confusion_matrix": cm.tolist(),
        "subject_results": subject_results,
        "note": "Research candidate only. Model selected after comparing candidates may be optimistic; no Kaggle submission."
    }
    Path("output").mkdir(exist_ok=True)
    Path("output/riemann_research.json").write_text(json.dumps(report, indent=2))
    print("RIEMANN_RESEARCH_RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
