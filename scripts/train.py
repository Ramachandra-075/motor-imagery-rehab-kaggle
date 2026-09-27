"""Single-subject EEG decoding with session-held-out validation.

The competition ZIP is read directly. Models use only EEG channels, never event
order as a predictor. The exact 680-row submission is assembled in subject order.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SLUG = "low-cost-motor-imagery-decoding-for-rehab"
CHANNELS = ["Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz"]
BANDS = [(8, 12), (12, 20), (20, 30)]
FS = 250
START = int(0.5 * FS)
LENGTH = int(4 * FS)


def read_epochs(archive: zipfile.ZipFile, name: str, training: bool):
    with archive.open(name) as stream:
        frame = pd.read_csv(stream, low_memory=False)
    if list(frame.columns) != ["time", *CHANNELS, "Marker_val"]:
        raise ValueError(f"Unexpected columns in {name}: {frame.columns.tolist()}")
    dt = np.median(np.diff(frame["time"].to_numpy(dtype=float)))
    if not 0.0037 < dt < 0.0043:
        raise ValueError(f"Expected 250 Hz for {name}; observed time step {dt}")
    labels = frame["Marker_val"].fillna("").astype(str).to_numpy()
    indices = np.flatnonzero(np.isin(labels, ["move", "rest"] if training else ["cue_start"]))
    raw = frame[CHANNELS].to_numpy(dtype=np.float64).T
    result, targets = [], []
    for idx in indices:
        window = raw[:, idx + START:idx + START + LENGTH]
        if window.shape != (len(CHANNELS), LENGTH):
            raise ValueError(f"Short EEG epoch at {name}, row {idx}")
        # Cleaning is per epoch, so validation and test statistics cannot leak.
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        window -= np.median(window, axis=1, keepdims=True)
        result.append(window)
        if training:
            targets.append(1 if labels[idx] == "move" else 0)
    if not result:
        raise ValueError(f"No trial markers in {name}")
    return np.stack(result).astype(np.float32), np.asarray(targets, dtype=int)


def filter_bank(epochs):
    out = []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype="bandpass", fs=FS, output="sos")
        out.append(sosfiltfilt(sos, epochs, axis=-1).astype(np.float32))
    return out


def band_features(filtered):
    features = []
    for band in filtered:
        variance = np.var(band, axis=-1) + 1e-12
        features.append(np.log(variance))
        # Contralateral asymmetry: motor cortex C3 versus C4.
        features.append(np.log(variance[:, 1] / variance[:, 3])[:, None])
    return np.hstack(features)


def fit_csp(x, y, n_components=4):
    """Regularized CSP spatial filters, fit exclusively on training epochs."""
    covs = []
    for epoch in x:
        cov = epoch @ epoch.T / epoch.shape[1]
        cov /= np.trace(cov) + 1e-12
        covs.append(cov)
    covs = np.asarray(covs)
    a, b = covs[y == 0].mean(0), covs[y == 1].mean(0)
    reg = 0.05 * np.trace(a + b) / len(CHANNELS)
    eigvals, vectors = eigh(a, a + b + reg * np.eye(len(CHANNELS)))
    order = list(range(n_components // 2)) + list(range(-n_components // 2, 0))
    return vectors[:, order].T


def csp_features(filtered, spatial):
    feats = []
    for band, projection in zip(filtered, spatial):
        variance = np.var(np.einsum("kc,nct->nkt", projection, band), axis=-1) + 1e-12
        feats.append(np.log(variance / variance.sum(axis=1, keepdims=True)))
    return np.hstack(feats)


def predict(train_bank, train_y, query_bank, method):
    if method == "bandpower":
        x, query = band_features(train_bank), band_features(query_bank)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=2000))
    elif method == "csp":
        spatial = [fit_csp(band, train_y) for band in train_bank]
        x, query = csp_features(train_bank, spatial), csp_features(query_bank, spatial)
        classifier = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    else:
        raise ValueError(method)
    classifier.fit(x, train_y)
    return classifier.predict_proba(query)[:, list(classifier.classes_).index(1)]


def discover(archive):
    trains, tests = defaultdict(list), {}
    for name in archive.namelist():
        base = Path(name).name
        train_match = re.fullmatch(r"(S\d+)-(\d+)_eeg\.csv", base)
        test_match = re.fullmatch(r"(S\d+)_test\.csv", base)
        if train_match:
            trains[train_match.group(1)].append((int(train_match.group(2)), name))
        if test_match:
            tests[test_match.group(1)] = name
    assert len(tests) == 17, f"Expected 17 subject test sets, got {len(tests)}"
    return trains, tests


def main():
    archive_path = next(Path("data").glob("*.zip"))
    Path("output").mkdir(exist_ok=True)
    scores, predictions = [], []
    with zipfile.ZipFile(archive_path) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            sessions = []
            for session_id, filename in sorted(trains[subject]):
                x, y = read_epochs(archive, filename, training=True)
                sessions.append((session_id, filter_bank(x), y))
            test_x, _ = read_epochs(archive, tests[subject], training=False)
            if len(test_x) != 40:
                raise ValueError(f"{subject} expected 40 test trials, got {len(test_x)}")
            test_bank = filter_bank(test_x)

            # Hold out the last recording session. No trials from it enter fitting.
            train_sessions, valid = sessions[:-1], sessions[-1]
            if not train_sessions:
                raise ValueError(f"Need multiple sessions for {subject}")
            train_bank = [np.concatenate([s[1][i] for s in train_sessions]) for i in range(len(BANDS))]
            train_y = np.concatenate([s[2] for s in train_sessions])
            val_bank, val_y = valid[1], valid[2]
            if len(np.unique(train_y)) != 2:
                raise ValueError(f"Training set lacks both classes for {subject}")
            val_predictions = {}
            for method in ("bandpower", "csp"):
                val_predictions[method] = predict(train_bank, train_y, val_bank, method)
            val_predictions["blend"] = 0.35 * val_predictions["bandpower"] + 0.65 * val_predictions["csp"]
            accuracy = {key: float(np.mean((prob >= 0.5) == val_y))
                        for key, prob in val_predictions.items()}
            scores.append({"subject": subject, "validation_session": valid[0],
                           "validation_trials": len(val_y), "accuracy": accuracy})
            full_bank = [np.concatenate([s[1][i] for s in sessions]) for i in range(len(BANDS))]
            full_y = np.concatenate([s[2] for s in sessions])
            probs = {method: predict(full_bank, full_y, test_bank, method)
                     for method in ("bandpower", "csp")}
            predictions.extend((0.35 * probs["bandpower"] + 0.65 * probs["csp"] >= 0.5)
                               .astype(int).tolist())
            print(subject, len(full_y), len(val_y), accuracy, flush=True)

    if len(predictions) != 680:
        raise ValueError(f"Expected 680 predictions, got {len(predictions)}")
    submission = pd.DataFrame({"ID": np.arange(680, dtype=int),
                               "TARGET": np.where(np.asarray(predictions) == 1, "move", "rest")})
    assert submission.columns.tolist() == ["ID", "TARGET"]
    assert submission["TARGET"].isin(["move", "rest"]).all()
    submission.to_csv("output/submission.csv", index=False)
    total = sum(x["validation_trials"] for x in scores)
    overall = {name: sum(x["accuracy"][name] * x["validation_trials"] for x in scores) / total
               for name in ("bandpower", "csp", "blend")}
    report = {"validation": "last recording session held out per subject",
              "validation_trials": total, "overall_accuracy": overall,
              "subjects": scores, "submission_rows": len(submission),
              "submission_class_counts": submission["TARGET"].value_counts().to_dict()}
    Path("output/report.json").write_text(json.dumps(report, indent=2))
    print("RESULT:", json.dumps({k: v for k, v in report.items() if k != "subjects"}), flush=True)


if __name__ == "__main__":
    main()
