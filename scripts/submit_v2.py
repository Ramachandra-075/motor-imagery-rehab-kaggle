"""Fit the selected EEG band-power model on all training sessions."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from train import FS, balanced_labels, discover, read_epochs
from search import BANDS, BAND_SETS, features, model

CONFIG = "all/full/one_two/logreg03"


def make_bank(epochs):
    bank = []
    for low, high in BANDS:
        sos = butter(4, [low, high], fs=FS, btype="bandpass", output="sos")
        bank.append(sosfiltfilt(sos, epochs, axis=-1).astype(np.float32))
    return bank


def main():
    archive_path = next(Path("data").glob("*.zip"))
    predictions = []
    detail = []
    with zipfile.ZipFile(archive_path) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            train_epochs, labels = [], []
            for _, filename in sorted(trains[subject]):
                x, y = read_epochs(archive, filename, True)
                train_epochs.append(x)
                labels.append(y)
            x_train = np.concatenate(train_epochs)
            y_train = np.concatenate(labels)
            x_test, _ = read_epochs(archive, tests[subject], False)
            if len(x_test) != 40:
                raise ValueError(f"{subject}: expected 40 test trials, found {len(x_test)}")
            selected = BAND_SETS["all"]
            train_feat = features(make_bank(x_train), selected, "full", "one_two")
            test_feat = features(make_bank(x_test), selected, "full", "one_two")
            estimator = model("logreg03")
            estimator.fit(train_feat, y_train)
            probabilities = estimator.predict_proba(test_feat)[:, 1]
            pred = balanced_labels(probabilities, 20)
            predictions.extend(np.where(pred, "move", "rest"))
            detail.append({"subject": subject, "training_trials": len(y_train),
                           "test_trials": len(x_test), "predicted_move": int(pred.sum())})
            print(subject, "trained", len(y_train), "test", len(x_test), flush=True)

    if len(predictions) != 680:
        raise ValueError(f"Expected 680 predictions, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method": CONFIG,
        "recent_validation_accuracy": 123 / 170,
        "recent_validation_trials": 170,
        "other_validation_accuracy": 528 / 800,
        "other_validation_trials": 800,
        "subject_details": detail,
        "note": "Local validation; Kaggle public score must be checked separately.",
    }, indent=2))
    print("WROTE", len(predictions), "rows", flush=True)


if __name__ == "__main__":
    main()
