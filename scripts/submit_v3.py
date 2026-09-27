"""Create V3 EEG submission with pre-cue reference and subject models."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover


def main():
    archive_path = next(Path("data").glob("*.zip"))
    predictions = []
    counts = []
    with zipfile.ZipFile(archive_path) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            labeled = [trials(z, name, True) for _, name in sorted(trains[subject])]
            x = np.concatenate([pair[0] for pair in labeled])
            y = np.concatenate([pair[1] for pair in labeled])
            query, _ = trials(z, tests[subject], False)
            if len(query) != 40:
                raise ValueError(f"{subject}: found {len(query)} test epochs, expected 40")
            train_features = build_features(x, "car", "post_halves_ratio")
            test_features = build_features(query, "car", "post_halves_ratio")
            classifier = model("logreg03")
            classifier.fit(train_features, y)
            probability = classifier.predict_proba(test_features)[:, 1]
            labels = balanced_labels(probability, 20)
            predictions.extend(np.where(labels, "move", "rest").tolist())
            counts.append({"subject": subject, "train_epochs": len(y),
                           "test_epochs": len(query)})
            print(subject, len(y), len(query), flush=True)
    if len(predictions) != 680:
        raise ValueError(f"Expected 680 predictions, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method": "car/post_halves_ratio/logreg03",
        "recent_validation": {"correct": 121, "total": 170, "accuracy": 121 / 170},
        "other_validation": {"correct": 590, "total": 800, "accuracy": 590 / 800},
        "subjects": counts,
        "row_count": len(predictions),
    }, indent=2))


if __name__ == "__main__":
    main()
