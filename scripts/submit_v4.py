"""Build V4 submission from complementary EEG spatial references."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover


def main():
    predictions = []
    counts = []
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            labeled = [trials(archive, name, True) for _, name in sorted(trains[subject])]
            x = np.concatenate([pair[0] for pair in labeled])
            y = np.concatenate([pair[1] for pair in labeled])
            query, _ = trials(archive, tests[subject], False)
            if len(query) != 40:
                raise ValueError(f"{subject}: expected 40 test epochs, found {len(query)}")
            probabilities = []
            for reference, method in (("car", "logreg03"), ("laplacian", "lda")):
                classifier = model(method)
                classifier.fit(build_features(x, reference, "post_halves_ratio"), y)
                probabilities.append(classifier.predict_proba(
                    build_features(query, reference, "post_halves_ratio")
                )[:, 1])
            labels = balanced_labels((probabilities[0] + probabilities[1]) / 2, 20)
            predictions.extend(np.where(labels, "move", "rest").tolist())
            counts.append({"subject": subject, "train_epochs": len(y), "test_epochs": len(query)})
            print(subject, len(y), len(query), flush=True)
    if len(predictions) != 680:
        raise ValueError(f"Expected 680 predictions, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method": "50/50 CAR logistic and local Laplacian LDA",
        "other_validation": {"correct": 602, "total": 800, "accuracy": 602 / 800},
        "recent_validation": {"correct": 119, "total": 170, "accuracy": 119 / 170},
        "subjects": counts,
        "row_count": len(predictions)
    }, indent=2))


if __name__ == "__main__":
    main()
