"""Test small spatial-reference ensembles against both held-out sessions."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover

CANDIDATES = [
    ("car", "post_halves_ratio", "logreg03"),
    ("laplacian", "post_halves_ratio", "lda"),
    ("car", "post", "logreg03"),
    ("car", "post_halves_ratio", "logreg3"),
]
WEIGHTS = {
    "base": (1, 0, 0, 0),
    "base_lap25": (.75, .25, 0, 0),
    "base_lap50": (.5, .5, 0, 0),
    "base_post25": (.75, 0, .25, 0),
    "base_post50": (.5, 0, .5, 0),
    "base_three": (.5, .25, .25, 0),
    "all_four": (.4, .2, .2, .2),
    "base_Cblend": (.5, 0, 0, .5),
}


def normalize(p):
    return (p - p.mean()) / (p.std() + 1e-8)


def main():
    scores = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                  "other_hits": 0, "other_trials": 0})
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, name in sorted(trains[subject]):
                x, y = trials(z, name, True)
                sessions.append((sid, x, y))
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            matrices = {}
            for ref, feat, _ in CANDIDATES:
                matrices[(ref, feat)] = build_features(x, ref, feat)
            splits = [sessions[-1][0]]
            if len(sessions) >= 3:
                splits.append(sessions[1][0])
            for idx, sid in enumerate(splits):
                validation = groups == sid
                predictions = []
                for ref, feat, method in CANDIDATES:
                    matrix = matrices[(ref, feat)]
                    classifier = model(method)
                    classifier.fit(matrix[~validation], y[~validation])
                    predictions.append(classifier.predict_proba(matrix[validation])[:, 1])
                for name, weights in WEIGHTS.items():
                    for style in ("direct", "scaled"):
                        probs = [normalize(p) for p in predictions] if style == "scaled" else predictions
                        blend = sum(w*p for w, p in zip(weights, probs))
                        labels = balanced_labels(blend, len(blend)//2)
                        key = name + "/" + style
                        split = "recent" if idx == 0 else "other"
                        scores[key][split+"_hits"] += int(np.sum(labels == y[validation]))
                        scores[key][split+"_trials"] += len(labels)
            print(subject, flush=True)
    result = []
    for name, s in scores.items():
        result.append({"config": name, **s,
                       "recent_accuracy": s["recent_hits"]/s["recent_trials"],
                       "other_accuracy": s["other_hits"]/s["other_trials"]})
    result.sort(key=lambda r: (r["other_accuracy"], r["recent_accuracy"]), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/spatialensemble.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
