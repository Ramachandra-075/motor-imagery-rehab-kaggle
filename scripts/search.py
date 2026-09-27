"""Compare EEG feature windows and classifiers on separate recording sessions."""
import itertools
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train import FS, balanced_labels, discover, read_epochs

BANDS = [(4, 8), (8, 12), (12, 16), (16, 22), (22, 30), (30, 40)]
BAND_SETS = {
    "all": (0, 1, 2, 3, 4, 5),
    "motor": (1, 2, 3, 4),
    "mu_beta": (1, 3, 4),
}
WINDOWS = {"full": (0, 1000), "early": (0, 625), "late": (375, 1000),
           "middle": (125, 875)}
SLICES = {
    "one": ((0, 1),),
    "two": ((0, 0.5), (0.5, 1)),
    "one_two": ((0, 1), (0, 0.5), (0.5, 1)),
}
MODELS = {"logreg03": ("logreg", .03), "logreg3": ("logreg", .3),
          "lda": ("lda", 0), "svm3": ("svm", 3)}


def features(bank, indices, window, slices):
    start, stop = WINDOWS[window]
    rows = []
    for fraction_a, fraction_b in SLICES[slices]:
        lo = start + int((stop - start) * fraction_a)
        hi = start + int((stop - start) * fraction_b)
        for index in indices:
            variance = np.var(bank[index][:, :, lo:hi], axis=-1) + 1e-12
            power = np.log(variance)
            # A dimensionless C3/C4 feature for left/right motor activity.
            rows.append(power)
            rows.append((power[:, 1] - power[:, 3])[:, None])
    return np.hstack(rows)


def model(name):
    kind, param = MODELS[name]
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=param, max_iter=2000))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=param, gamma="scale", probability=False))
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def main():
    archive = next(Path("data").glob("*.zip"))
    summaries = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                    "other_hits": 0, "other_trials": 0})
    detail = {}
    with zipfile.ZipFile(archive) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sets = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(z, path, training=True)
                sets.append((sid, x, y))
            xs = np.concatenate([s[1] for s in sets])
            ys = np.concatenate([s[2] for s in sets])
            group = np.concatenate([np.full(len(s[2]), s[0]) for s in sets])
            bank = []
            for low, high in BANDS:
                sos = butter(4, [low, high], fs=FS, btype="bandpass", output="sos")
                bank.append(sosfiltfilt(sos, xs, axis=-1).astype(np.float32))
            validations = [sets[-1][0]]
            if len(sets) >= 3:
                validations.append(sets[1][0])
            subject_top = []
            for band_name, indices in BAND_SETS.items():
                for window, slices in itertools.product(WINDOWS, SLICES):
                    feat = features(bank, indices, window, slices)
                    for classifier in MODELS:
                        key = "/".join((band_name, window, slices, classifier))
                        for split_i, val_session in enumerate(validations):
                            is_val = group == val_session
                            if len(np.unique(ys[~is_val])) < 2:
                                continue
                            estimator = model(classifier)
                            estimator.fit(feat[~is_val], ys[~is_val])
                            if classifier.startswith("svm"):
                                proba = estimator.decision_function(feat[is_val])
                            else:
                                proba = estimator.predict_proba(feat[is_val])[:, 1]
                            pred = balanced_labels(proba, len(proba) // 2)
                            hits = int(np.sum(pred == ys[is_val]))
                            prefix = "recent" if split_i == 0 else "other"
                            summaries[key][prefix + "_hits"] += hits
                            summaries[key][prefix + "_trials"] += len(pred)
                        subject_top.append((key, summaries[key]["recent_hits"]))
            print(subject, "sessions", [s[0] for s in sets], "trial_count", len(ys), flush=True)

    result = []
    for key, val in summaries.items():
        rec = val["recent_hits"] / val["recent_trials"]
        oth = (val["other_hits"] / val["other_trials"]
               if val["other_trials"] else None)
        result.append({"config": key, "recent_accuracy": rec, "other_accuracy": oth, **val})
    result.sort(key=lambda r: (r["recent_accuracy"], r["other_accuracy"] or 0), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/search.json").write_text(json.dumps(result, indent=2))
    print("TOP 20", json.dumps(result[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
