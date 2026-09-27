"""Second EEG search: slow cortical potentials, broad power, time bins."""
import itertools
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from train import FS, balanced_labels, discover, read_epochs
from search import model

BANDS = [(0.5, 3), (3, 7), (7, 13), (13, 22), (22, 35), (35, 60)]
BAND_SETS = {
    "slow": (0, 1),
    "slow_mu": (0, 1, 2, 3),
    "all": (0, 1, 2, 3, 4, 5),
    "motor": (2, 3, 4),
}
MODES = ("whole", "halves", "whole_halves", "thirds")
ERP = ("none", "four", "eight")
CLASSIFIERS = ("logreg03", "logreg3", "lda")


def filter_epochs(x):
    result = []
    for low, high in BANDS:
        sos = butter(4, [low, high], fs=FS, btype="bandpass", output="sos")
        result.append(sosfiltfilt(sos, x, axis=-1).astype(np.float32))
    return result


def feature_matrix(bank, band_set, mode, erp):
    selections = {"whole": [(0, 1000)],
                  "halves": [(0, 500), (500, 1000)],
                  "whole_halves": [(0, 1000), (0, 500), (500, 1000)],
                  "thirds": [(0, 333), (333, 666), (666, 1000)]}[mode]
    parts = []
    if band_set:
        for start, stop in selections:
            for band in BAND_SETS[band_set]:
                power = np.log(np.var(bank_array := bank[band][:, :, start:stop], axis=-1) + 1e-12)
                parts.append(power)
                parts.append((power[:, 1] - power[:, 3])[:, None])
    if erp != "none":
        bins = 4 if erp == "four" else 8
        # The signed slow waveform contains attempted-movement potentials.
        slow = bank[0]
        for i in range(bins):
            mean = slow[:, :, i * 1000 // bins:(i+1) * 1000 // bins].mean(axis=-1)
            parts.append(mean)
    return np.hstack(parts)


def main():
    archive = next(Path("data").glob("*.zip"))
    scores = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                 "other_hits": 0, "other_trials": 0})
    with zipfile.ZipFile(archive) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(z, path, True)
                sessions.append((sid, x, y))
            xs = np.concatenate([s[1] for s in sessions])
            ys = np.concatenate([s[2] for s in sessions])
            group = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            bank = filter_epochs(xs)
            splits = [sessions[-1][0]]
            if len(sessions) >= 3:
                splits.append(sessions[1][0])
            for bands, mode, erp in itertools.product((*BAND_SETS, ""), MODES, ERP):
                if bands == "" and erp == "none":
                    continue
                if bands == "" and mode != "whole":
                    continue
                feat = feature_matrix(bank, bands, mode, erp)
                for classifier in CLASSIFIERS:
                    key = "/".join((bands or "erp_only", mode, erp, classifier))
                    for split_num, val_sid in enumerate(splits):
                        mask = group == val_sid
                        estimator = model(classifier)
                        estimator.fit(feat[~mask], ys[~mask])
                        probabilities = estimator.predict_proba(feat[mask])[:, 1]
                        pred = balanced_labels(probabilities, len(probabilities) // 2)
                        k = "recent" if split_num == 0 else "other"
                        scores[key][k + "_hits"] += int(np.sum(pred == ys[mask]))
                        scores[key][k + "_trials"] += len(pred)
            print(subject, "complete", flush=True)

    results = []
    for key, s in scores.items():
        results.append({"config": key, **s,
                        "recent_accuracy": s["recent_hits"] / s["recent_trials"],
                        "other_accuracy": s["other_hits"] / s["other_trials"]})
    results.sort(key=lambda r: (r["other_accuracy"], r["recent_accuracy"]), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/search2.json").write_text(json.dumps(results, indent=2))
    print("TOP OTHER", json.dumps(results[:20], indent=2), flush=True)
    print("TOP RECENT", json.dumps(sorted(results, key=lambda r: r["recent_accuracy"], reverse=True)[:10], indent=2), flush=True)


if __name__ == "__main__":
    main()
