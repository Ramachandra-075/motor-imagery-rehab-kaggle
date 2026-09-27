"""Compare local feature selection and nonlinear margins with pooled transfer."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover

REFS = (("car", "logreg03"), ("laplacian", "lda"))


def candidate(name):
    if name == "rbf":
        return make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", probability=True,
                                                    random_state=42))
    if name == "select30":
        return make_pipeline(SelectKBest(f_classif, k=30), StandardScaler(),
                             LogisticRegression(C=.03, max_iter=2000))
    if name == "select60":
        return make_pipeline(SelectKBest(f_classif, k=60), StandardScaler(),
                             LogisticRegression(C=.03, max_iter=2000))
    raise ValueError(name)


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, tests = discover(archive)
        for subject in sorted(tests):
            sessions = [(sid, *trials(archive, name, True)) for sid, name in sorted(trains[subject])]
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            records[subject] = dict(y=y, groups=groups, sessions=sessions,
                                    features={ref: build_features(x, ref, "post_halves_ratio")
                                              for ref, _ in REFS})
            print("features", subject, flush=True)
    totals = defaultdict(lambda: {"recent_hits": 0, "recent_trials": 0,
                                  "other_hits": 0, "other_trials": 0})
    for subject, record in records.items():
        splits = [("recent", record["sessions"][-1][0])]
        if len(record["sessions"]) >= 3:
            splits.append(("other", record["sessions"][1][0]))
        for split, sid in splits:
            val = record["groups"] == sid
            reference_probs = []
            pools = []
            others = {name: [] for name in ("rbf", "select30", "select60")}
            for ref, method in REFS:
                x, y = record["features"][ref], record["y"]
                clf = model(method)
                clf.fit(x[~val], y[~val])
                reference_probs.append(clf.predict_proba(x[val])[:, 1])
                for name in others:
                    clf = candidate(name)
                    clf.fit(x[~val], y[~val])
                    others[name].append(clf.predict_proba(x[val])[:, 1])
                pooled_x, pooled_y = [], []
                for other_subject, other in records.items():
                    other_x = other["features"][ref]
                    use = ~val if other_subject == subject else np.ones(len(other["y"]), bool)
                    scaler = StandardScaler().fit(other_x[use])
                    pooled_x.append(scaler.transform(other_x[use]))
                    pooled_y.append(other["y"][use])
                    if other_subject == subject:
                        query = scaler.transform(x[val])
                global_model = LogisticRegression(C=.03, max_iter=1500)
                global_model.fit(np.concatenate(pooled_x), np.concatenate(pooled_y))
                pools.append(global_model.predict_proba(query)[:, 1])
            base = np.mean(reference_probs, axis=0)
            pool = np.mean(pools, axis=0)
            probabilities = {"v4": base, "transfer50": .5*base+.5*pool}
            for name, prob_list in others.items():
                p = np.mean(prob_list, axis=0)
                probabilities[name] = p
                probabilities["transfer25_"+name] = .375*base+.375*pool+.25*p
                probabilities["transfer50_"+name] = .25*base+.25*pool+.5*p
            for name, probs in probabilities.items():
                result = balanced_labels(probs, len(probs)//2)
                totals[name][split+"_hits"] += int(np.sum(result == record["y"][val]))
                totals[name][split+"_trials"] += len(result)
        print("validated", subject, flush=True)
    summary = [{"config": name, **counts,
                "recent_accuracy": counts["recent_hits"]/counts["recent_trials"],
                "other_accuracy": counts["other_hits"]/counts["other_trials"]}
               for name, counts in totals.items()]
    summary.sort(key=lambda r: (r["other_accuracy"], r["recent_accuracy"]), reverse=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/localmodels.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
