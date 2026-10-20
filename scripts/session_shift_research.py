"""Session-shift research: compare baseline vs unsupervised CORAL alignment.

Validation holds out complete recording sessions (closer to the Kaggle test setup
than random folds). CORAL uses only unlabeled held-out-session features. No Kaggle
submission is performed.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from baseline import build_features, trials
from nonlinear_v9 import REFS, rich_features
from train import balanced_labels, discover


def coral_align(source, target, eps=1e-5):
    """Map target covariance to source covariance using unlabeled target data."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sm, tm = source.mean(0), target.mean(0)
    sc = np.cov(source - sm, rowvar=False) + eps * np.eye(source.shape[1])
    tc = np.cov(target - tm, rowvar=False) + eps * np.eye(target.shape[1])
    # Symmetric matrix powers avoid complex values from tiny numerical errors.
    def power(cov, exponent):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, eps)
        return (vecs * vals**exponent) @ vecs.T
    transform = power(tc, -0.5) @ power(sc, 0.5)
    return (target - tm) @ transform + sm


def eval_one(train_x, train_y, valid_x, valid_y, do_coral):
    scaler = StandardScaler().fit(train_x)
    tr = scaler.transform(train_x)
    va = scaler.transform(valid_x)
    if do_coral:
        va = coral_align(tr, va)
    clf = LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced")
    clf.fit(tr, train_y)
    prob = clf.predict_proba(va)[:, list(clf.classes_).index(1)]
    pred = balanced_labels(prob, len(valid_y) // 2)
    return {
        "accuracy": float(accuracy_score(valid_y, pred)),
        "auc": float(roc_auc_score(valid_y, prob)),
        "correct": int(np.sum(valid_y == pred)),
        "total": int(len(valid_y)),
    }


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, _ = discover(archive)
        for subject in sorted(trains):
            sessions = [(sid, *trials(archive, path, True))
                        for sid, path in sorted(trains[subject])]
            record = {"sessions": sessions, "y": np.concatenate([s[2] for s in sessions])}
            for family in ("old", "rich"):
                record[family] = {}
                for ref, _ in REFS:
                    record[family][ref] = [
                        build_features(x, ref, "post_halves_ratio") if family == "old"
                        else rich_features(x, ref)
                        for _, x, _ in sessions
                    ]
            records[subject] = record
            print("prepared subject", subject, flush=True)

    # Test last, second, and earliest sessions separately. Aggregate per trial.
    results = defaultdict(lambda: {"correct": 0, "total": 0, "auc_weighted": []})
    per_subject = {}
    for subject, record in records.items():
        sessions = record["sessions"]
        if len(sessions) < 3:
            hold_indices = [len(sessions) - 1, 0]
        else:
            hold_indices = [len(sessions) - 1, 1, 0]
        per_subject[str(subject)] = {}
        for hold_idx in dict.fromkeys(hold_indices):
            sid, _, valid_y = sessions[hold_idx]
            train_indices = [i for i in range(len(sessions)) if i != hold_idx]
            for family in ("old", "rich"):
                for ref, _ in REFS:
                    Xs = record[family][ref]
                    train_x = np.concatenate([Xs[i] for i in train_indices])
                    train_y = np.concatenate([sessions[i][2] for i in train_indices])
                    valid_x = Xs[hold_idx]
                    for aligned in (False, True):
                        name = f"{family}_{ref}_{'coral' if aligned else 'baseline'}"
                        score = eval_one(train_x, train_y, valid_x, valid_y, aligned)
                        key = f"{name}_{'recent' if hold_idx == len(sessions)-1 else ('earliest' if hold_idx == 0 else 'middle')}"
                        results[key]["correct"] += score["correct"]
                        results[key]["total"] += score["total"]
                        results[key]["auc_weighted"].append((score["auc"], score["total"]))
                        per_subject[str(subject)].setdefault(str(sid), {})[name] = score
            print("evaluated subject/session", subject, sid, flush=True)

    summary = {}
    for key, val in results.items():
        summary[key] = {
            "correct": val["correct"], "total": val["total"],
            "accuracy": val["correct"] / val["total"],
            "weighted_auc": float(np.average([x[0] for x in val["auc_weighted"]],
                                             weights=[x[1] for x in val["auc_weighted"]])),
        }
    # Select best by the average of recent/middle/earliest held-out accuracy,
    # not by random-fold CV. Candidate scores are diagnostics, not Kaggle scores.
    families = sorted({k.rsplit("_", 1)[0] for k in summary})
    family_summary = {}
    for family in families:
        parts = [summary[k]["accuracy"] for k in summary
                 if k.startswith(family + "_")]
        if parts:
            family_summary[family] = {"mean_heldout_accuracy": float(np.mean(parts)),
                                      "split_scores": {k: v["accuracy"] for k, v in summary.items()
                                                       if k.startswith(family + "_")}}
    report = {
        "method": "whole-session holdout; per-subject; baseline logistic vs unsupervised CORAL alignment",
        "validation_note": "This is a local validation score, not a Kaggle leaderboard score. Held-out features are unlabeled for CORAL.",
        "summary": summary, "family_summary": family_summary, "per_subject": per_subject,
    }
    Path("output").mkdir(exist_ok=True)
    Path("output/session_shift_research.json").write_text(json.dumps(report, indent=2))
    print("SESSION_SHIFT_SUMMARY", json.dumps(family_summary), flush=True)


if __name__ == "__main__":
    main()
