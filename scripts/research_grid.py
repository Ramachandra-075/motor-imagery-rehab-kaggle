"""Grid-search distinct EEG model hyperparameters with subject-wise five-fold OOF validation.

This script only trains/evaluates and writes research_grid.json. It never submits to Kaggle.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from baseline import trials
from nonlinear_v9 import REFS, probabilities, rich_features
from train import balanced_labels, discover

RICH_C_VALUES = (0.01, 0.03, 0.1)
CATBOOST_CONFIGS = (
    {"name": "shallow", "iterations": 180, "depth": 3, "learning_rate": 0.04, "l2_leaf_reg": 10},
    {"name": "standard", "iterations": 220, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 10},
    {"name": "deeper_regularized", "iterations": 300, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 15},
)
BLEND_WEIGHTS = (0.1, 0.2, 0.3)


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, _ = discover(archive)
        for subject in sorted(trains):
            sessions = [(sid, *trials(archive, path, True))
                        for sid, path in sorted(trains[subject])]
            x = np.concatenate([s[1] for s in sessions])
            records[subject] = {
                "y": np.concatenate([s[2] for s in sessions]),
                "sessions": sessions,
                "old": {ref: None for ref, _ in REFS},
                "rich": {ref: rich_features(x, ref) for ref, _ in REFS},
            }
            print("grid features", subject, flush=True)

    # Every trial receives predictions only from a fold that did not train on it.
    results = []
    total_runs = len(RICH_C_VALUES) * len(CATBOOST_CONFIGS)
    run_number = 0
    for rich_c in RICH_C_VALUES:
        for cat_cfg in CATBOOST_CONFIGS:
            run_number += 1
            params = {k: v for k, v in cat_cfg.items() if k != "name"}
            candidate_oof = {weight: {"truth": [], "pred": []} for weight in BLEND_WEIGHTS}
            for subject, record in records.items():
                y = record["y"]
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260927)
                subject_truth = []
                subject_probs = []
                for train_ix, valid_ix in splitter.split(np.zeros(len(y)), y):
                    out = probabilities(records, subject, train_ix, valid_ix,
                                        rich_c=rich_c, cat_params=params)
                    subject_truth.extend(y[valid_ix].astype(int).tolist())
                    subject_probs.extend((out["v6"], out["cat"]))
                truth = np.asarray(subject_truth, dtype=int)
                v6 = subject_probs[0::2]
                cat = subject_probs[1::2]
                # Fold-specific class balancing is applied below using the exact same
                # deterministic five-fold splits as the validation baseline.
                fold_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260927).split(
                    np.zeros(len(y)), y))
                for weight in BLEND_WEIGHTS:
                    pred_parts = []
                    truth_parts = []
                    offset = 0
                    for _, valid_ix in fold_splits:
                        n = len(valid_ix)
                        probs = (1.0 - weight) * np.asarray(v6[offset:offset+n]) + weight * np.asarray(cat[offset:offset+n])
                        pred_parts.extend(balanced_labels(probs, n // 2).astype(int).tolist())
                        truth_parts.extend(y[valid_ix].astype(int).tolist())
                        offset += n
                    candidate_oof[weight]["truth"].extend(truth_parts)
                    candidate_oof[weight]["pred"].extend(pred_parts)
                print(f"grid {run_number}/{total_runs} {subject} C={rich_c} cat={cat_cfg['name']}", flush=True)

            for weight in BLEND_WEIGHTS:
                truth = np.asarray(candidate_oof[weight]["truth"], dtype=int)
                pred = np.asarray(candidate_oof[weight]["pred"], dtype=int)
                cm = confusion_matrix(truth, pred, labels=[0, 1])
                accuracy = float(np.mean(truth == pred))
                results.append({
                    "rich_logistic_C": rich_c,
                    "catboost": cat_cfg,
                    "catboost_blend_weight": weight,
                    "correct": int(np.sum(truth == pred)),
                    "total": int(len(truth)),
                    "accuracy": accuracy,
                    "confusion_matrix": cm.tolist(),
                    "labels": ["rest", "move"],
                })
            print("grid config complete", run_number, flush=True)

    # Stable tie-breaks favor the simpler/less nonlinear configuration.
    results.sort(key=lambda r: (r["accuracy"], -r["catboost"]["depth"],
                                -r["catboost"]["iterations"], -r["catboost_blend_weight"]),
                 reverse=True)
    best = results[0]
    best["target_accuracy"] = 0.90
    best["target_reached"] = bool(best["accuracy"] > 0.90)
    report = {
        "validation": "subject-wise stratified five-fold out-of-fold; balanced labels per fold",
        "selection_note": "Best score is selected from this grid, so it is a model-selection score, not an untouched final test estimate.",
        "candidate_count": len(results),
        "rich_logistic_C_values": list(RICH_C_VALUES),
        "catboost_configurations": list(CATBOOST_CONFIGS),
        "blend_weights": list(BLEND_WEIGHTS),
        "best_candidate": best,
        "all_candidates": results,
    }
    Path("output").mkdir(exist_ok=True)
    Path("output/research_grid.json").write_text(json.dumps(report, indent=2))
    print("BEST_GRID_CANDIDATE", json.dumps(best), flush=True)
    print("RESEARCH_TARGET", "REACHED" if best["target_reached"] else "NOT_REACHED", flush=True)


if __name__ == "__main__":
    main()
