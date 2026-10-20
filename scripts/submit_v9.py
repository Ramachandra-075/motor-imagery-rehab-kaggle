"""Build V9 spectral and nonlinear blend for manual Kaggle submission."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from richfeatures import rich_features
from search import model
from train import balanced_labels, discover

REFS=(("car","logreg03"),("laplacian","lda"))


def main():
    grid = json.loads(Path("output/research_grid.json").read_text())
    best = grid["best_candidate"]
    rich_c = float(best["rich_logistic_C"])
    cat_config = {k: v for k, v in best["catboost"].items() if k != "name"}
    cat_weight = float(best["catboost_blend_weight"])
    records={}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains,tests=discover(archive)
        for subject in sorted(tests):
            labeled=[trials(archive,name,True) for _,name in sorted(trains[subject])]
            x=np.concatenate([pair[0] for pair in labeled])
            y=np.concatenate([pair[1] for pair in labeled])
            query,_=trials(archive,tests[subject],False)
            if len(query)!=40:
                raise ValueError(f"{subject}: expected 40 test epochs, found {len(query)}")
            records[subject]={
                "y":y,
                "old_train":{ref:build_features(x,ref,"post_halves_ratio") for ref,_ in REFS},
                "old_test":{ref:build_features(query,ref,"post_halves_ratio") for ref,_ in REFS},
                "rich_train":{ref:rich_features(x,ref) for ref,_ in REFS},
                "rich_test":{ref:rich_features(query,ref) for ref,_ in REFS}}
            print("features",subject,flush=True)
    pooled={name:{"old":[],"rich":[]} for name in records}
    for family in ("old","rich"):
        for ref,_ in REFS:
            x_all,y_all,query={},[],{}
            for subject,record in records.items():
                scaler=StandardScaler().fit(record[family+"_train"][ref])
                x_all[subject]=scaler.transform(record[family+"_train"][ref])
                y_all.append(record["y"])
                query[subject]=scaler.transform(record[family+"_test"][ref])
            global_c = rich_c if family == "rich" else 0.03
            global_model=LogisticRegression(C=global_c,max_iter=1500)
            global_model.fit(np.concatenate(list(x_all.values())),np.concatenate(y_all))
            for subject in records:
                pooled[subject][family].append(global_model.predict_proba(query[subject])[:,1])
    predictions=[]
    for subject,record in records.items():
        local={"old":[],"rich":[],"cat":[]}
        for ref,method in REFS:
            old_model=model(method)
            old_model.fit(record["old_train"][ref],record["y"])
            local["old"].append(old_model.predict_proba(record["old_test"][ref])[:,1])
            rich_model=make_pipeline(StandardScaler(),LogisticRegression(C=rich_c,max_iter=1500))
            rich_model.fit(record["rich_train"][ref],record["y"])
            local["rich"].append(rich_model.predict_proba(record["rich_test"][ref])[:,1])
            cat_settings={"iterations":220,"depth":4,"learning_rate":.04,
                "l2_leaf_reg":10,"loss_function":"Logloss","verbose":False,
                "thread_count":1,"random_seed":20260928}
            cat_settings.update(cat_config)
            cat=CatBoostClassifier(**cat_settings)
            cat.fit(record["rich_train"][ref],record["y"])
            local["cat"].append(cat.predict_proba(record["rich_test"][ref])[:,1])
        old=.5*np.mean(local["old"],axis=0)+.5*np.mean(pooled[subject]["old"],axis=0)
        rich=.5*np.mean(local["rich"],axis=0)+.5*np.mean(pooled[subject]["rich"],axis=0)
        v6=.25*old+.75*rich
        labels=balanced_labels((1.0-cat_weight)*v6+cat_weight*np.mean(local["cat"],axis=0),20)
        predictions.extend(np.where(labels,"move","rest").tolist())
        print("predicted",subject,len(record["y"]),int(labels.sum()),flush=True)
    if len(predictions)!=680:
        raise ValueError(f"Expected 680 predictions, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["ID","TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method":"selected EEG research-grid candidate",
        "selected_candidate":best,
        "fivefold_oof_accuracy":best["accuracy"],
        "fivefold_oof_correct":best["correct"],
        "fivefold_oof_total":best["total"],
        "confusion_matrix":best["confusion_matrix"],
        "confusion_matrix_labels":best["labels"],
        "target_accuracy":0.90,
        "target_reached":bool(best["accuracy"] > 0.90),
        "kaggle_public_score":"not submitted",
        "kaggle_submitted":False
    },indent=2))


if __name__=="__main__":
    main()
