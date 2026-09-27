"""Five-fold out-of-fold AUC and accuracy for V4/V5; never submits to Kaggle."""
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from search import model
from train import balanced_labels, discover

REFS = (("car", "logreg03"), ("laplacian", "lda"))


def main():
    records = {}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains, _ = discover(archive)
        for subject in sorted(trains):
            labeled = [trials(archive, path, True) for _, path in sorted(trains[subject])]
            x = np.concatenate([pair[0] for pair in labeled])
            y = np.concatenate([pair[1] for pair in labeled])
            records[subject] = {
                "y":y,
                "features":{ref:build_features(x,ref,"post_halves_ratio") for ref,_ in REFS}
            }
            print("features",subject,len(y),flush=True)
    details = []
    full_labels = {"v4":[], "v5":[]}
    full_proba = {"v4":[], "v5":[]}
    for subject, record in records.items():
        y=record["y"]
        oof={name:np.full(len(y),np.nan) for name in ("v4","v5")}
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927)
        for train_ix,valid_ix in folds.split(np.zeros(len(y)),y):
            local_probs=[]
            pooled_probs=[]
            for ref,method in REFS:
                x=record["features"][ref]
                local=model(method)
                local.fit(x[train_ix],y[train_ix])
                local_probs.append(local.predict_proba(x[valid_ix])[:,1])
                normalized,labels=[],[]
                for other_subject,other in records.items():
                    other_x=other["features"][ref]
                    use=train_ix if other_subject==subject else np.arange(len(other["y"]))
                    scaler=StandardScaler().fit(other_x[use])
                    normalized.append(scaler.transform(other_x[use]))
                    labels.append(other["y"][use])
                    if other_subject==subject:
                        valid_x=scaler.transform(x[valid_ix])
                pooled=LogisticRegression(C=.03,max_iter=1500)
                pooled.fit(np.concatenate(normalized),np.concatenate(labels))
                pooled_probs.append(pooled.predict_proba(valid_x)[:,1])
            v4=np.mean(local_probs,axis=0)
            v5=.5*v4+.5*np.mean(pooled_probs,axis=0)
            oof["v4"][valid_ix]=v4
            oof["v5"][valid_ix]=v5
        record_result={"subject":subject,"n":len(y)}
        for name,prob in oof.items():
            assert np.isfinite(prob).all()
            predicted=balanced_labels(prob,len(y)//2)
            record_result[name]={"auc":float(roc_auc_score(y,prob)),
                                 "accuracy":float(np.mean(predicted==y)),
                                 "correct":int(np.sum(predicted==y))}
            full_labels[name].extend(y.tolist())
            full_proba[name].append((y,prob))
        details.append(record_result)
        print("evaluated",subject,record_result,flush=True)
    results={}
    for name in ("v4","v5"):
        hits=sum(row[name]["correct"] for row in details)
        total=sum(row["n"] for row in details)
        auc=np.average([row[name]["auc"] for row in details],
                       weights=[row["n"] for row in details])
        results[name]={"accuracy":hits/total,"correct":hits,"trials":total,
                       "subject_weighted_auc":float(auc),
                       "subject_macro_auc":float(np.mean([r[name]["auc"] for r in details]))}
    output={"method":"five stratified shuffled folds within each subject; 20260927 seed",
            "note":"Random folds share recording sessions; prefer held-out-session results to predict Kaggle accuracy. AUC and accuracy are different metrics.",
            "results":results,"subjects":details}
    Path("output").mkdir(exist_ok=True)
    Path("output/fivefold.json").write_text(json.dumps(output,indent=2))
    print("SUMMARY",json.dumps(results,indent=2))


if __name__=="__main__":
    main()
