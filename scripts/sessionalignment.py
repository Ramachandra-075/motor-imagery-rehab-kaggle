"""Test transductive session alignment of narrow-band EEG features.

Session statistics use only unlabeled feature values. No Kaggle submission.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features,trials
from richfeatures import rich_features,probabilities
from train import balanced_labels,discover

REFS=("car","laplacian")


def align(x,groups):
    output=np.empty_like(x)
    for session in np.unique(groups):
        mask=groups==session
        output[mask]=StandardScaler().fit_transform(x[mask])
    return output


def session_probs(records,subject,tr,va):
    item=records[subject]
    local,pooled=[],[]
    for ref in REFS:
        x=item["aligned"][ref]
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
        classifier.fit(x[tr],item["y"][tr])
        local.append(classifier.predict_proba(x[va])[:,1])
        matrix,labels=[],[]
        for other_subject,other in records.items():
            other_x=other["aligned"][ref]
            ix=tr if other_subject==subject else np.arange(len(other["y"]))
            matrix.append(other_x[ix])
            labels.append(other["y"][ix])
        global_clf=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
        global_clf.fit(np.concatenate(matrix),np.concatenate(labels))
        pooled.append(global_clf.predict_proba(x[va])[:,1])
    return .5*np.mean(local,axis=0)+.5*np.mean(pooled,axis=0)


def main():
    records={}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains,_=discover(archive)
        for subject in sorted(trains):
            sessions=[(sid,*trials(archive,path,True)) for sid,path in sorted(trains[subject])]
            x=np.concatenate([s[1] for s in sessions])
            groups=np.concatenate([np.full(len(s[2]),s[0]) for s in sessions])
            rich={ref:rich_features(x,ref) for ref in REFS}
            records[subject]={"y":np.concatenate([s[2] for s in sessions]),
                "groups":groups,"sessions":sessions,
                "old":{ref:build_features(x,ref,"post_halves_ratio") for ref in REFS},
                "rich":rich,
                "aligned":{ref:align(rich[ref],groups) for ref in REFS}}
            print("features",subject,flush=True)
    results={}
    for regime in ("fivefold","recent","other","first"):
        hits=defaultdict(int)
        total=defaultdict(int)
        aucs=defaultdict(list)
        for subject,item in records.items():
            y=item["y"]
            if regime=="fivefold":
                out={name:np.full(len(y),np.nan) for name in
                     ("v6","session","session25","session50")}
                folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927)
                for tr,va in folds.split(np.zeros(len(y)),y):
                    v6=probabilities(records,subject,tr,va)["blend75"]
                    p=session_probs(records,subject,tr,va)
                    for name,pred in (("v6",v6),("session",p),
                                      ("session25",.75*v6+.25*p),
                                      ("session50",.5*v6+.5*p)):
                        out[name][va]=pred
                truth=y
            else:
                if regime in ("first","other") and len(item["sessions"])<3:
                    continue
                sid=(item["sessions"][-1][0] if regime=="recent" else
                     item["sessions"][1][0] if regime=="other" else
                     item["sessions"][0][0])
                va=np.flatnonzero(item["groups"]==sid)
                tr=np.flatnonzero(item["groups"]!=sid)
                v6=probabilities(records,subject,tr,va)["blend75"]
                p=session_probs(records,subject,tr,va)
                out={"v6":v6,"session":p,"session25":.75*v6+.25*p,
                     "session50":.5*v6+.5*p}
                truth=y[va]
            for name,pred in out.items():
                assert np.isfinite(pred).all()
                hits[name]+=int(np.sum(balanced_labels(pred,len(pred)//2)==truth))
                total[name]+=len(truth)
                aucs[name].append((len(truth),float(roc_auc_score(truth,pred))))
            print(regime,subject,flush=True)
        results[regime]={name:{"correct":hits[name],"total":total[name],
            "accuracy":hits[name]/total[name],
            "subject_weighted_auc":float(np.average(
                [value for _,value in aucs[name]],
                weights=[count for count,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(results[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/sessionalignment.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
