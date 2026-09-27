"""Evaluate narrow-band model regularization with five-fold and session checks."""
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

from baseline import build_features, trials
from richfeatures import rich_features, probabilities
from train import balanced_labels, discover

REFS=("car","laplacian")
STRENGTHS=(.003,.03,.3)


def predict(records,subject,tr,va):
    record=records[subject]
    base=probabilities(records,subject,tr,va)["v5"]
    local={c:[] for c in STRENGTHS}
    pooled={c:[] for c in STRENGTHS}
    for ref in REFS:
        x=record["rich"][ref]
        for c in STRENGTHS:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000))
            model.fit(x[tr],record["y"][tr])
            local[c].append(model.predict_proba(x[va])[:,1])
        matrix,labels=[],[]
        for other_subject,other in records.items():
            other_x=other["rich"][ref]
            ix=tr if other_subject==subject else np.arange(len(other["y"]))
            scaler=StandardScaler().fit(other_x[ix])
            matrix.append(scaler.transform(other_x[ix]))
            labels.append(other["y"][ix])
            if other_subject==subject:
                query=scaler.transform(x[va])
        pool_x,pool_y=np.concatenate(matrix),np.concatenate(labels)
        for c in STRENGTHS:
            model=LogisticRegression(C=c,max_iter=2000)
            model.fit(pool_x,pool_y)
            pooled[c].append(model.predict_proba(query)[:,1])
    result={"v5":base}
    for c in STRENGTHS:
        rich=.5*np.mean(local[c],axis=0)+.5*np.mean(pooled[c],axis=0)
        result[f"rich_{c:g}"]=rich
        result[f"blend_{c:g}"]=.25*base+.75*rich
    return result


def main():
    records={}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains,_=discover(archive)
        for subject in sorted(trains):
            sessions=[(sid,*trials(archive,path,True)) for sid,path in sorted(trains[subject])]
            x=np.concatenate([s[1] for s in sessions])
            records[subject]={"y":np.concatenate([s[2] for s in sessions]),
                "groups":np.concatenate([np.full(len(s[2]),s[0]) for s in sessions]),
                "sessions":sessions,
                "old":{ref:build_features(x,ref,"post_halves_ratio") for ref in REFS},
                "rich":{ref:rich_features(x,ref) for ref in REFS}}
            print("features",subject,flush=True)
    results={}
    for regime in ("fivefold","recent","other"):
        hits=defaultdict(int)
        counts=defaultdict(int)
        aucs=defaultdict(list)
        for subject,record in records.items():
            y=record["y"]
            if regime=="fivefold":
                oof={name:np.full(len(y),np.nan) for name in
                     ["v5",*[f"rich_{c:g}" for c in STRENGTHS],
                      *[f"blend_{c:g}" for c in STRENGTHS]]}
                folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927)
                for tr,va in folds.split(np.zeros(len(y)),y):
                    for name,p in predict(records,subject,tr,va).items():
                        oof[name][va]=p
                truth=y
            else:
                if regime=="other" and len(record["sessions"])<3:
                    continue
                sid=record["sessions"][-1][0] if regime=="recent" else record["sessions"][1][0]
                va=np.flatnonzero(record["groups"]==sid)
                tr=np.flatnonzero(record["groups"]!=sid)
                oof=predict(records,subject,tr,va)
                truth=y[va]
            for name,p in oof.items():
                assert np.isfinite(p).all()
                pred=balanced_labels(p,len(p)//2)
                hits[name]+=int(np.sum(pred==truth))
                counts[name]+=len(truth)
                aucs[name].append((len(truth),float(roc_auc_score(truth,p))))
            print(regime,subject,flush=True)
        results[regime]={name:{"correct":hits[name],"total":counts[name],
            "accuracy":hits[name]/counts[name],"subject_weighted_auc":float(
                np.average([v for _,v in aucs[name]],weights=[n for n,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(results[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/regularization.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
