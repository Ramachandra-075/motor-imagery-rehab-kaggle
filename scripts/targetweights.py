"""Evaluate target-subject weighting for cross-subject narrow-band models."""
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
WEIGHTS=(1,3,8)


def predict(records,subject,tr,va):
    record=records[subject]
    v5=probabilities(records,subject,tr,va)["v5"]
    local_probs=[]
    pooled_probs={weight:[] for weight in WEIGHTS}
    for ref in REFS:
        x=record["rich"][ref]
        local=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
        local.fit(x[tr],record["y"][tr])
        local_probs.append(local.predict_proba(x[va])[:,1])
        all_x,all_y,all_weights=[],[],[]
        for other_subject,other in records.items():
            matrix=other["rich"][ref]
            use=tr if other_subject==subject else np.arange(len(other["y"]))
            scaler=StandardScaler().fit(matrix[use])
            all_x.append(scaler.transform(matrix[use]))
            all_y.append(other["y"][use])
            all_weights.extend([other_subject==subject]*len(use))
            if other_subject==subject:
                query=scaler.transform(x[va])
        data=np.concatenate(all_x)
        labels=np.concatenate(all_y)
        target=np.array(all_weights)
        for weight in WEIGHTS:
            clf=LogisticRegression(C=.03,max_iter=1500)
            sample_weights=np.where(target,weight,1.0)
            clf.fit(data,labels,sample_weight=sample_weights)
            pooled_probs[weight].append(clf.predict_proba(query)[:,1])
    results={"v5":v5}
    for weight in WEIGHTS:
        rich=.5*np.mean(local_probs,axis=0)+.5*np.mean(pooled_probs[weight],axis=0)
        results[f"rich_weight{weight}"]=rich
        results[f"blend_weight{weight}"]=.25*v5+.75*rich
    return results


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
    for regime in ("fivefold","recent","other","first"):
        hits=defaultdict(int)
        counts=defaultdict(int)
        aucs=defaultdict(list)
        for subject,record in records.items():
            y=record["y"]
            if regime=="fivefold":
                names=["v5",*[f"rich_weight{w}" for w in WEIGHTS],
                       *[f"blend_weight{w}" for w in WEIGHTS]]
                oof={name:np.full(len(y),np.nan) for name in names}
                folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927)
                for tr,va in folds.split(np.zeros(len(y)),y):
                    for name,p in predict(records,subject,tr,va).items():
                        oof[name][va]=p
                truth=y
            else:
                if regime in ("other","first") and len(record["sessions"])<3:
                    continue
                sid=(record["sessions"][-1][0] if regime=="recent" else
                     record["sessions"][1][0] if regime=="other" else record["sessions"][0][0])
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
    Path("output/targetweights.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
