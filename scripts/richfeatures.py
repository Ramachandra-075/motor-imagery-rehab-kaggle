"""Assess narrow-band temporal EEG representation with both fold and session validation.

No Kaggle submission.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features, rereference, trials
from search import model
from train import FS, balanced_labels, discover

REFS=(("car","logreg03"),("laplacian","lda"))
BANDS=((4,8),(8,10),(10,12),(12,15),(15,18),(18,22),(22,26),(26,30),(30,38))


def rich_features(epochs, reference):
    x=rereference(epochs,reference)
    features=[]
    for lo,hi in BANDS:
        sos=butter(4,[lo,hi],btype="bandpass",fs=FS,output="sos")
        f=sosfiltfilt(sos,x,axis=-1)
        base=np.log(np.var(f[:,:,0:250],axis=-1)+1e-10)
        post=np.log(np.var(f[:,:,375:1375],axis=-1)+1e-10)
        first=np.log(np.var(f[:,:,375:875],axis=-1)+1e-10)
        second=np.log(np.var(f[:,:,875:1375],axis=-1)+1e-10)
        features.extend((post,post-base,first,second))
    return np.hstack(features).astype(np.float32)


def probabilities(records, subject, train_ix, valid_ix):
    record=records[subject]
    probs={"old_local":[],"old_pooled":[],"rich_local":[],"rich_pooled":[]}
    for ref,method in REFS:
        for family in ("old","rich"):
            x=record[family][ref]
            if family=="old":
                clf=model(method)
            else:
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
            clf.fit(x[train_ix],record["y"][train_ix])
            probs[family+"_local"].append(clf.predict_proba(x[valid_ix])[:,1])
            tr,labels=[],[]
            for other_subject,other in records.items():
                matrix=other[family][ref]
                use=train_ix if other_subject==subject else np.arange(len(other["y"]))
                scaler=StandardScaler().fit(matrix[use])
                tr.append(scaler.transform(matrix[use]))
                labels.append(other["y"][use])
                if other_subject==subject:
                    query=scaler.transform(x[valid_ix])
            global_clf=LogisticRegression(C=.03,max_iter=1500)
            global_clf.fit(np.concatenate(tr),np.concatenate(labels))
            probs[family+"_pooled"].append(global_clf.predict_proba(query)[:,1])
    old=.5*np.mean(probs["old_local"],axis=0)+.5*np.mean(probs["old_pooled"],axis=0)
    rich=.5*np.mean(probs["rich_local"],axis=0)+.5*np.mean(probs["rich_pooled"],axis=0)
    return {"v5":old,"rich":rich,"blend25":.75*old+.25*rich,
            "blend50":.5*old+.5*rich,"blend75":.25*old+.75*rich}


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
                "old":{ref:build_features(x,ref,"post_halves_ratio") for ref,_ in REFS},
                "rich":{ref:rich_features(x,ref) for ref,_ in REFS}}
            print("features",subject,flush=True)
    results={}
    for regime in ("fivefold","recent","other"):
        hits=defaultdict(int)
        counts=defaultdict(int)
        aucs=defaultdict(list)
        for subject,record in records.items():
            y=record["y"]
            if regime=="fivefold":
                splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927).split(
                    np.zeros(len(y)),y))
                out={name:np.full(len(y),np.nan) for name in ("v5","rich","blend25","blend50","blend75")}
                for tr,va in splits:
                    for name,p in probabilities(records,subject,tr,va).items():
                        out[name][va]=p
                truth=y
            else:
                if regime=="other" and len(record["sessions"])<3:
                    continue
                sid=record["sessions"][-1][0] if regime=="recent" else record["sessions"][1][0]
                va=np.flatnonzero(record["groups"]==sid)
                tr=np.flatnonzero(record["groups"]!=sid)
                out=probabilities(records,subject,tr,va)
                truth=y[va]
            for name,p in out.items():
                assert np.isfinite(p).all(),(subject,name)
                pred=balanced_labels(p,len(p)//2)
                hits[name]+=int(np.sum(pred==truth))
                counts[name]+=len(truth)
                aucs[name].append((len(truth),float(roc_auc_score(truth,p))))
            print(regime,subject,flush=True)
        results[regime]={name:{"correct":hits[name],"total":counts[name],
            "accuracy":hits[name]/counts[name],
            "subject_weighted_auc":float(np.average(
                [v for _,v in aucs[name]],weights=[n for n,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(results[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/richfeatures.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
