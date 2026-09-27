"""Test band-specific EEG channel connectivity as an independent ensemble signal."""
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
from richfeatures import rich_features, probabilities
from train import FS, balanced_labels, discover

REFS=("car","laplacian")
BANDS=((4,8),(8,13),(13,22),(22,30))


def connectivity(epochs,reference):
    x=rereference(epochs,reference)
    # Lower-dimensional motor / posterior network.
    channels=(1,2,3,4,5,6)
    if reference=="laplacian":
        channels+=(8,9,10)
    upper=np.triu_indices(len(channels),1)
    parts=[]
    for lo,hi in BANDS:
        sos=butter(4,[lo,hi],btype="bandpass",fs=FS,output="sos")
        band=sosfiltfilt(sos,x,axis=-1)[:,channels]
        pair=[]
        for start,stop in ((0,250),(375,1375)):
            segment=band[:,:,start:stop]
            segment=segment-segment.mean(axis=-1,keepdims=True)
            scale=np.sqrt(np.sum(segment**2,axis=-1,keepdims=True)+1e-10)
            segment=segment/scale
            cor=np.einsum("nct,ndt->ncd",segment,segment)[:,upper[0],upper[1]]
            pair.append(np.arctanh(np.clip(cor,-.99,.99)))
        parts.extend((pair[1],pair[1]-pair[0]))
    return np.hstack(parts).astype(np.float32)


def connectivity_probs(records,subject,tr,va):
    item=records[subject]
    local_p=[]
    pool_p=[]
    for ref in REFS:
        x=item["conn"][ref]
        local=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
        local.fit(x[tr],item["y"][tr])
        local_p.append(local.predict_proba(x[va])[:,1])
        matrices,labels=[],[]
        for other_subject,other in records.items():
            other_x=other["conn"][ref]
            use=tr if other_subject==subject else np.arange(len(other["y"]))
            scaler=StandardScaler().fit(other_x[use])
            matrices.append(scaler.transform(other_x[use]))
            labels.append(other["y"][use])
            if other_subject==subject:
                query=scaler.transform(x[va])
        pooled=LogisticRegression(C=.03,max_iter=1500)
        pooled.fit(np.concatenate(matrices),np.concatenate(labels))
        pool_p.append(pooled.predict_proba(query)[:,1])
    return .5*np.mean(local_p,axis=0)+.5*np.mean(pool_p,axis=0)


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
                "rich":{ref:rich_features(x,ref) for ref in REFS},
                "conn":{ref:connectivity(x,ref) for ref in REFS}}
            print("features",subject,flush=True)
    results={}
    for regime in ("fivefold","recent","other"):
        hits=defaultdict(int)
        totals=defaultdict(int)
        aucs=defaultdict(list)
        for subject,record in records.items():
            y=record["y"]
            if regime=="fivefold":
                splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927).split(
                    np.zeros(len(y)),y))
                oof={name:np.full(len(y),np.nan) for name in
                     ("rich75","conn","conn25","conn50")}
                for tr,va in splits:
                    initial=probabilities(records,subject,tr,va)
                    rich=initial["blend75"]
                    conn=connectivity_probs(records,subject,tr,va)
                    for name,p in {"rich75":rich,"conn":conn,"conn25":.75*rich+.25*conn,
                                   "conn50":.5*rich+.5*conn}.items():
                        oof[name][va]=p
                truth=y
            else:
                if regime=="other" and len(record["sessions"])<3:
                    continue
                sid=record["sessions"][-1][0] if regime=="recent" else record["sessions"][1][0]
                va=np.flatnonzero(record["groups"]==sid)
                tr=np.flatnonzero(record["groups"]!=sid)
                initial=probabilities(records,subject,tr,va)
                rich=initial["blend75"]
                conn=connectivity_probs(records,subject,tr,va)
                oof={"rich75":rich,"conn":conn,"conn25":.75*rich+.25*conn,
                     "conn50":.5*rich+.5*conn}
                truth=y[va]
            for name,p in oof.items():
                assert np.isfinite(p).all()
                pred=balanced_labels(p,len(p)//2)
                hits[name]+=int(np.sum(pred==truth))
                totals[name]+=len(truth)
                aucs[name].append((len(truth),float(roc_auc_score(truth,p))))
            print(regime,subject,flush=True)
        results[regime]={name:{"correct":hits[name],"total":totals[name],
            "accuracy":hits[name]/totals[name],"subject_weighted_auc":float(
                np.average([v for _,v in aucs[name]],weights=[n for n,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(results[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/connectivity.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
