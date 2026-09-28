"""Regularized filter-bank CSP: evaluate V6 plus a distinct spatial decoder.

Paper-inspired feature selection stays inside each train fold; no Kaggle submission.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter,sosfiltfilt
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features,rereference,trials
from richfeatures import rich_features,probabilities
from train import FS,balanced_labels,discover

BANDS=((4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36))


def filter_bank(x):
    x=rereference(x,"car")
    return [sosfiltfilt(butter(4,[lo,hi],fs=FS,btype="bandpass",output="sos"),
                        x,axis=-1).astype(np.float32) for lo,hi in BANDS]


def spatial_filters(bank,y,train):
    post=bank[train,:,375:1375]
    post=post-post.mean(axis=-1,keepdims=True)
    cov=post @ post.transpose(0,2,1) / post.shape[-1]
    cov/=np.trace(cov,axis1=1,axis2=2)[:,None,None]+1e-9
    c0=cov[y[train]==0].mean(axis=0)
    c1=cov[y[train]==1].mean(axis=0)
    pooled=.5*(c0+c1)
    # Shrink class covariance to pooled covariance to stabilize small subjects.
    c0=.85*c0+.15*pooled
    c1=.85*c1+.15*pooled
    reg=.03*np.trace(c0+c1)/len(c0)
    _,vectors=eigh(c1,c0+c1+reg*np.eye(c0.shape[0]))
    return vectors[:,[0,1,-2,-1]].T


def csp_features(banks,y,train):
    parts=[]
    for bank in banks:
        projection=spatial_filters(bank,y,train)
        projected=np.einsum("kc,nct->nkt",projection,bank,optimize=True)
        baseline=np.log(np.var(projected[:,:,0:250],axis=-1)+1e-10)
        post=np.log(np.var(projected[:,:,375:1375],axis=-1)+1e-10)
        early=np.log(np.var(projected[:,:,375:875],axis=-1)+1e-10)
        late=np.log(np.var(projected[:,:,875:1375],axis=-1)+1e-10)
        parts.extend((post,post-baseline,early,late))
    return np.hstack(parts)


def score_csp(banks,y,train,valid):
    x=csp_features(banks,y,train)
    clf=make_pipeline(SelectKBest(f_classif,k=12),StandardScaler(),
                      LogisticRegression(C=.1,max_iter=1500))
    clf.fit(x[train],y[train])
    return clf.predict_proba(x[valid])[:,1]


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
                "old":{ref:build_features(x,ref,"post_halves_ratio")
                       for ref in ("car","laplacian")},
                "rich":{ref:rich_features(x,ref)
                        for ref in ("car","laplacian")}}
            print("features",subject,flush=True)
    results={}
    for regime in ("fivefold","recent","other","first"):
        hits=defaultdict(int)
        totals=defaultdict(int)
        aucs=defaultdict(list)
        with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
            trains,_=discover(archive)
            for subject,record in records.items():
                x=np.concatenate([trials(archive,path,True)[0]
                                  for _,path in sorted(trains[subject])])
                banks=filter_bank(x)
                y=record["y"]
                if regime=="fivefold":
                    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260927)
                    names=("v6","csp","csp10","csp15","csp20","csp25","csp50")
                    out={name:np.full(len(y),np.nan) for name in names}
                    for tr,va in folds.split(np.zeros(len(y)),y):
                        v6=probabilities(records,subject,tr,va)["blend75"]
                        csp=score_csp(banks,y,tr,va)
                        for name,p in (("v6",v6),("csp",csp),
                                       ("csp10",.9*v6+.1*csp),
                                       ("csp15",.85*v6+.15*csp),
                                       ("csp20",.8*v6+.2*csp),
                                       ("csp25",.75*v6+.25*csp),
                                       ("csp50",.5*v6+.5*csp)):
                            out[name][va]=p
                    truth=y
                else:
                    if regime in ("other","first") and len(record["sessions"])<3:
                        continue
                    sid=(record["sessions"][-1][0] if regime=="recent" else
                         record["sessions"][1][0] if regime=="other" else
                         record["sessions"][0][0])
                    va=np.flatnonzero(record["groups"]==sid)
                    tr=np.flatnonzero(record["groups"]!=sid)
                    v6=probabilities(records,subject,tr,va)["blend75"]
                    csp=score_csp(banks,y,tr,va)
                    out={"v6":v6,"csp":csp,"csp10":.9*v6+.1*csp,
                         "csp15":.85*v6+.15*csp,"csp20":.8*v6+.2*csp,
                         "csp25":.75*v6+.25*csp,
                         "csp50":.5*v6+.5*csp}
                    truth=y[va]
                for name,p in out.items():
                    assert np.isfinite(p).all()
                    pred=balanced_labels(p,len(p)//2)
                    hits[name]+=int(np.sum(pred==truth))
                    totals[name]+=len(truth)
                    aucs[name].append((len(truth),float(roc_auc_score(truth,p))))
                print(regime,subject,flush=True)
        results[regime]={name:{"correct":hits[name],"total":totals[name],
            "accuracy":hits[name]/totals[name],
            "subject_weighted_auc":float(np.average(
                [v for _,v in aucs[name]],weights=[n for n,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(results[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/fbcsp.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
