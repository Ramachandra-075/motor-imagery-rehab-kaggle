"""Evaluate regularized filter-bank covariance geometry against V6.

No Kaggle submission is made.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from richfeatures import rich_features
from search import model
from train import FS, balanced_labels, discover

REFS=(("car","logreg03"),("laplacian","lda"))
BANDS=((8,12),(12,18),(18,26),(26,34))
CHANNELS=(0,1,2,3,5)


def covariance_features(epochs):
    # Covariance is trace normalized, then mapped into a pooled tangent chart.
    signals=epochs[:,CHANNELS,:]
    rows=[]
    tri=np.triu_indices(len(CHANNELS))
    for lo,hi in BANDS:
        sos=butter(4,[lo,hi],btype="bandpass",fs=FS,output="sos")
        filtered=sosfiltfilt(sos,signals,axis=-1)
        matrices=[]
        for start,end in ((375,875),(875,1375)):
            window=filtered[:,:,start:end]
            window=window-window.mean(axis=2,keepdims=True)
            cov=window@window.transpose(0,2,1)/(end-start)
            trace=np.trace(cov,axis1=1,axis2=2)
            cov=cov/(trace[:,None,None]+1e-12)
            cov=cov+np.eye(len(CHANNELS))[None,:,:]*.025
            matrices.append(cov)
        reference=np.mean(np.concatenate(matrices,axis=0),axis=0)
        values,vectors=eigh(reference)
        whiten=(vectors/np.sqrt(np.maximum(values,1e-8)))@vectors.T
        for cov in matrices:
            aligned=whiten[None,:,:]@cov@whiten[None,:,:]
            values,vectors=eigh(aligned)
            logged=(vectors*np.log(np.maximum(values,1e-8))[:,None,:])@vectors.transpose(0,2,1)
            rows.append(logged[:,tri[0],tri[1]])
    return np.concatenate(rows,axis=1).astype(np.float32)


def features(epochs):
    return {
        "old":{ref:build_features(epochs,ref,"post_halves_ratio") for ref,_ in REFS},
        "rich":{ref:rich_features(epochs,ref) for ref,_ in REFS},
        "cov":covariance_features(epochs)
    }


def probabilities(records,subject,tr,va):
    r=records[subject]
    parts=defaultdict(list)
    for ref,method in REFS:
        for family in ("old","rich"):
            matrix=r[family][ref]
            clf=model(method) if family=="old" else make_pipeline(
                StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
            clf.fit(matrix[tr],r["y"][tr])
            parts[family+"_local"].append(clf.predict_proba(matrix[va])[:,1])
            x_all,y_all=[],[]
            for other_name,other in records.items():
                ids=tr if other_name==subject else np.arange(len(other["y"]))
                scaler=StandardScaler().fit(other[family][ref][ids])
                x_all.append(scaler.transform(other[family][ref][ids]))
                y_all.append(other["y"][ids])
                if other_name==subject:
                    target=scaler.transform(matrix[va])
            global_clf=LogisticRegression(C=.03,max_iter=1500)
            global_clf.fit(np.concatenate(x_all),np.concatenate(y_all))
            parts[family+"_pooled"].append(global_clf.predict_proba(target)[:,1])
    old=.5*np.mean(parts["old_local"],axis=0)+.5*np.mean(parts["old_pooled"],axis=0)
    rich=.5*np.mean(parts["rich_local"],axis=0)+.5*np.mean(parts["rich_pooled"],axis=0)
    v6=.25*old+.75*rich

    cov=r["cov"]
    local=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000))
    local.fit(cov[tr],r["y"][tr])
    local_p=local.predict_proba(cov[va])[:,1]
    pooled_x,pooled_y=[],[]
    for other_name,other in records.items():
        ids=tr if other_name==subject else np.arange(len(other["y"]))
        scaler=StandardScaler().fit(other["cov"][ids])
        pooled_x.append(scaler.transform(other["cov"][ids]))
        pooled_y.append(other["y"][ids])
        if other_name==subject:
            target=scaler.transform(cov[va])
    pooled=LogisticRegression(C=.005,max_iter=1500)
    pooled.fit(np.concatenate(pooled_x),np.concatenate(pooled_y))
    cov_p=.5*local_p+.5*pooled.predict_proba(target)[:,1]
    return {"v6":v6,"cov":cov_p,"blend10":.9*v6+.1*cov_p,
            "blend20":.8*v6+.2*cov_p,"blend30":.7*v6+.3*cov_p}


def main():
    records={}
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as archive:
        trains,_=discover(archive)
        for subject in sorted(trains):
            sessions=[(sid,*trials(archive,path,True)) for sid,path in sorted(trains[subject])]
            epochs=np.concatenate([s[1] for s in sessions])
            y=np.concatenate([s[2] for s in sessions])
            records[subject]={"y":y,"groups":np.concatenate(
                [np.full(len(s[2]),s[0]) for s in sessions]),
                "sessions":sessions,**features(epochs)}
            print("features",subject,flush=True)
    scores={}
    for regime in ("recent","other","earliest","fivefold"):
        hits=defaultdict(int); totals=defaultdict(int); aucs=defaultdict(list)
        for subject,r in records.items():
            y=r["y"]
            if regime=="other" and len(r["sessions"])<3:
                continue
            if regime=="fivefold":
                splits=StratifiedKFold(n_splits=5,shuffle=True,
                    random_state=20260927).split(np.zeros(len(y)),y)
                out={name:np.full(len(y),np.nan) for name in (
                    "v6","cov","blend10","blend20","blend30")}
                for tr,va in splits:
                    for name,p in probabilities(records,subject,tr,va).items():
                        out[name][va]=p
                truth=y
            else:
                position={"recent":-1,"other":1,"earliest":0}[regime]
                sid=r["sessions"][position][0]
                va=np.flatnonzero(r["groups"]==sid)
                tr=np.flatnonzero(r["groups"]!=sid)
                out=probabilities(records,subject,tr,va)
                truth=y[va]
            for name,p in out.items():
                pred=balanced_labels(p,len(p)//2)
                hits[name]+=int(np.sum(pred==truth))
                totals[name]+=len(truth)
                aucs[name].append((len(truth),float(roc_auc_score(truth,p))))
            print(regime,subject,flush=True)
        scores[regime]={name:{"correct":hits[name],"total":totals[name],
            "accuracy":hits[name]/totals[name],
            "weighted_auc":float(np.average([v for _,v in aucs[name]],
                                  weights=[n for n,_ in aucs[name]]))}
            for name in hits}
        print("RESULT",regime,json.dumps(scores[regime]),flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/covariance_v8.json").write_text(json.dumps(scores,indent=2))


if __name__=="__main__":
    main()
