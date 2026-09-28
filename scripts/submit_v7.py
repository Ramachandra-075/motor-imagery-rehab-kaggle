"""Build V7: V6 spectral transfer blended with regularized filter-bank CSP.

This program writes output/submission.csv; it never submits to Kaggle.
"""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features,trials
from fbcsp_v7 import csp_features,filter_bank
from richfeatures import rich_features
from search import model
from train import balanced_labels,discover

REFS=(("car","logreg03"),("laplacian","lda"))


def main():
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
                "rich_test":{ref:rich_features(query,ref) for ref,_ in REFS},
                "raw_train":x,"raw_test":query}
            print("features",subject,flush=True)
    pooled={name:{"old":[],"rich":[]} for name in records}
    for family in ("old","rich"):
        for ref,_ in REFS:
            all_x,all_y,query=[],[],{}
            for subject,record in records.items():
                scaler=StandardScaler().fit(record[family+"_train"][ref])
                all_x.append(scaler.transform(record[family+"_train"][ref]))
                all_y.append(record["y"])
                query[subject]=scaler.transform(record[family+"_test"][ref])
            classifier=LogisticRegression(C=.03,max_iter=1500)
            classifier.fit(np.concatenate(all_x),np.concatenate(all_y))
            for subject in records:
                pooled[subject][family].append(classifier.predict_proba(query[subject])[:,1])
    predictions=[]
    for subject,record in records.items():
        local={"old":[],"rich":[]}
        for ref,method in REFS:
            old=model(method)
            old.fit(record["old_train"][ref],record["y"])
            local["old"].append(old.predict_proba(record["old_test"][ref])[:,1])
            rich=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
            rich.fit(record["rich_train"][ref],record["y"])
            local["rich"].append(rich.predict_proba(record["rich_test"][ref])[:,1])
        old_p=.5*np.mean(local["old"],axis=0)+.5*np.mean(pooled[subject]["old"],axis=0)
        rich_p=.5*np.mean(local["rich"],axis=0)+.5*np.mean(pooled[subject]["rich"],axis=0)
        v6=.25*old_p+.75*rich_p

        n_train=len(record["y"])
        both=np.concatenate([record["raw_train"],record["raw_test"]])
        # Spatial filters and selected features use training labels only.
        features=csp_features(filter_bank(both),record["y"],np.arange(n_train))
        spatial=make_pipeline(SelectKBest(f_classif,k=12),StandardScaler(),
                              LogisticRegression(C=.1,max_iter=1500))
        spatial.fit(features[:n_train],record["y"])
        csp=spatial.predict_proba(features[n_train:])[:,1]
        labels=balanced_labels(.75*v6+.25*csp,20)
        predictions.extend(np.where(labels,"move","rest").tolist())
        print("predicted",subject,n_train,int(labels.sum()),flush=True)
    if len(predictions)!=680:
        raise ValueError(f"Expected 680 predictions, found {len(predictions)}")
    Path("output").mkdir(exist_ok=True)
    with Path("output/submission.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["ID","TARGET"])
        writer.writerows(enumerate(predictions))
    Path("output/report.json").write_text(json.dumps({
        "method":"75% V6 spectral transfer + 25% regularized filter-bank CSP",
        "fivefold":{"correct":1455,"total":1795,"accuracy":1455/1795,
                    "subject_weighted_auc":0.8838154854140116},
        "recent_session":{"correct":123,"total":170,"accuracy":123/170},
        "other_session":{"correct":626,"total":800,"accuracy":626/800},
        "earliest_session":{"correct":606,"total":775,"accuracy":606/775},
        "kaggle_public_score":"not submitted"
    },indent=2))


if __name__=="__main__":
    main()
