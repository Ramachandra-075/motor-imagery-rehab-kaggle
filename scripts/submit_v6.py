"""Build unsubmitted V6 candidate from the validation-backed narrow-band ensemble."""
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features, trials
from richfeatures import rich_features
from search import model
from train import balanced_labels, discover

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
            global_model=LogisticRegression(C=.03,max_iter=1500)
            global_model.fit(np.concatenate(list(x_all.values())),np.concatenate(y_all))
            for subject in records:
                pooled[subject][family].append(global_model.predict_proba(query[subject])[:,1])
    predictions=[]
    for subject,record in records.items():
        local={"old":[],"rich":[]}
        for ref,method in REFS:
            old_model=model(method)
            old_model.fit(record["old_train"][ref],record["y"])
            local["old"].append(old_model.predict_proba(record["old_test"][ref])[:,1])
            rich_model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500))
            rich_model.fit(record["rich_train"][ref],record["y"])
            local["rich"].append(rich_model.predict_proba(record["rich_test"][ref])[:,1])
        old=.5*np.mean(local["old"],axis=0)+.5*np.mean(pooled[subject]["old"],axis=0)
        rich=.5*np.mean(local["rich"],axis=0)+.5*np.mean(pooled[subject]["rich"],axis=0)
        labels=balanced_labels(.25*old+.75*rich,20)
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
        "method":"25% V5 plus 75% narrow-band temporal EEG, local and cross-subject",
        "fivefold":{"accuracy":1413/1795,"correct":1413,"total":1795,
                    "subject_weighted_auc":0.868199138635641},
        "recent_session":{"accuracy":127/170,"correct":127,"total":170},
        "other_session":{"accuracy":622/800,"correct":622,"total":800},
        "kaggle_public_score":"not submitted"
    },indent=2))


if __name__=="__main__":
    main()
