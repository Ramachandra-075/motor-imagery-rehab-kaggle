"""Inspect within-session label order and balance for legitimate sequence priors."""
import json
import zipfile
from pathlib import Path

import numpy as np
from baseline import trials
from train import discover

def main():
    report=[]
    with zipfile.ZipFile(next(Path("data").glob("*.zip"))) as z:
        trains,_=discover(z)
        for subject in sorted(trains):
            for sid,path in sorted(trains[subject]):
                _,y=trials(z,path,True)
                transitions=int(np.sum(y[1:]!=y[:-1]))
                ten_counts=[int(y[i:i+10].sum()) for i in range(0,len(y),10)]
                report.append({"subject":subject,"session":sid,"n":len(y),
                               "move":int(y.sum()),"transition_rate":transitions/(len(y)-1),
                               "ten_move_counts":ten_counts})
    print("SESSION_COUNT",len(report))
    print("TRANSITION_RATE",float(np.average([r["transition_rate"] for r in report],
                                               weights=[r["n"]-1 for r in report])))
    print("TEN_BLOCK_BALANCED",sum(count==5 for r in report for count in r["ten_move_counts"]),
          "/",sum(len(r["ten_move_counts"]) for r in report))
    print("SESSION_BALANCED",sum(r["move"]*2==r["n"] for r in report),"/",len(report))
    print("COUNTS",json.dumps({str(n):sum(r["n"]==n for r in report)
                                for n in sorted({r["n"] for r in report})}))
    Path("output").mkdir(exist_ok=True)
    Path("output/sequence.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
