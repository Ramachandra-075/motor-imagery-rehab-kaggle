"""Inspect EEG event markers and time resolution; do not expose raw signals."""
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

archive = next(Path("data").glob("*.zip"))
result = {"pages": {}, "sessions": [], "test": [], "file_count": 0}
for page in ("description", "evaluation", "rules"):
    cmd = ["kaggle", "competitions", "pages", "-c",
           "low-cost-motor-imagery-decoding-for-rehab",
           "--page-name", page, "--content"]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    result["pages"][page] = (run.stdout if run.returncode == 0 else run.stderr)[:16000]
with zipfile.ZipFile(archive) as z:
    files = [n for n in z.namelist() if n.endswith(".csv")]
    result["file_count"] = len(files)
    for name in files:
        with z.open(name) as stream:
            frame = pd.read_csv(stream, usecols=["time", "Marker_val"])
        marks = frame["Marker_val"].fillna(-999).to_numpy()
        changes = np.flatnonzero(marks[1:] != marks[:-1]) + 1
        events = [(int(i), str(marks[i])) for i in changes
                  if str(marks[i]) not in ("0", "0.0")]
        detail = {
            "file": name, "rows": len(frame),
            "time_first": str(frame["time"].iloc[0]),
            "time_last": str(frame["time"].iloc[-1]),
            "time_step_median": float(np.median(np.diff(frame["time"].to_numpy(dtype=float)))),
            "marker_counts": dict(Counter(map(str, marks)).most_common(16)),
            "first_events": events[:12],
            "event_count": len(events),
        }
        (result["test"] if "_test" in name else result["sessions"]).append(detail)
Path("metadata.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2)[:57000])
