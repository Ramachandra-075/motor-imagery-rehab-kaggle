"""Report dataset layout without publishing competition data."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import whosmat

root = Path("data")
report = {"files": [], "tables": [], "arrays": [], "matrices": []}
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    item = {"path": str(path), "bytes": path.stat().st_size}
    report["files"].append(item)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            item["members"] = [
                {"name": m.filename, "bytes": m.file_size}
                for m in members[:120]
            ]
            item["member_count"] = len(members)
            for member in members:
                lower = member.filename.lower()
                if lower.endswith((".csv", ".tsv")) and member.file_size < 200_000_000:
                    with archive.open(member) as source:
                        frame = pd.read_csv(source, sep=chr(9) if lower.endswith(".tsv") else ",", nrows=4)
                    report["tables"].append({
                        "path": member.filename,
                        "columns": list(frame.columns),
                        "preview": frame.head(2).astype(str).to_dict(orient="records")
                        if "submission" in lower else [],
                    })
                elif lower.endswith(".npy") and member.file_size < 200_000_000:
                    with archive.open(member) as source:
                        arr = np.load(source, allow_pickle=False)
                    report["arrays"].append({
                        "path": member.filename, "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    })
    elif path.suffix.lower() in (".csv", ".tsv"):
        frame = pd.read_csv(path, sep=chr(9) if path.suffix.lower() == ".tsv" else ",", nrows=4)
        report["tables"].append({
            "path": str(path), "columns": list(frame.columns),
            "preview": frame.head(2).astype(str).to_dict(orient="records")
            if "submission" in path.name.lower() else [],
        })
    elif path.suffix.lower() == ".npy" and item["bytes"] < 200_000_000:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        report["arrays"].append({"path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype)})
    elif path.suffix.lower() == ".mat":
        try:
            report["matrices"].append({"path": str(path), "variables": whosmat(path)})
        except Exception as exc:
            report["matrices"].append({"path": str(path), "error": str(exc)[:150]})
Path("inspection.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:55000])
