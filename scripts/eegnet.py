"""End-to-end EEGNet-style classifier on raw trial waveforms."""
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import balanced_labels, discover, read_epochs

torch.set_num_threads(2)


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (8, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(.4),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(.4),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.body(x)


def normalize(epochs):
    # Fixed data-dependent normalization used for every subject.
    x = epochs.astype(np.float32).copy()
    x -= np.mean(x, axis=-1, keepdims=True)
    scale = np.std(x, axis=-1, keepdims=True)
    x /= np.maximum(scale, 1e-6)
    return np.clip(x, -5, 5)[:, None]


def fit_predict(xtrain, ytrain, xquery, seed, epochs=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtrain = torch.tensor(normalize(xtrain))
    xquery = torch.tensor(normalize(xquery))
    ytrain = torch.tensor(ytrain, dtype=torch.long)
    net = EEGNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    loss_fn = nn.CrossEntropyLoss()
    probs = []
    for epoch in range(epochs):
        net.train()
        indices = rng.permutation(len(ytrain))
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 16)))):
            optimizer.zero_grad()
            # Small temporal shifts preserve the attempted-movement label.
            sample = xtrain[batch]
            if epoch > 2:
                sample = torch.roll(sample, shifts=int(rng.integers(-8, 9)), dims=-1)
            logits = net(sample)
            loss = loss_fn(logits, ytrain[batch])
            loss.backward()
            optimizer.step()
        if epoch >= epochs - 10:
            net.eval()
            with torch.no_grad():
                probs.append(net(xquery).softmax(1)[:, 1].numpy())
    return np.mean(probs, axis=0)


def main():
    archive = next(Path("data").glob("*.zip"))
    detail = []
    with zipfile.ZipFile(archive) as z:
        trains, tests = discover(z)
        for subject in sorted(tests):
            sessions = []
            for sid, path in sorted(trains[subject]):
                x, y = read_epochs(z, path, True)
                sessions.append((sid, x, y))
            x = np.concatenate([s[1] for s in sessions])
            y = np.concatenate([s[2] for s in sessions])
            groups = np.concatenate([np.full(len(s[2]), s[0]) for s in sessions])
            for label, sid in [("recent", sessions[-1][0]),
                               *([("other", sessions[1][0])] if len(sessions) >= 3 else [])]:
                val = groups == sid
                prob = np.mean([fit_predict(x[~val], y[~val], x[val], seed)
                                for seed in (7, 23)], axis=0)
                pred = balanced_labels(prob, len(prob)//2)
                detail.append({"subject": subject, "split": label, "trials": len(pred),
                               "correct": int(np.sum(pred == y[val]))})
            print(subject, detail[-1], flush=True)
    totals = {}
    for label in ("recent", "other"):
        rows = [r for r in detail if r["split"] == label]
        totals[label] = {"correct": sum(r["correct"] for r in rows),
                         "trials": sum(r["trials"] for r in rows)}
        totals[label]["accuracy"] = totals[label]["correct"] / totals[label]["trials"]
    result = {"method": "EEGNet raw signal, 40 epochs, two seeds", "totals": totals, "details": detail}
    Path("output").mkdir(exist_ok=True)
    Path("output/eegnet.json").write_text(json.dumps(result, indent=2))
    print("SUMMARY", json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
