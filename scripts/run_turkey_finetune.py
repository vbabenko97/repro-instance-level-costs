"""Claim 4 (TOY): top-block fine-tuning vs frozen features on Turkey, same split.

No GPU is available (HF Jobs 402). Full end-to-end ResNet-50 fine-tuning is
impractical on this Apple M4 (MPS) in reasonable wall-clock, so this is a
reduced-*scope* proxy for the paper's "Turkey ResNet50 frozen → ResNet-FT"
comparison (Table 2 / Fig. 5). It tests the *mechanism* of Claim 4 — adapting
representations beyond frozen ImageNet features improves BOTH NEC and error rate
on the identical split — by fine-tuning only ResNet-50's top block (layer4 + fc)
on cached trunk (through-layer3) features, against the frozen-feature +
logistic-regression baseline.

Scale/scope reductions vs the paper (labeled toy): MPS not GPU; stratified
subset; only the top block is fine-tuned (not the full network); 3 seeds
(paper 10). Paper target (Table 2): frozen 4.2 NEC / 6.7 error → ResNet-FT
2.3 / 4.8 (full end-to-end, GPU, 10 seeds).
"""

import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch import nn
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import error_rate, nec, sign_label, split_80_10_10, vote_log_odds

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "turkey" / "Turkey.zip"
N_SUB = int(os.environ.get("FT_N", "2500"))
EPOCHS = int(os.environ.get("FT_EPOCHS", "12"))
SEEDS = [0, 1, 2]
device = "mps" if torch.backends.mps.is_available() else "cpu"

z = zipfile.ZipFile(ZIP)
ann = json.loads(z.read("Turkey/annotations.json"))
votes = defaultdict(lambda: defaultdict(int))
for rec in ann:
    for a in rec["annotations"]:
        votes[a["image_path"]][a["class_label"]] += 1
paths = sorted(votes)
n_inj = np.array([votes[p]["head_injury"] + votes[p]["plumage_injury"] for p in paths], float)
n_not = np.array([votes[p]["not_injured"] for p in paths], float)
delta_all = vote_log_odds(n_inj, n_not)

rng = np.random.default_rng(0)
y_all = sign_label(delta_all)
sub = np.concatenate(
    [
        rng.choice(np.flatnonzero(y_all == c), size=int(N_SUB * np.mean(y_all == c)), replace=False)
        for c in (-1, 1)
    ]
)
sub.sort()
paths = [paths[i] for i in sub]
delta = delta_all[sub]
y01 = (sign_label(delta) == 1).astype(np.float32)
print(f"toy subset: {len(paths)} images, injured(+) frac={float(np.mean(y01)):.3f}, {EPOCHS} epochs, {len(SEEDS)} seeds, device={device}")

tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Split ResNet-50 into a frozen trunk (conv1..layer3) and a trainable top
# (layer4 + avgpool + fc). Cache trunk features once; the top block trains fast.
base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
trunk = nn.Sequential(
    base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3
).eval().to(device)

print("caching trunk (through layer3) features...")
trunk_feats = []
with torch.no_grad():
    buf = []
    for i, p in enumerate(paths):
        buf.append(tf(Image.open(io.BytesIO(z.read(p))).convert("RGB")))
        if len(buf) == 64 or i == len(paths) - 1:
            trunk_feats.append(trunk(torch.stack(buf).to(device)).cpu())
            buf = []
trunk_feats = torch.cat(trunk_feats)  # (N, 1024, 14, 14)
print(f"trunk feature tensor: {tuple(trunk_feats.shape)}")

# Frozen baseline: global-avg-pool the trunk-through-layer4 (ImageNet) features
# + LogReg. Reuse the standard 2048-d frozen features already produced by
# run_turkey.py when available for parity; else pool layer3 here.
frozen_npz = ROOT / "data" / "turkey" / "features.npz"
frozen_all = np.load(frozen_npz)["X"] if frozen_npz.exists() else None


def make_top():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    top = nn.Sequential(m.layer4, m.avgpool, nn.Flatten(), nn.Linear(m.fc.in_features, 1))
    return top.to(device)


def eval_top(top, idx):
    top.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 128):
            out.append(top(trunk_feats[idx[s : s + 128]].to(device)).squeeze(1).cpu().numpy())
    yhat = np.where(np.concatenate(out) >= 0, 1, -1)
    return nec(delta[idx], yhat), error_rate(delta[idx], yhat)


rows = []
for seed in SEEDS:
    tr, _va, te = split_80_10_10(len(delta), delta, seed)
    # frozen baseline on same split
    if frozen_all is not None:
        Xf = frozen_all[sub]
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xf[tr], sign_label(delta[tr]))
        fn, fe = nec(delta[te], clf.predict(Xf[te])), error_rate(delta[te], clf.predict(Xf[te]))
    else:
        fn, fe = float("nan"), float("nan")
    # fine-tune top block
    top = make_top()
    opt = torch.optim.AdamW(top.parameters(), lr=1e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y01)
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        top.train()
        order = tr[torch.randperm(len(tr), generator=g).numpy()]
        for s in range(0, len(order), 32):
            b = order[s : s + 32]
            opt.zero_grad()
            out = top(trunk_feats[b].to(device)).squeeze(1)
            loss = lossf(out, yt[b].to(device))
            loss.backward()
            opt.step()
    tn, teo = eval_top(top, te)
    print(f"seed {seed}: frozen NEC={fn * 100:.2f} err={fe * 100:.2f} | top-FT NEC={tn * 100:.2f} err={teo * 100:.2f}")
    rows.append({"seed": seed, "frozen_nec": fn, "frozen_error": fe, "ft_nec": tn, "ft_error": teo})

df = pd.DataFrame(rows)
m = df.mean(numeric_only=True) * 100
print(f"\nTOY ({N_SUB} imgs, top-block FT, {EPOCHS} ep, {len(SEEDS)} seeds):")
print(f"  frozen      NEC={m['frozen_nec']:.2f} error={m['frozen_error']:.2f}")
print(f"  top-block FT NEC={m['ft_nec']:.2f} error={m['ft_error']:.2f}")
print(f"  paper (full FT, GPU): frozen 4.2/6.7 -> ResNet-FT 2.3/4.8")

out = ROOT / "outputs" / "turkey_ft"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "per_seed.csv", index=False)
pd.DataFrame(
    [
        {"model": "frozen (toy subset)", "nec": round(m["frozen_nec"], 2), "error": round(m["frozen_error"], 2)},
        {"model": "top-block FT (toy subset)", "nec": round(m["ft_nec"], 2), "error": round(m["ft_error"], 2)},
        {"model": "frozen (paper, full)", "nec": 4.2, "error": 6.7},
        {"model": "ResNet-FT (paper, full)", "nec": 2.3, "error": 4.8},
    ]
).to_csv(out / "summary.csv", index=False)
print(f"wrote {out}/summary.csv")
