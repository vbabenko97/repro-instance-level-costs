"""Turkey image experiment (Claim 2, reduced-scale proxy).

Source (verified public): Zenodo 10.5281/zenodo.8115942 (DCIC benchmark),
Turkey.zip = 8,040 PNGs + annotations.json (84 annotator records, each a list of
{image_path, class_label}). Classes: head_injury, plumage_injury, not_injured.

Per-image vote-margin cost (Eq. 3): Δ = log((n_yes+1)/(n_no+1)).

Binarization ambiguity: the paper says "injured is the minority class", which
only holds if injured = head_injury alone (~11%); counting plumage_injury as
injury makes injured the ~87% majority. We report BOTH:
  - head_only:  yes = head_injury,               no = not_injured + plumage_injury
  - any_injury: yes = head_injury + plumage_injury, no = not_injured
The claim under test is the NEC < error-rate gap on the image modality, which
should hold under either binarization.

Model: frozen ResNet-50 (ImageNet) penultimate features + logistic regression —
the paper's "Turkey ResNet50 frozen embeddings" row. This is a reduced-scale
proxy: CPU/MPS, ImageNet-frozen features (no domain fine-tuning). Paper target
(Table 2, Turkey ResNet50 frozen): standard CE 4.2 NEC / 6.7 error;
|Δ|-weighted 3.6 / 6.1 (%); ratio 1.6x.
"""

import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import error_rate, nec, run_strategies, sign_label, summarize

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "turkey" / "Turkey.zip"
FEAT = ROOT / "data" / "turkey" / "features.npz"
ZENODO_URL = "https://zenodo.org/records/8115942/files/Turkey.zip"

if not ZIP.exists():
    import urllib.request

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ZENODO_URL} (~553 MB)")
    urllib.request.urlretrieve(ZENODO_URL, ZIP)

z = zipfile.ZipFile(ZIP)
import json

ann = json.loads(z.read("Turkey/annotations.json"))
votes = defaultdict(lambda: defaultdict(int))
for rec in ann:
    for a in rec["annotations"]:
        votes[a["image_path"]][a["class_label"]] += 1
paths = sorted(votes)
print(f"{len(paths)} images with annotations")

n_head = np.array([votes[p]["head_injury"] for p in paths], dtype=float)
n_plum = np.array([votes[p]["plumage_injury"] for p in paths], dtype=float)
n_not = np.array([votes[p]["not_injured"] for p in paths], dtype=float)

delta_head = np.log((n_head + 1.0) / (n_not + n_plum + 1.0))
delta_any = np.log((n_head + n_plum + 1.0) / (n_not + 1.0))

if FEAT.exists():
    X = np.load(FEAT)["X"]
    print(f"loaded cached features {X.shape}")
else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"extracting ResNet-50 features on {device}")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    feats = np.zeros((len(paths), 2048), dtype=np.float32)
    batch, idxs = [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = Image.open(io.BytesIO(z.read(p))).convert("RGB")
            batch.append(tf(img))
            idxs.append(i)
            if len(batch) == 64 or i == len(paths) - 1:
                out = net(torch.stack(batch).to(device)).cpu().numpy()
                feats[idxs] = out
                batch, idxs = [], []
                if (i + 1) % 1024 == 0 or i == len(paths) - 1:
                    print(f"  {i + 1}/{len(paths)}")
    X = feats
    np.savez_compressed(FEAT, X=X)
    print(f"cached features to {FEAT}")

out = ROOT / "outputs" / "turkey"
out.mkdir(parents=True, exist_ok=True)
summaries = {}
for name, delta in (("head_only", delta_head), ("any_injury", delta_any)):
    pos = float(np.mean(sign_label(delta) == 1))
    print(f"\n[{name}] injured(+) fraction = {pos:.3f}")
    rows = []
    for seed in range(10):
        rows += run_strategies(
            X,
            delta,
            make_clf=lambda s: LogisticRegression(max_iter=2000, C=1.0, random_state=s),
            make_reg=lambda s: Ridge(alpha=1.0, random_state=s),
            seed=seed,
        )
    s = summarize(rows)
    s["ratio_err_over_nec"] = (s["error_mean"] / s["nec_mean"]).round(2)
    s.insert(0, "binarization", name)
    print(s.to_string(index=False))
    pd.DataFrame(rows).to_csv(out / f"per_seed_{name}.csv", index=False)
    summaries[name] = s

pd.concat(summaries.values(), ignore_index=True).to_csv(out / "summary.csv", index=False)
print(f"\nwrote {out}/summary.csv and per_seed_*.csv")
