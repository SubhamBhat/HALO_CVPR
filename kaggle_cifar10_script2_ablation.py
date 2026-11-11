"""
SCRIPT 2: CIFAR-10-LT Full Component Ablation Grid (Single Kaggle GPU Cell)
===========================================================================
Isolates every HALO component and records BOTH Raw (tau=0) and Calibrated (+tau*) metrics:
  - Weighting exponents: code default (0.33->0.33), paper derivation (1.0->0.5), Only W_rep (beta=1), Only W_pen (beta=0.5)
  - Phase transition gamma(A): quadratic (p=2), linear (p=1), high-order (p=5), discrete switch (epoch 50), rep_only, dyn_only
  - Failure penalty Delta_c: logarithmic (1 + log M_c), linear gap, direct ratio (M_c), none (Delta_c = 1)
  - Normalization & optimizer: with/without l1 trace norm under both Adam and SGD, plus with/without EMA
"""
import os, glob, tarfile, csv, time, math, random, warnings
from dataclasses import dataclass, replace
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ======================= EDIT THIS BLOCK PER NOTEBOOK ======================= #
CONFIG = dict(
    backbones    = ["resnet18"],   # 17 variants x 3 seeds on resnet18 takes ~5.0 hours total
    seeds        = [0, 1, 2],
    epochs       = 100,
    lr           = 1e-4,
    weight_decay = 0.0,
    img_size     = 32,
    tau_max      = 3.0,
    tau_step     = 0.05,
    use_amp      = False,
    out_dir      = "/kaggle/working/cifar10_ablation" if os.path.isdir("/kaggle/working") else "./cifar10_ablation",
)
# ============================================================================ #

NUM_CLASSES = 10
CIFAR10_SAMPLES = [5000, 5000, 5000, 5000, 5000, 500, 500, 500, 100, 100]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
BATCH_SIZES = {"squeezenet": 128, "squeezenet1_1": 128, "mobilenet_v2": 128, "resnet18": 128, "efficientnet_b0": 64, "densenet121": 32}
MEAN_T = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
STD_T = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def gpu_prep(x_u8, train=False, img_size=32):
    x = x_u8.to(DEVICE, non_blocking=True).float().div_(255.0)
    if train:
        B = x.size(0)
        x_pad = F.pad(x, (4, 4, 4, 4), mode="constant", value=0.0)
        dy = torch.randint(0, 9, (B, 1, 1, 1), device=DEVICE)
        dx = torch.randint(0, 9, (B, 1, 1, 1), device=DEVICE)
        iy = dy + torch.arange(32, device=DEVICE).view(1, 1, 32, 1)
        ix = dx + torch.arange(32, device=DEVICE).view(1, 1, 1, 32)
        ib = torch.arange(B, device=DEVICE).view(B, 1, 1, 1)
        ic = torch.arange(3, device=DEVICE).view(1, 3, 1, 1)
        x = x_pad[ib, ic, iy, ix]
        flip = torch.rand((B, 1, 1, 1), device=DEVICE) < 0.5
        x = torch.where(flip, x.flip(-1), x)
    x = (x - MEAN_T) / STD_T
    if img_size != 32:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


class GPUTensorLoader:
    def __init__(self, x_u8, y_long, batch_size, shuffle=False):
        self.x = x_u8.to(DEVICE)
        self.y = y_long.to(DEVICE)
        self.bs = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return (self.x.size(0) + self.bs - 1) // self.bs
    def __iter__(self):
        n = self.x.size(0)
        idx = torch.randperm(n, device=DEVICE) if self.shuffle else torch.arange(n, device=DEVICE)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            yield self.x[b], self.y[b]


@dataclass(frozen=True)
class V:
    name: str = "full"
    beta_rep: float = 1.0
    beta_pen: float = 0.5
    gate: str = "poly"        # poly | discrete_epoch | rep_only | dyn_only
    p: float = 2.0
    penalty: str = "log"      # log | linear | ratio | none
    l1norm: bool = True
    ema: bool = True
    eps: float = 1e-9
    optimizer: str = "adam"   # adam | sgd


def get_ablation_variants():
    f = V()
    c_exp = replace(f, name="cube_root_exponents(0.33->0.33)", beta_rep=0.33, beta_pen=0.33)
    return [
        f,
        c_exp,
        replace(f, name="only_W_rep(undampened,beta=1)", beta_rep=1.0, beta_pen=1.0, gate="rep_only", penalty="none"),
        replace(f, name="only_W_pen(sqrt,beta=0.5)", beta_rep=0.5, beta_pen=0.5, gate="rep_only", penalty="none"),
        replace(f, name="gate_linear(p=1)", p=1.0),
        replace(f, name="gate_high_order(p=5)", p=5.0),
        replace(f, name="gate_discrete(ts=50)", gate="discrete_epoch"),
        replace(c_exp, name="cuberoot_gate_linear(p=1)", p=1.0),
        replace(c_exp, name="cuberoot_gate_high_order(p=5)", p=5.0),
        replace(c_exp, name="cuberoot_gate_discrete(ts=50)", gate="discrete_epoch"),
        replace(f, name="penalty_none(Delta=1)", penalty="none"),
        replace(f, name="penalty_linear_gap", penalty="linear"),
        replace(f, name="penalty_ratio_Mc", penalty="ratio"),
        replace(f, name="no_l1norm(adam)", l1norm=False),
        replace(f, name="with_l1norm(sgd)", optimizer="sgd", l1norm=True),
        replace(f, name="no_l1norm(sgd)", optimizer="sgd", l1norm=False),
        replace(f, name="no_ema", ema=False),
    ]


def setup_cifar10_lt(batch_size, img_size=224):
    root, dl = "./data/cifar10", True
    dir_hits = glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)
    tar_hits = glob.glob("/kaggle/input/**/cifar-10-python.tar.gz", recursive=True)
    if dir_hits: root, dl = os.path.dirname(dir_hits[0]), False
    elif tar_hits:
        root = "/kaggle/working/cifar10_extracted"
        if not os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
            os.makedirs(root, exist_ok=True)
            with tarfile.open(tar_hits[0]) as t: t.extractall(root)
        dl = False
    full_train = tv_datasets.CIFAR10(root=root, train=True, download=dl)
    full_test = tv_datasets.CIFAR10(root=root, train=False, download=dl)
    targets = np.array(full_train.targets, dtype=np.int64)
    test_targets = np.array(full_test.targets, dtype=np.int64)
    selected = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(targets == c)[0]
        np.random.seed(42 + c)
        selected.extend(np.random.choice(cls_idx, size=min(CIFAR10_SAMPLES[c], len(cls_idx)), replace=False).tolist())
    np.random.seed(42); np.random.shuffle(selected)
    tr_idx, va_idx = train_test_split(selected, test_size=0.15, stratify=targets[selected], random_state=42)
    counts = np.array([int((targets[tr_idx] == c).sum()) for c in range(NUM_CLASSES)], dtype=np.float64)

    x_tr = torch.from_numpy(full_train.data[tr_idx]).permute(0, 3, 1, 2).contiguous()
    y_tr = torch.from_numpy(targets[tr_idx]).long()
    x_va = torch.from_numpy(full_train.data[va_idx]).permute(0, 3, 1, 2).contiguous()
    y_va = torch.from_numpy(targets[va_idx]).long()
    x_te = torch.from_numpy(full_test.data).permute(0, 3, 1, 2).contiguous()
    y_te = torch.from_numpy(test_targets).long()
    return {
        "train_loader": GPUTensorLoader(x_tr, y_tr, batch_size=batch_size, shuffle=True),
        "val_loader": GPUTensorLoader(x_va, y_va, batch_size=256, shuffle=False),
        "test_loader": GPUTensorLoader(x_te, y_te, batch_size=256, shuffle=False),
        "class_counts": counts,
    }


class BackboneWithFeatures(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x): return self.classifier(self.backbone(x))


def get_model(name):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        bb, clf = nn.Sequential(*list(m.children())[:-1], nn.Flatten()), nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        bb = nn.Sequential(m.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        clf = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        bb = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        clf = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        bb = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        clf = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    elif name in ("squeezenet", "squeezenet1_1"):
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        bb, clf = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), nn.Linear(512, NUM_CLASSES)
    else:
        raise ValueError(name)
    return BackboneWithFeatures(bb, clf)


class FlexWeights:
    def __init__(self, counts, v: V):
        self.v, self.c = v, len(counts)
        ratio = counts.max() / np.maximum(counts, 1.0)
        self.w_rep, self.w_pen = ratio ** v.beta_rep, ratio ** v.beta_pen
        self.w_dyn, self.init = self.w_pen.copy(), False
        self.f1_smooth = np.ones(self.c)
    def _norm(self, w):
        return self.c * w / (w.sum() + 1e-9) if self.v.l1norm else w.copy()
    def update(self, f1):
        v = self.v; self.init = True
        mult = np.ones(self.c, dtype=np.float64)
        if f1 is not None and v.penalty != "none":
            if v.ema:
                self.f1_smooth = 0.9 * self.f1_smooth + 0.1 * f1
            else:
                self.f1_smooth = f1.copy()
            s, smax = self.f1_smooth, self.f1_smooth.max()
            if smax > 0:
                M_c = (smax + v.eps) / (s + v.eps)
                if v.penalty == "log": mult = 1.0 + np.log(np.maximum(M_c, 1.0))
                elif v.penalty == "linear": mult = 1.0 + (smax - s) / (smax + v.eps)
                elif v.penalty == "ratio": mult = M_c
        self.w_dyn = self.w_pen * mult
    def get(self, gamma):
        if not self.init: return self._norm(self.w_rep)
        return self._norm((1.0 - gamma) * self.w_rep + gamma * self.w_dyn)


def gate_value(v: V, A: float, ep: int, E: int) -> float:
    if v.gate == "poly": return float(A) ** v.p
    if v.gate == "discrete_epoch": return 1.0 if ep > (E // 2) else 0.0
    if v.gate == "rep_only": return 0.0
    if v.gate == "dyn_only": return 1.0
    raise ValueError(v.gate)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def softmax_np(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

def per_class_f1(t, p, sw=None):
    if sw is None:
        return f1_score(t, p, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    np.add.at(cm, (t, p), sw)
    tp = np.diag(cm)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    return np.where(denom > 0, 2.0 * tp / denom, 0.0)

def ece_np(probs, t, n_bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins, ece = np.linspace(0.0, 1.0, n_bins + 1), 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any(): ece += m.mean() * abs((pred[m] == t[m]).mean() - conf[m].mean())
    return float(ece)

def compute_metrics(logits, t):
    probs = softmax_np(logits); pred = probs.argmax(axis=1)
    f1 = per_class_f1(t, pred)
    return dict(f1=float(f1.mean()), acc=float((pred == t).mean()), ece=float(ece_np(probs, t)),
                head_f1=float(f1[:5].mean()), med_f1=float(f1[5:8].mean()), tail_f1=float(f1[8:].mean()))

@torch.no_grad()
def collect(model, loader):
    model.eval(); L, T = [], []
    for x, y in loader:
        with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
            L.append(model(gpu_prep(x, train=False, img_size=CONFIG["img_size"])).float().cpu().numpy())
        T.append(y.cpu().numpy())
    return np.concatenate(L), np.concatenate(T)


def train_ablation(v: V, seed: int, backbone: str, data, tau_grid):
    E = CONFIG["epochs"]
    set_seed(seed)
    model = get_model(backbone).to(DEVICE)
    counts = data["class_counts"]
    wm = FlexWeights(counts, v)
    if v.penalty == "none": wm.update(None)
    opt = (optim.SGD(model.parameters(), lr=CONFIG["lr"] * 10.0, momentum=0.9, weight_decay=CONFIG["weight_decay"])
           if v.optimizer == "sgd" else optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda")
    best_f1, best_state, acc_hist = -1.0, None, [0.0]

    for ep in range(1, E + 1):
        model.train()
        w_active = torch.as_tensor(wm.get(gate_value(v, acc_hist[-1], ep, E)), dtype=torch.float32, device=DEVICE)
        tp, tt = [], []
        for x, y_dev in data["train_loader"]:
            x_dev = gpu_prep(x, train=True, img_size=CONFIG["img_size"])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
                out = model(x_dev)
                if v.l1norm:
                    loss = F.cross_entropy(out, y_dev, weight=w_active)
                else:
                    loss = (F.cross_entropy(out, y_dev, reduction="none") * w_active[y_dev]).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tp.append(out.argmax(1).detach().cpu()); tt.append(y_dev.cpu())
        sched.step()
        tp_np, tt_np = torch.cat(tp).numpy(), torch.cat(tt).numpy()
        acc_hist.append(float((tp_np == tt_np).mean()))
        if v.penalty != "none": wm.update(per_class_f1(tt_np, tp_np))
        Lv, Tv = collect(model, data["val_loader"])
        fv = float(per_class_f1(Tv, Lv.argmax(1)).mean())
        if fv > best_f1:
            best_f1 = fv
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == E:
            print(f"    [HALO/{v.name}] ep {ep:3d}/{E} | best_val_f1={best_f1:.4f}", flush=True)

    model.load_state_dict(best_state)
    Lv, Tv = collect(model, data["val_loader"])
    Lt, Tt = collect(model, data["test_loader"])
    log_prior = np.log(counts / counts.sum() + 1e-12)
    sw = 1.0 / (np.exp(log_prior)[Tv] + 1e-9)
    best_tau, best_vf1 = 0.0, -1.0
    for tau in tau_grid:
        f = per_class_f1(Tv, (Lv - tau * log_prior).argmax(1), sw=sw).mean()
        if f > best_vf1: best_vf1, best_tau = float(f), float(tau)
    return [
        ("raw", 0.0, compute_metrics(Lt, Tt)),
        ("tau", best_tau, compute_metrics(Lt - best_tau * log_prior, Tt)),
    ], best_f1


def summarize(csv_path, latex_path):
    grouped = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            grouped[(r["backbone"], r["variant"], r["stage"])][int(r["seed"])] = {k: float(r[k]) for k in ("f1", "acc", "ece", "tail_f1", "tau")}
    fmt = lambda v: "--" if not v else (f"{v[0]:.4f}" if len(v) == 1 else f"{np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
    tex = ["% Script 2: CIFAR-10-LT Ablation Study (Raw vs +tau*)"]
    for bb in sorted({k[0] for k in grouped}):
        print(f"\n{'='*110}\nCIFAR-10-LT Ablation Grid | Backbone: {bb}\n{'='*110}")
        print(f"{'Ablation Variant':<34} | {'Raw F1 (tau=0)':<19} | {'+tau* F1':<19} | {'Raw ECE':<10} | {'+tau* ECE':<10}")
        print("-" * 110)
        tex += [f"\n% {bb}", "\\begin{tabular}{l|cc|cc}\\toprule",
                "Ablation Variant & Raw F1 ($\\tau{=}0$) & $+\\tau^\\star$ F1 & Raw ECE & $+\\tau^\\star$ ECE \\\\ \\midrule"]
        for vname in [v.name for v in get_ablation_variants()]:
            rd, td = grouped.get((bb, vname, "raw"), {}), grouped.get((bb, vname, "tau"), {})
            if not rd and not td: continue
            rf1, tf1 = [rd[s]["f1"] for s in sorted(rd)], [td[s]["f1"] for s in sorted(td)]
            rece, tece = [rd[s]["ece"] for s in sorted(rd)], [td[s]["ece"] for s in sorted(td)]
            re_s, te_s = (f"{np.mean(rece):.4f}" if rece else "--"), (f"{np.mean(tece):.4f}" if tece else "--")
            print(f"{vname:<34} | {fmt(rf1):<19} | {fmt(tf1):<19} | {re_s:<10} | {te_s:<10}")
            tex.append(f"{vname.replace('_', '\\_')} & {fmt(rf1).replace('+/-', '$\\pm$')} & {fmt(tf1).replace('+/-', '$\\pm$')} & {re_s} & {te_s} \\\\")
        tex.append("\\bottomrule\\end{tabular}")
    with open(latex_path, "w", encoding="utf-8") as f: f.write("\n".join(tex) + "\n")
    print(f"\nSaved LaTeX ablation table to: {latex_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("NO GPU DETECTED (Accelerator is 'None')! In Kaggle right sidebar -> Session Options -> Accelerator -> select 'GPU T4 x2' or 'GPU P100' before running.")
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}", flush=True)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    out_csv = os.path.join(CONFIG["out_dir"], "ablation_runs.csv")
    fields = ["dataset", "backbone", "seed", "variant", "stage", "f1", "acc", "ece", "head_f1", "med_f1", "tail_f1", "tau", "sel_f1", "epochs", "seconds"]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f): done.add((r["backbone"], r["seed"], r["variant"]))
    else:
        with open(out_csv, "w", newline="") as f: csv.DictWriter(f, fields).writeheader()

    tau_grid = np.round(np.arange(0.0, CONFIG["tau_max"] + 1e-9, CONFIG["tau_step"]), 4)
    variants = get_ablation_variants()
    for bb in CONFIG["backbones"]:
        data = setup_cifar10_lt(BATCH_SIZES.get(bb, 64), CONFIG["img_size"])
        for seed in CONFIG["seeds"]:
            for v in variants:
                if (bb, str(seed), v.name) in done: continue
                print(f"\n=== [Script 2: Ablation] CIFAR-10-LT | {bb} | seed {seed} | {v.name} ===", flush=True)
                t0 = time.time()
                rows, sel_f1 = train_ablation(v, seed, bb, data, tau_grid)
                el = round(time.time() - t0, 1)
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fields)
                    for stg, pval, met in rows:
                        w.writerow(dict(dataset="cifar10_lt", backbone=bb, seed=seed, variant=v.name, stage=stg,
                                        tau=pval, sel_f1=round(sel_f1, 6), epochs=CONFIG["epochs"], seconds=el, **met))
                done.add((bb, str(seed), v.name))
                print("   " + " | ".join(f"{s}: F1={met['f1']:.4f} ECE={met['ece']:.4f} ({p:.2f})" for s, p, met in rows), flush=True)
    summarize(out_csv, os.path.join(CONFIG["out_dir"], "ablation_tables.tex"))

if __name__ == "__main__":
    main()
