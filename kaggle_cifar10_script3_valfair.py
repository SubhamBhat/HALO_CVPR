"""
SCRIPT 3: CIFAR-10-LT Validation-Fairness & 100%-Data Suite (Single Kaggle GPU Cell)
====================================================================================
Directly answers the Area Chair, Reviewer mtBX, and Reviewer VuMh:
  - Compares train-F1, train-CE-loss, 5-fold OOF-F1, and disjoint split-val feedback.
  - Evaluates standard 85/15, 50/50 disjoint Split-Val, Zero-Val-Labels (last ckpt, tau=1),
    and 100% Full-Data training for both HALO and key baselines.
"""
import os, glob, tarfile, csv, time, math, random, warnings
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
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
    backbones    = ["resnet18"],   # 12 runs x 3 seeds on resnet18 takes ~3.6 hours total
    seeds        = [0, 1, 2],
    epochs       = 100,
    lr           = 1e-4,
    weight_decay = 0.0,
    img_size     = 32,
    tau_max      = 3.0,
    tau_step     = 0.05,
    use_amp      = False,
    out_dir      = "/kaggle/working/cifar10_valfair" if os.path.isdir("/kaggle/working") else "./cifar10_valfair",
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
class RunSpec:
    method: str
    variant: str
    feedback: str = "train"   # train | train_loss | oof5 | val_halfA
    full_data: bool = False


VALFAIR_PLAN = [
    RunSpec("HALO", "train_fb", feedback="train", full_data=False),
    RunSpec("HALO", "train_loss_fb", feedback="train_loss", full_data=False),
    RunSpec("HALO", "oof5_fb", feedback="oof5", full_data=False),
    RunSpec("HALO", "val_halfA_fb", feedback="val_halfA", full_data=False),
    RunSpec("HALO", "100pct_data(train_fb)", feedback="train", full_data=True),
    RunSpec("HALO", "100pct_data(loss_fb)", feedback="train_loss", full_data=True),
    RunSpec("CE", "100pct_data", full_data=True),
    RunSpec("WCE", "100pct_data", full_data=True),
    RunSpec("CB", "100pct_data", full_data=True),
    RunSpec("LA", "100pct_data", full_data=True),
    RunSpec("BalancedSoftmax", "100pct_data", full_data=True),
    RunSpec("LDAM-DRW", "100pct_data", full_data=True),
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
    full_counts = np.array([int((targets[selected] == c).sum()) for c in range(NUM_CLASSES)], dtype=np.float64)

    va_targets = targets[va_idx]
    rng = np.random.default_rng(42)
    half_A, half_B = [], []
    for c in range(NUM_CLASSES):
        pos = rng.permutation(np.where(va_targets == c)[0]).tolist()
        k = len(pos) // 2
        half_A.extend(pos[:k]); half_B.extend(pos[k:])
    half_A, half_B = np.array(sorted(half_A)), np.array(sorted(half_B))

    x_tr = torch.from_numpy(full_train.data[tr_idx]).permute(0, 3, 1, 2).contiguous()
    y_tr = torch.from_numpy(targets[tr_idx]).long()
    x_full = torch.from_numpy(full_train.data[selected]).permute(0, 3, 1, 2).contiguous()
    y_full = torch.from_numpy(targets[selected]).long()
    x_va = torch.from_numpy(full_train.data[va_idx]).permute(0, 3, 1, 2).contiguous()
    y_va = torch.from_numpy(targets[va_idx]).long()
    x_te = torch.from_numpy(full_test.data).permute(0, 3, 1, 2).contiguous()
    y_te = torch.from_numpy(test_targets).long()

    return {
        "train_loader": GPUTensorLoader(x_tr, y_tr, batch_size=batch_size, shuffle=True),
        "full_train_loader": GPUTensorLoader(x_full, y_full, batch_size=batch_size, shuffle=True),
        "val_loader": GPUTensorLoader(x_va, y_va, batch_size=256, shuffle=False),
        "test_loader": GPUTensorLoader(x_te, y_te, batch_size=256, shuffle=False),
        "val_half_A": half_A, "val_half_B": half_B,
        "class_counts": counts, "full_counts": full_counts,
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


class WeightManager:
    def __init__(self, counts):
        self.c = len(counts)
        self.w_rep = counts.max() / counts
        self.w_pen = np.sqrt(counts.max() / counts)
        self.w_dyn = self.w_pen.copy()
        self.sig_smooth = np.ones(self.c) / self.c
        self.sigma2 = np.ones(self.c) * 0.01
        self.init = False
    def _norm(self, w): return self.c * w / (w.sum() + 1e-9)
    def update(self, sig, is_loss=False):
        self.init = True
        sig = np.asarray(sig, dtype=np.float64)
        inv = sig - self.sig_smooth
        self.sigma2 = 0.9 * self.sigma2 + 0.1 * (inv ** 2)
        lr = 0.1 / (0.1 + self.sigma2 / (self.sigma2.mean() + 1e-9))
        self.sig_smooth = self.sig_smooth + lr * inv
        s = self.sig_smooth
        if is_loss:
            M_c = (s + 1e-9) / (s.min() + 1e-9)
        else:
            smax = s.max()
            M_c = (smax + 1e-9) / (s + 1e-9) if smax > 0 else np.ones(self.c)
        self.w_dyn = self.w_pen * (1.0 + np.log(np.maximum(M_c, 1.0)))
    def get(self, gamma):
        if not self.init: return self._norm(self.w_rep)
        return self._norm((1.0 - gamma) * self.w_rep + gamma * self.w_dyn)


def make_oof_folds(train_loader, k_folds, seed, batch_size):
    x_all, y_all = train_loader.x, train_loader.y
    labels = y_all.cpu().numpy()
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k_folds)]
    for c in range(NUM_CLASSES):
        for i, ix in enumerate(rng.permutation(np.where(labels == c)[0])):
            folds[i % k_folds].append(int(ix))
    all_idx, pairs = set(range(len(labels))), []
    for k in range(k_folds):
        oof_idx = sorted(folds[k]); sub_tr = sorted(all_idx - set(oof_idx))
        pairs.append((GPUTensorLoader(x_all[sub_tr], y_all[sub_tr], batch_size=batch_size, shuffle=True),
                      GPUTensorLoader(x_all[oof_idx], y_all[oof_idx], batch_size=256, shuffle=False)))
    return pairs


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


def train_valfair(spec: RunSpec, seed: int, backbone: str, data, tau_grid):
    E = CONFIG["epochs"]
    bs = BATCH_SIZES.get(backbone, 64)
    set_seed(seed)
    model = get_model(backbone).to(DEVICE)
    counts = data["full_counts"] if spec.full_data else data["class_counts"]
    is_halo = (spec.method == "HALO")
    wm = WeightManager(counts) if is_halo else None

    pi = counts / counts.sum()
    log_pi_t = torch.tensor(np.log(pi + 1e-12), dtype=torch.float32, device=DEVICE)
    log_n_t = torch.tensor(np.log(np.maximum(counts, 1.0)), dtype=torch.float32, device=DEVICE)
    wce_w = torch.tensor(counts.max() / counts, dtype=torch.float32, device=DEVICE)
    cb_w = (1.0 - 0.9999) / (1.0 - np.power(0.9999, counts) + 1e-8)
    cb_w_t = torch.tensor(cb_w / cb_w.sum() * NUM_CLASSES, dtype=torch.float32, device=DEVICE)
    ldam_m = 1.0 / np.sqrt(np.sqrt(counts)); ldam_m = torch.tensor(ldam_m * (0.5 / ldam_m.max()), dtype=torch.float32, device=DEVICE)

    base_loader = data["full_train_loader"] if spec.full_data else data["train_loader"]
    oof_pairs = make_oof_folds(data["train_loader"], 5, seed, bs) if (is_halo and spec.feedback == "oof5") else None
    drw_start = int(0.8 * E)
    opt = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda")

    best_full, best_halfA = -1.0, -1.0
    state_full, state_halfA = None, None
    half_A, half_B = data["val_half_A"], data["val_half_B"]
    acc_hist = [0.0]

    for ep in range(1, E + 1):
        model.train()
        w_active = torch.as_tensor(wm.get(acc_hist[-1] ** 2), dtype=torch.float32, device=DEVICE) if is_halo else None
        act_dl = oof_pairs[(ep - 1) % 5][0] if oof_pairs is not None else base_loader
        tp, tt, t_losses = [], [], []
        for x, y_dev in act_dl:
            x_dev = gpu_prep(x, train=True, img_size=CONFIG["img_size"])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
                out = model(x_dev)
                if is_halo:
                    loss = F.cross_entropy(out, y_dev, weight=w_active)
                elif spec.method == "CE": loss = F.cross_entropy(out, y_dev)
                elif spec.method == "WCE": loss = F.cross_entropy(out, y_dev, weight=wce_w)
                elif spec.method == "CB": loss = F.cross_entropy(out, y_dev, weight=cb_w_t)
                elif spec.method == "LA": loss = F.cross_entropy(out + log_pi_t, y_dev)
                elif spec.method == "BalancedSoftmax": loss = F.cross_entropy(out + log_n_t, y_dev)
                elif spec.method == "LDAM-DRW":
                    oh = F.one_hot(y_dev, out.size(1)).bool()
                    loss = F.cross_entropy(30.0 * torch.where(oh, out - ldam_m[y_dev][:, None], out), y_dev,
                                           weight=cb_w_t if ep > drw_start else None)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if is_halo:
                tp.append(out.argmax(1).detach().cpu()); tt.append(y_dev.cpu())
                if spec.feedback == "train_loss":
                    with torch.no_grad(): t_losses.append(F.cross_entropy(out.detach().float(), y_dev, reduction="none").cpu())
        sched.step()

        if is_halo:
            tp_np, tt_np = torch.cat(tp).numpy(), torch.cat(tt).numpy()
            acc_hist.append(float((tp_np == tt_np).mean()))
            if spec.feedback == "train": wm.update(per_class_f1(tt_np, tp_np))
            elif spec.feedback == "train_loss":
                lnp = torch.cat(t_losses).numpy()
                wm.update([lnp[tt_np == c].mean() if (tt_np == c).any() else 0.0 for c in range(NUM_CLASSES)], is_loss=True)
            elif spec.feedback == "oof5":
                Lo, To = collect(model, oof_pairs[(ep - 1) % 5][1])
                wm.update(per_class_f1(To, Lo.argmax(1)))

        if not spec.full_data:
            Lv, Tv = collect(model, data["val_loader"])
            if is_halo and spec.feedback == "val_halfA":
                wm.update(per_class_f1(Tv[half_A], Lv[half_A].argmax(1)))
            f_full = float(per_class_f1(Tv, Lv.argmax(1)).mean())
            f_halfA = float(per_class_f1(Tv[half_A], Lv[half_A].argmax(1)).mean())
            if f_full > best_full:
                best_full = f_full; state_full = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if f_halfA > best_halfA:
                best_halfA = f_halfA; state_halfA = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == E:
            print(f"    [{spec.method}/{spec.variant}] ep {ep:3d}/{E} | val_best_f1={best_full:.4f}", flush=True)

    log_prior = np.log(counts / counts.sum() + 1e-12)
    L_last, T_test = collect(model, data["test_loader"])
    if spec.full_data:
        return [
            ("raw_100pct", 0.0, compute_metrics(L_last, T_test)),
            ("tau1_100pct", 1.0, compute_metrics(L_last - 1.0 * log_prior, T_test)),
        ], float("nan")

    model.load_state_dict(state_halfA)
    L_halfA_test, _ = collect(model, data["test_loader"])
    model.load_state_dict(state_full)
    L_val, T_val = collect(model, data["val_loader"])
    L_test, _ = collect(model, data["test_loader"])

    sw = 1.0 / (np.exp(log_prior)[T_val] + 1e-9)
    best_tau, best_vf1 = 0.0, -1.0
    for tau in tau_grid:
        f = per_class_f1(T_val, (L_val - tau * log_prior).argmax(1), sw=sw).mean()
        if f > best_vf1: best_vf1, best_tau = float(f), float(tau)
    sw_B = 1.0 / (np.exp(log_prior)[T_val[half_B]] + 1e-9)
    best_tau_B, best_vf1_B = 0.0, -1.0
    for tau in tau_grid:
        f = per_class_f1(T_val[half_B], (L_val[half_B] - tau * log_prior).argmax(1), sw=sw_B).mean()
        if f > best_vf1_B: best_vf1_B, best_tau_B = float(f), float(tau)

    return [
        ("raw", 0.0, compute_metrics(L_test, T_test)),
        ("tau", best_tau, compute_metrics(L_test - best_tau * log_prior, T_test)),
        ("split_val", best_tau_B, compute_metrics(L_halfA_test - best_tau_B * log_prior, T_test)),
        ("no_val_labels", 1.0, compute_metrics(L_last - 1.0 * log_prior, T_test)),
    ], best_full


def summarize(csv_path, latex_path):
    grouped = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            grouped[(r["backbone"], r["method"], r["variant"], r["stage"])][int(r["seed"])] = {k: float(r[k]) for k in ("f1", "acc", "ece", "tail_f1", "tau")}
    fmt = lambda v: "--" if not v else (f"{v[0]:.4f}" if len(v) == 1 else f"{np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
    tex = ["% Script 3: CIFAR-10-LT Validation-Fairness & 100%-Data Table"]
    for bb in sorted({k[0] for k in grouped}):
        print(f"\n{'='*118}\nCIFAR-10-LT Validation-Fairness | Backbone: {bb}\n{'='*118}")
        print(f"{'Method [Variant]':<32} | {'Raw F1':<19} | {'+tau* F1':<19} | {'Split-Val F1':<19} | {'No-Val(tau=1) F1':<19}")
        print("-" * 118)
        tex += [f"\n% {bb}", "\\begin{tabular}{l|cccc}\\toprule",
                "Method [Variant] & Raw F1 ($\\tau{=}0$) & $+\\tau^\\star$ F1 & Split-Val (50/50) F1 & No-Val ($\\tau{=}1$) F1 \\\\ \\midrule"]
        for spec in VALFAIR_PLAN:
            m, v = spec.method, spec.variant
            if spec.full_data:
                rd = grouped.get((bb, m, v, "raw_100pct"), {})
                td = grouped.get((bb, m, v, "tau1_100pct"), {})
                rf1, nf1 = [rd[s]["f1"] for s in sorted(rd)], [td[s]["f1"] for s in sorted(td)]
                sf1, tf1 = [], []
            else:
                rd = grouped.get((bb, m, v, "raw"), {})
                td = grouped.get((bb, m, v, "tau"), {})
                sd = grouped.get((bb, m, v, "split_val"), {})
                nd = grouped.get((bb, m, v, "no_val_labels"), {})
                rf1, tf1 = [rd[s]["f1"] for s in sorted(rd)], [td[s]["f1"] for s in sorted(td)]
                sf1, nf1 = [sd[s]["f1"] for s in sorted(sd)], [nd[s]["f1"] for s in sorted(nd)]
            lbl = f"{m} [{v}]"
            print(f"{lbl:<32} | {fmt(rf1):<19} | {fmt(tf1):<19} | {fmt(sf1):<19} | {fmt(nf1):<19}")
            tex.append(f"{lbl.replace('_', '\\_')} & {fmt(rf1).replace('+/-', '$\\pm$')} & {fmt(tf1).replace('+/-', '$\\pm$')} & "
                       f"{fmt(sf1).replace('+/-', '$\\pm$')} & {fmt(nf1).replace('+/-', '$\\pm$')} \\\\")
        tex.append("\\bottomrule\\end{tabular}")
    with open(latex_path, "w", encoding="utf-8") as f: f.write("\n".join(tex) + "\n")
    print(f"\nSaved LaTeX validation-fairness table to: {latex_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("NO GPU DETECTED (Accelerator is 'None')! In Kaggle right sidebar -> Session Options -> Accelerator -> select 'GPU T4 x2' or 'GPU P100' before running.")
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}", flush=True)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    out_csv = os.path.join(CONFIG["out_dir"], "valfair_runs.csv")
    fields = ["dataset", "backbone", "seed", "method", "variant", "stage", "f1", "acc", "ece", "head_f1", "med_f1", "tail_f1", "tau", "sel_f1", "epochs", "seconds"]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f): done.add((r["backbone"], r["seed"], r["method"], r["variant"]))
    else:
        with open(out_csv, "w", newline="") as f: csv.DictWriter(f, fields).writeheader()

    tau_grid = np.round(np.arange(0.0, CONFIG["tau_max"] + 1e-9, CONFIG["tau_step"]), 4)
    for bb in CONFIG["backbones"]:
        data = setup_cifar10_lt(BATCH_SIZES.get(bb, 64), CONFIG["img_size"])
        for seed in CONFIG["seeds"]:
            for spec in VALFAIR_PLAN:
                key = (bb, str(seed), spec.method, spec.variant)
                if key in done: continue
                print(f"\n=== [Script 3: Val-Fairness] CIFAR-10-LT | {bb} | seed {seed} | {spec.method} [{spec.variant}] ===", flush=True)
                t0 = time.time()
                rows, sel_f1 = train_valfair(spec, seed, bb, data, tau_grid)
                el = round(time.time() - t0, 1)
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fields)
                    for stg, pval, met in rows:
                        w.writerow(dict(dataset="cifar10_lt", backbone=bb, seed=seed, method=spec.method, variant=spec.variant,
                                        stage=stg, tau=pval, sel_f1=round(sel_f1, 6) if not np.isnan(sel_f1) else "",
                                        epochs=CONFIG["epochs"], seconds=el, **met))
                done.add(key)
                print("   " + " | ".join(f"{s}: F1={met['f1']:.4f} ECE={met['ece']:.4f} ({p:.2f})" for s, p, met in rows), flush=True)
    summarize(out_csv, os.path.join(CONFIG["out_dir"], "valfair_tables.tex"))

if __name__ == "__main__":
    main()
