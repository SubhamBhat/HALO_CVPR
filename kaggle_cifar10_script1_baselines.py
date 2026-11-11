"""
SCRIPT 1: CIFAR-10-LT Fair Baselines + Shared Post-Hoc Calibration (Single Kaggle GPU Cell)
===========================================================================================
Runs all 15 methods (CE, WCE, CB, Focal, LDAM, LDAM-DRW, DBM, ALPA, RobustFocal, DCAL, LA,
BalancedSoftmax, cRT, LWS, and HALO) and evaluates every method under:
  1. "raw"        : Uncalibrated test logits (tau = 0)
  2. "tau"        : Shared post-hoc prior calibration (tau* searched on validation Macro-F1)
  3. "temp_scale" : Post-hoc NLL temperature scaling (T* searched on validation NLL for fair ECE)
"""
import os, glob, tarfile, csv, time, math, copy, random, warnings
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None

warnings.filterwarnings("ignore")

# ======================= EDIT THIS BLOCK PER NOTEBOOK ======================= #
CONFIG = dict(
    backbones     = ["resnet18", "mobilenet_v2"],  # NB1: ["resnet18","mobilenet_v2"], NB2: ["efficientnet_b0","squeezenet"], NB3: ["densenet121"]
    seeds         = [0, 1, 2],                     # Or set backbones=all 5 and run seeds=[0] in NB1, [1] in NB2, [2] in NB3
    methods       = ["CE", "WCE", "CB", "Focal", "LDAM", "LDAM-DRW", "DBM", "ALPA",
                     "RobustFocal", "DCAL", "LA", "BalancedSoftmax", "HALO"],  # cRT & LWS run automatically after CE
    epochs        = 100,
    stage2_epochs = 30,
    lr            = 1e-4,
    weight_decay  = 0.0,
    img_size      = 32,
    tau_max       = 3.0,
    tau_step      = 0.05,
    use_amp       = False,
    out_dir       = "/kaggle/working/cifar10_baselines" if os.path.isdir("/kaggle/working") else "./cifar10_baselines",
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


def setup_cifar10_lt(batch_size, img_size=224):
    root, dl = "./data/cifar10", True
    dir_hits = glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)
    tar_hits = glob.glob("/kaggle/input/**/cifar-10-python.tar.gz", recursive=True)
    if dir_hits:
        root, dl = os.path.dirname(dir_hits[0]), False
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
        bb = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        clf = nn.Linear(512, NUM_CLASSES)
    else:
        raise ValueError(name)
    return BackboneWithFeatures(bb, clf)


class CrossEntropyLossBaseline(nn.Module):
    def __init__(self, c): super().__init__(); self.crit = nn.CrossEntropyLoss()
    def forward(self, z, y): return self.crit(z, y)

class WCELoss(nn.Module):
    def __init__(self, c): super().__init__(); self.crit = nn.CrossEntropyLoss(weight=torch.tensor(c.max() / c, dtype=torch.float32, device=DEVICE))
    def forward(self, z, y): return self.crit(z, y)

class ClassBalancedLoss(nn.Module):
    def __init__(self, c, beta=0.9999):
        super().__init__()
        w = (1.0 - beta) / (1.0 - np.power(beta, c) + 1e-8)
        self.crit = nn.CrossEntropyLoss(weight=torch.tensor(w / w.sum() * len(c), dtype=torch.float32, device=DEVICE))
    def forward(self, z, y): return self.crit(z, y)

class FocalLoss(nn.Module):
    def __init__(self, c, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, z, y):
        ce = F.cross_entropy(z, y, reduction="none")
        return (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()

class LDAMLoss(nn.Module):
    def __init__(self, c, max_m=0.5, s=30.0):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(c)); m = m * (max_m / m.max())
        self.m = torch.tensor(m, dtype=torch.float32, device=DEVICE); self.s = s
    def forward(self, z, y):
        oh = F.one_hot(y, z.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(oh, z - self.m[y][:, None], z), y)

class LDAMDRWLoss(nn.Module):
    def __init__(self, c, max_m=0.5, s=30.0, beta=0.9999):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(c)); m = m * (max_m / m.max())
        w = (1.0 - beta) / (1.0 - np.power(beta, c) + 1e-8); w = w / w.sum() * len(c)
        self.m = torch.tensor(m, dtype=torch.float32, device=DEVICE)
        self.w = torch.tensor(w, dtype=torch.float32, device=DEVICE); self.s = s
    def forward(self, z, y, drw=False):
        oh = F.one_hot(y, z.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(oh, z - self.m[y][:, None], z), y, weight=self.w if drw else None)

class DBMLoss(nn.Module):
    def __init__(self, c, scale=30.0, base_m=0.5):
        super().__init__()
        self.mc = torch.tensor(base_m * (c.max() / c), dtype=torch.float32, device=DEVICE); self.s = scale
    def forward(self, z, y):
        p = F.softmax(z, dim=1).gather(1, y.view(-1, 1)).view(-1)
        margin = self.mc[y] + (1.0 - p).detach() * 0.2
        oh = F.one_hot(y, z.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(oh, z - margin[:, None], z), y)

class ALPALoss(nn.Module):
    def __init__(self, c, a0=1.0, a1=2.0, b1=1.5): super().__init__(); self.a0, self.a1, self.b1 = a0, a1, b1
    def forward(self, z, y):
        ce = F.cross_entropy(z, y, reduction="none"); pt = torch.exp(-ce)
        return (((self.a0 + self.a1 * (1 - pt)) / (1 + self.b1 * (1 - pt))) * ce).mean()

class RobustFocalLoss(nn.Module):
    def __init__(self, c, gamma=2.0, margin=0.2): super().__init__(); self.gamma, self.margin = gamma, margin
    def forward(self, z, y):
        tp = F.softmax(z, dim=1).gather(1, y.view(-1, 1)).view(-1)
        ce = F.cross_entropy(z, y, reduction="none")
        return (((1 - tp) ** self.gamma) * (tp < (1 - self.margin)).float() * ce).mean()

class PublishedDCALoss(nn.Module):
    def __init__(self, c, m=0.3, s=30.0, lambda_mdca=0.1):
        super().__init__(); self.m, self.s, self.lambda_mdca = m, s, lambda_mdca
    def forward(self, z, y, w_active=None):
        cos = z / torch.norm(z, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        oh = F.one_hot(y, z.size(1)).bool()
        ce = F.cross_entropy(torch.where(oh, cos - self.m, cos) * self.s, y, weight=w_active)
        probs = F.softmax(z, dim=1); conf, preds = probs.max(1)[0], probs.argmax(1)
        corr = (preds == y).float(); mdca, uniq = 0.0, torch.unique(y)
        for c in uniq:
            m = y == c
            w = w_active[c] if w_active is not None else 1.0
            mdca += w * torch.abs(conf[m].mean() - corr[m].mean())
        return ce + self.lambda_mdca * (mdca / max(1, len(uniq)))

class LogitAdjustedLoss(nn.Module):
    def __init__(self, c, tau=1.0):
        super().__init__()
        pi = c / c.sum()
        self.off = torch.tensor(tau * np.log(pi + 1e-12), dtype=torch.float32, device=DEVICE)
    def forward(self, z, y): return F.cross_entropy(z + self.off, y)

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.log_n = torch.tensor(np.log(np.maximum(c, 1.0)), dtype=torch.float32, device=DEVICE)
    def forward(self, z, y): return F.cross_entropy(z + self.log_n, y)

class LWSHead(nn.Module):
    def __init__(self, clf):
        super().__init__(); self.clf = clf
        for p in self.clf.parameters(): p.requires_grad_(False)
        self.scale = nn.Parameter(torch.ones(NUM_CLASSES))
    def forward(self, f): return self.clf(f) * self.scale

class DynamicWeightManager:
    def __init__(self, counts):
        self.c = len(counts)
        self.w_rep = counts.max() / counts
        self.w_pen = np.sqrt(counts.max() / counts)
        self.w_dyn = self.w_pen.copy()
        self.f1_smooth = np.ones(self.c)
        self.init = False
    def _norm(self, w): return self.c * w / (w.sum() + 1e-9)
    def update(self, f1):
        self.init = True
        self.f1_smooth = 0.9 * self.f1_smooth + 0.1 * f1
        eps = 1e-9
        M_c = (self.f1_smooth.max() + eps) / (self.f1_smooth + eps)
        mult = 1.0 + np.log(np.maximum(M_c, 1.0))
        self.w_dyn = self.w_pen * mult
    def get(self, gamma):
        if not self.init: return self._norm(self.w_rep)
        return self._norm((1.0 - gamma) * self.w_rep + gamma * self.w_dyn)

LOSS_MAP = {
    "CE": CrossEntropyLossBaseline, "WCE": WCELoss, "CB": ClassBalancedLoss, "Focal": FocalLoss,
    "LDAM": LDAMLoss, "LDAM-DRW": LDAMDRWLoss, "DBM": DBMLoss, "ALPA": ALPALoss,
    "RobustFocal": RobustFocalLoss, "DCAL": PublishedDCALoss, "LA": LogitAdjustedLoss,
    "BalancedSoftmax": BalancedSoftmaxLoss,
}


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

def search_tau(val_logits, val_t, log_prior, tau_grid):
    sw = 1.0 / (np.exp(log_prior)[val_t] + 1e-9)
    best_tau, best_f1 = 0.0, -1.0
    for tau in tau_grid:
        f = per_class_f1(val_t, (val_logits - tau * log_prior).argmax(axis=1), sw=sw).mean()
        if f > best_f1: best_f1, best_tau = float(f), float(tau)
    return best_tau, best_f1

def search_temperature(val_logits, val_t):
    z_t, y_t = torch.tensor(val_logits, dtype=torch.float32), torch.tensor(val_t, dtype=torch.long)
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.arange(0.2, 2.0, 0.05), np.arange(2.0, 5.05, 0.25)]):
        nll = F.cross_entropy(z_t / float(T), y_t).item()
        if nll < best_nll: best_nll, best_T = nll, float(T)
    return best_T

@torch.no_grad()
def collect(model, loader):
    model.eval(); L, T = [], []
    for x, y in loader:
        with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
            L.append(model(gpu_prep(x, train=False, img_size=CONFIG["img_size"])).float().cpu().numpy())
        T.append(y.cpu().numpy())
    return np.concatenate(L), np.concatenate(T)

@torch.no_grad()
def collect_feats(model, loader):
    model.eval(); X, T = [], []
    for x, y in loader:
        with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
            X.append(model.backbone(gpu_prep(x, train=False, img_size=CONFIG["img_size"])).float().cpu())
        T.append(y.cpu())
    return torch.cat(X), torch.cat(T)

def finish_eval(L_val, T_val, L_test, T_test, counts, tau_grid):
    log_prior = np.log(counts / counts.sum() + 1e-12)
    tau_star, _ = search_tau(L_val, T_val, log_prior, tau_grid)
    T_star = search_temperature(L_val, T_val)
    return [
        ("raw", 0.0, compute_metrics(L_test, T_test)),
        ("tau", tau_star, compute_metrics(L_test - tau_star * log_prior, T_test)),
        ("temp_scale", T_star, compute_metrics(L_test / T_star, T_test)),
    ]

def run_stage2(ce_model, method, data, tau_grid, seed):
    set_seed(seed)
    Xtr, Ttr = collect_feats(ce_model, data["train_loader"])
    Xva, Tva = collect_feats(ce_model, data["val_loader"])
    Xte, Tte = collect_feats(ce_model, data["test_loader"])
    if method == "cRT":
        head = copy.deepcopy(ce_model.classifier)
        for m in head.modules():
            if hasattr(m, "reset_parameters"): m.reset_parameters()
    else:
        head = LWSHead(copy.deepcopy(ce_model.classifier))
    head = head.to(DEVICE)
    cnt = np.bincount(Ttr.numpy(), minlength=NUM_CLASSES).astype(np.float64)
    sw = torch.tensor(1.0 / np.maximum(cnt[Ttr.numpy()], 1.0), dtype=torch.double)
    dl = DataLoader(TensorDataset(Xtr, Ttr), batch_size=256,
                    sampler=WeightedRandomSampler(sw, num_samples=len(Ttr), replacement=True))
    opt = optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    def pred(X):
        head.eval()
        with torch.no_grad(): return head(X.to(DEVICE)).float().cpu().numpy()
    best_f1, best_state = -1.0, None
    for _ in range(CONFIG["stage2_epochs"]):
        head.train()
        for x, y in dl:
            opt.zero_grad(); F.cross_entropy(head(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
        f = per_class_f1(Tva.numpy(), pred(Xva).argmax(1)).mean()
        if f > best_f1: best_f1, best_state = float(f), copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return finish_eval(pred(Xva), Tva.numpy(), pred(Xte), Tte.numpy(), data["class_counts"], tau_grid), best_f1

def train_method(method, seed, backbone, data, tau_grid):
    E = CONFIG["epochs"]
    set_seed(seed)
    model = get_model(backbone).to(DEVICE)
    counts = data["class_counts"]
    is_halo = (method == "HALO")
    is_dcal = method in ("DCAL", "HALO")
    crit = None if is_halo else LOSS_MAP[method](counts).to(DEVICE)
    wm = DynamicWeightManager(counts) if is_dcal else None
    opt = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda")
    drw_start = int(0.8 * E)
    best_f1, best_state, acc_hist = -1.0, None, [0.0]

    for ep in range(1, E + 1):
        model.train()
        gamma = (acc_hist[-1] ** 2) if is_halo else 0.0
        w_active = torch.as_tensor(wm.get(gamma), dtype=torch.float32, device=DEVICE) if is_dcal else None
        tp, tt = [], []
        for x, y_dev in data["train_loader"]:
            x_dev = gpu_prep(x, train=True, img_size=CONFIG["img_size"])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"] and DEVICE.type == "cuda"):
                out = model(x_dev)
                if is_halo:
                    loss = F.cross_entropy(out, y_dev, weight=w_active)
                elif method == "DCAL":
                    loss = crit(out, y_dev, w_active=w_active)
                elif method == "LDAM-DRW":
                    loss = crit(out, y_dev, drw=(ep > drw_start))
                else:
                    loss = crit(out, y_dev)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if is_dcal: tp.append(out.argmax(1).detach().cpu()); tt.append(y_dev.cpu())
        sched.step()
        if is_dcal:
            tp_np, tt_np = torch.cat(tp).numpy(), torch.cat(tt).numpy()
            acc_hist.append(float((tp_np == tt_np).mean()))
            wm.update(per_class_f1(tt_np, tp_np))
        Lv, Tv = collect(model, data["val_loader"])
        fv = float(per_class_f1(Tv, Lv.argmax(1)).mean())
        if fv > best_f1:
            best_f1 = fv
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == E:
            print(f"    [{method}] ep {ep:3d}/{E} | best_val_f1={best_f1:.4f}", flush=True)

    model.load_state_dict(best_state)
    Lv, Tv = collect(model, data["val_loader"])
    Lt, Tt = collect(model, data["test_loader"])
    return finish_eval(Lv, Tv, Lt, Tt, counts, tau_grid), best_f1, model


def summarize(csv_path, latex_path):
    grouped = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            grouped[(r["backbone"], r["method"], r["stage"])][int(r["seed"])] = {k: float(r[k]) for k in ("f1", "acc", "ece", "tail_f1", "tau")}
    fmt = lambda v: "--" if not v else (f"{v[0]:.4f}" if len(v) == 1 else f"{np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")
    tex = ["% Script 1: CIFAR-10-LT Fair Baselines (Raw vs +tau* vs +T*)"]
    for bb in sorted({k[0] for k in grouped}):
        print(f"\n{'='*110}\nCIFAR-10-LT Baselines | Backbone: {bb}\n{'='*110}")
        print(f"{'Method':<18} | {'Raw F1 (tau=0)':<19} | {'+tau* F1':<19} | {'Tail F1 (+tau*)':<19} | {'Raw ECE':<8} | {'+T* ECE':<8} | {'p vs HALO'}")
        print("-" * 110)
        h_tau = grouped.get((bb, "HALO", "tau"), {})
        methods = sorted({k[1] for k in grouped if k[0] == bb}, key=lambda m: (m == "HALO", m))
        tex += [f"\n% {bb}", "\\begin{tabular}{l|cc|c|cc|c}\\toprule",
                "Method & Raw F1 ($\\tau{=}0$) & $+\\tau^\\star$ F1 & Tail F1 ($+\\tau^\\star$) & Raw ECE & $+T^\\star$ ECE & $p$-val \\\\ \\midrule"]
        for m in methods:
            rd, td, pd = grouped.get((bb, m, "raw"), {}), grouped.get((bb, m, "tau"), {}), grouped.get((bb, m, "temp_scale"), {})
            rf1, tf1 = [rd[s]["f1"] for s in sorted(rd)], [td[s]["f1"] for s in sorted(td)]
            tail = [td[s]["tail_f1"] for s in sorted(td)]
            rece, tece = [rd[s]["ece"] for s in sorted(rd)], [pd[s]["ece"] for s in sorted(pd)]
            com = sorted(set(td) & set(h_tau))
            if m == "HALO" or len(com) < 2: p_str = "--"
            else:
                a, b = np.array([h_tau[s]["f1"] for s in com]), np.array([td[s]["f1"] for s in com])
                p_val = float(ttest_rel(a, b)[1]) if (ttest_rel and not np.allclose(a, b)) else 1.0
                p_str = f"{p_val:.4f}"
            re_s, te_s = (f"{np.mean(rece):.4f}" if rece else "--"), (f"{np.mean(tece):.4f}" if tece else "--")
            print(f"{m:<18} | {fmt(rf1):<19} | {fmt(tf1):<19} | {fmt(tail):<19} | {re_s:<8} | {te_s:<8} | {p_str}")
            tex.append(f"{m} & {fmt(rf1).replace('+/-', '$\\pm$')} & {fmt(tf1).replace('+/-', '$\\pm$')} & {fmt(tail).replace('+/-', '$\\pm$')} & {re_s} & {te_s} & {p_str} \\\\")
        tex.append("\\bottomrule\\end{tabular}")
    with open(latex_path, "w", encoding="utf-8") as f: f.write("\n".join(tex) + "\n")
    print(f"\nSaved LaTeX tables to: {latex_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("NO GPU DETECTED (Accelerator is 'None')! In Kaggle right sidebar -> Session Options -> Accelerator -> select 'GPU T4 x2' or 'GPU P100' before running.")
    print(f"GPU Detected: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}", flush=True)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    out_csv = os.path.join(CONFIG["out_dir"], "baselines_runs.csv")
    fields = ["dataset", "backbone", "seed", "method", "stage", "f1", "acc", "ece", "head_f1", "med_f1", "tail_f1", "tau", "sel_f1", "epochs", "seconds"]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f): done.add((r["backbone"], r["seed"], r["method"]))
    else:
        with open(out_csv, "w", newline="") as f: csv.DictWriter(f, fields).writeheader()

    tau_grid = np.round(np.arange(0.0, CONFIG["tau_max"] + 1e-9, CONFIG["tau_step"]), 4)
    for bb in CONFIG["backbones"]:
        data = setup_cifar10_lt(BATCH_SIZES.get(bb, 64), CONFIG["img_size"])
        for seed in CONFIG["seeds"]:
            for m in CONFIG["methods"]:
                if (bb, str(seed), m) in done: continue
                print(f"\n=== [Script 1: Baselines] CIFAR-10-LT | {bb} | seed {seed} | {m} ===", flush=True)
                t0 = time.time()
                rows, sel_f1, model = train_method(m, seed, bb, data, tau_grid)
                el = round(time.time() - t0, 1)
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fields)
                    for stg, pval, met in rows:
                        w.writerow(dict(dataset="cifar10_lt", backbone=bb, seed=seed, method=m, stage=stg,
                                        tau=pval, sel_f1=round(sel_f1, 6), epochs=CONFIG["epochs"], seconds=el, **met))
                done.add((bb, str(seed), m))
                print("   " + " | ".join(f"{s}: F1={met['f1']:.4f} ECE={met['ece']:.4f} ({p:.2f})" for s, p, met in rows), flush=True)

                if m == "CE":
                    for s2 in ("cRT", "LWS"):
                        if (bb, str(seed), s2) in done: continue
                        t1 = time.time()
                        s2_rows, s2_sel = run_stage2(model, s2, data, tau_grid, seed)
                        s2_el = round(time.time() - t1, 1)
                        with open(out_csv, "a", newline="") as f:
                            w = csv.DictWriter(f, fields)
                            for stg, pval, met in s2_rows:
                                w.writerow(dict(dataset="cifar10_lt", backbone=bb, seed=seed, method=s2, stage=stg,
                                                tau=pval, sel_f1=round(s2_sel, 6), epochs=CONFIG["stage2_epochs"], seconds=s2_el, **met))
                        done.add((bb, str(seed), s2))
                        print(f"   [{s2} (Stage-2)] " + " | ".join(f"{s}: F1={met['f1']:.4f} ({p:.2f})" for s, p, met in s2_rows), flush=True)
    summarize(out_csv, os.path.join(CONFIG["out_dir"], "baselines_tables.tex"))

if __name__ == "__main__":
    main()
