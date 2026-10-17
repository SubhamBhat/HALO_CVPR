"""
HALO Revision Experiment Suite (Points 1 & 2 of the Meta-Review / Reviewer Revision Plan)
=========================================================================================
Drop-in experiment runner that reuses the exact data splits, backbones, and baseline losses from
`halo_unified_training.py` while adding every experiment requested by the AC and Reviewers (yQ6h, VuMh, mtBX):

  1. Fair Post-Hoc Calibration Across All Methods (Plan #1, #7):
     - Applies the exact same post-hoc prior calibration grid (tau* in [0, tau_max]) to EVERY baseline and
       EVERY ablation variant using the same validation set and checkpoint selection rule.
     - Records 3 evaluation stages for every single run:
         * "raw"        : uncalibrated model logits (tau = 0)
         * "tau"        : prior-calibrated logits z - tau* log(pi) maximizing validation Macro-F1
         * "temp_scale" : post-hoc temperature scaling z / T* minimizing validation NLL (for fair ECE comparison)

  2. Complete Baseline Suite + Margin-Scale Tuning (Plan #1, #6, Problem #7, #9):
     - Original 9 baselines: CE, WCE, CB, Focal, LDAM, DBM, ALPA, RobustFocal, DCAL
     - 5 new baselines:
         * LA              : Logit Adjustment loss (Menon et al., ICLR 2021)
         * BalancedSoftmax : Balanced Meta-Softmax loss (Ren et al., NeurIPS 2020)
         * LDAM-DRW        : LDAM + Deferred Re-Weighting at 80% epochs (Cao et al., NeurIPS 2019)
         * cRT             : Decoupled Classifier Re-Training (Kang et al., ICLR 2020)
         * LWS             : Decoupled Learnable Weight Scaling (Kang et al., ICLR 2020)
     - `--suite tune_margin`: sweeps margin scale s in {1, 5, 10, 30} and cosine-normalized classifier heads
       to diagnose and fix LDAM / DBM underperformance on SqueezeNet (Problem #7).

  3. Full Multi-Dataset Ablation Grid (Plan #2, Problems #1, #2, #3, #4):
     - Evaluates every variant BOTH raw (tau=0) and calibrated (+tau*), eliminating the calibration confound.
     - Weighting exponents (beta_rep, beta_pen): paper (1.0 -> 0.5), code (0.33 -> 0.33), pure sqrt (0.5 -> 0.5),
       undampened IW (1.0 -> 1.0), Only W_rep (gamma=0, beta=1.0), Only W_pen (gamma=0, beta=0.5).
     - Phase transition gamma(A): quadratic (p=2), linear (p=1), high-order (p=5), discrete epoch switch (t=E/2),
       discrete accuracy switch (A>=0.5), rep_only (gamma=0), dyn_only (gamma=1).
     - Failure penalty Delta_c: logarithmic (1 + log M_c), linear gap, direct ratio (M_c), none (Delta_c = 1).
     - Normalization & optimizer: with/without l1 trace norm under both Adam and SGD (Problem #2).
     - Stabilizer sensitivity: eps in {1e-9, 1e-6, 1e-4, 1e-2, 1e-1} + zero-val class policy (Problem #4).

  4. Validation-Fairness & Label-Free Feedback Variants (Plan #3, #4):
     - `feedback="train"`      : EMA of per-class training F1 (uses zero validation labels for weight updates).
     - `feedback="train_loss"` : EMA of per-class training CE loss (completely F1-free and val-label-free).
     - `feedback="oof"`        : Rotating K-fold out-of-fold F1 computed strictly inside the 85% training split.
     - `feedback="val"`        : Validation F1 feedback (as written in Algorithm 1).
     - `split_val=True`        : Stratified 50/50 split of validation set — Half A for feedback/checkpointing,
                                 Half B strictly for post-hoc tau* selection (no double-dipping; Reviewer VuMh).
     - `no_val_labels`         : Zero validation labels used anywhere (train feedback + last checkpoint + canonical
                                 Bayes prior adjustment tau = 1.0).
     - `full_data=True`        : Trains baselines and HALO on 100% of (train + val) data under identical budget.

  5. Per-Class Split Audit (`--report_data`, Problem #4, Reviewer VuMh):
     - Exports exact per-class train/val/test counts and reports classes with 0 validation or test samples.

Usage examples:
  python halo_experiments.py --report_data --datasets cifar10_lt cifar100_lt oxford_pet cub200 stanford_cars
  python halo_experiments.py --suite baselines --datasets cifar10_lt --backbones resnet18 --seeds 0 1 2
  python halo_experiments.py --suite ablation --datasets cifar10_lt cub200 stanford_cars --backbones resnet18 --seeds 0 1 2
  python halo_experiments.py --suite valfair --datasets cifar10_lt cub200 stanford_cars --backbones resnet18 --seeds 0 1 2
  python halo_experiments.py --suite tune_margin --datasets cifar100_lt oxford_pet cub200 --backbones squeezenet
"""
import os
import csv
import time
import copy
import random
import argparse
from dataclasses import dataclass, replace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score

import halo_unified_training as H

cfg = H.cfg
FIELDS = [
    "dataset", "backbone", "seed", "method", "variant", "stage",
    "f1", "acc", "ece", "head_f1", "med_f1", "tail_f1",
    "tau", "sel_f1", "epochs", "seconds"
]


# =============================================================================
# 1. EXPERIMENT VARIANT SPECIFICATION
# =============================================================================
@dataclass(frozen=True)
class V:
    """Configuration for a single method/ablation/fairness variant."""
    name: str = "full"
    # --- HALO weighting & schedule ---
    beta_rep: float = 0.33        # Exponent on (N_max / N_c) for W_rep (paper: 1.0, code default: 0.33)
    beta_pen: float = 0.33        # Exponent on (N_max / N_c) for W_pen (paper: 0.5, code default: 0.33)
    gate: str = "poly"            # poly (A^p) | discrete_epoch (t >= E/2) | discrete_acc | rep_only | dyn_only
    p: float = 2.0                # Polynomial exponent for gate == "poly"
    gate_thresh: float = 0.5      # Threshold for gate == "discrete_acc"
    penalty: str = "log"          # log (1 + log M_c) | linear | ratio (M_c) | none (Delta_c = 1)
    feedback: str = "train"       # train | train_loss | oof | val | none
    oof_folds: int = 5            # Number of rotating folds within train split when feedback == "oof"
    zero_val_policy: str = "raw"  # raw (use F1=0 when val count is 0) | neutral (set Delta_c=1 if count==0)
    l1norm: bool = True           # Apply l1 trace normalization K * w / sum(w)
    ema: bool = True              # Variance-weighted EMA smoothing on feedback signal
    eps: float = 1e-9             # Numerical stabilizer in M_c = (F1_max + eps) / (F1_c + eps)
    label_smoothing: float = 0.2  # Label smoothing for HALO weighted CE
    # --- Optimization & protocol (shared across methods) ---
    optimizer: str = "adam"       # adam | sgd (used to test l1-norm under SGD vs Adam, Problem #2)
    select: str = "best"          # best (val Macro-F1 checkpoint) | last (final epoch, no val selection)
    tau_mode: str = "search"      # search (grid search on val) | fixed1 (canonical tau=1, no val labels)
    split_val: bool = False       # Disjoint 50/50 val split: Half A -> feedback/ckpt, Half B -> tau*
    full_data: bool = False       # Train on 100% (train + val); implies select="last", tau_mode="fixed1"
    # --- Margin baseline tuning (Problem #7) ---
    scale: float = None           # Logit scale s for LDAM / LDAM-DRW / DBM (None -> default 30.0)
    cosine_head: bool = False     # Use l2-normalized cosine classifier head for margin losses


def ablation_suite():
    """Comprehensive ablation grid addressing Plan #2 and Problems #1, #2, #3, #4."""
    f = V()
    p_base = replace(f, name="paper_exponents(1.0->0.5)", beta_rep=1.0, beta_pen=0.5)
    return [
        # Full models (code exponents vs paper-stated exponents)
        f,
        p_base,
        # 1. Importance-sampling exponent ablations (Problem #3)
        replace(f, name="only_W_rep(undampened,beta=1)", beta_rep=1.0, beta_pen=1.0, gate="rep_only", penalty="none", feedback="none"),
        replace(f, name="only_W_pen(sqrt,beta=0.5)", beta_rep=0.5, beta_pen=0.5, gate="rep_only", penalty="none", feedback="none"),
        replace(f, name="only_W_pen(cbrt,beta=0.33)", beta_rep=0.33, beta_pen=0.33, gate="rep_only", penalty="none", feedback="none"),
        replace(f, name="beta_both=0.5(sqrt)", beta_rep=0.5, beta_pen=0.5),
        replace(f, name="beta_both=1.0(naive_IW)", beta_rep=1.0, beta_pen=1.0),
        # 2. Phase transition schedule gamma(A) ablations (Sec. 3.2 & 3.4)
        replace(f, name="gate_linear(p=1)", p=1.0),
        replace(f, name="gate_high_order(p=5)", p=5.0),
        replace(f, name="gate_discrete_epoch(ts=50)", gate="discrete_epoch"),
        replace(f, name="gate_discrete_acc(A>=0.5)", gate="discrete_acc", gate_thresh=0.5),
        replace(f, name="gate_rep_only(gamma=0)", gate="rep_only"),
        replace(f, name="gate_dyn_only(gamma=1)", gate="dyn_only"),
        # Same transition ablations under paper exponents (1.0 -> 0.5) so W_rep != W_pen
        replace(p_base, name="paper_gate_linear(p=1)", p=1.0),
        replace(p_base, name="paper_gate_high_order(p=5)", p=5.0),
        replace(p_base, name="paper_gate_discrete_epoch", gate="discrete_epoch"),
        # 3. Failure penalty form ablations (Sec. 3.3: log vs linear vs ratio M_c vs none)
        replace(f, name="penalty_linear_gap", penalty="linear"),
        replace(f, name="penalty_ratio_Mc", penalty="ratio"),
        replace(f, name="penalty_none(Delta=1)", penalty="none", feedback="none"),
        # 4. Feedback source ablations
        replace(f, name="feedback_val", feedback="val"),
        replace(f, name="feedback_train_loss", feedback="train_loss"),
        replace(f, name="feedback_oof5", feedback="oof", oof_folds=5),
        replace(f, name="feedback_none(Delta=1)", feedback="none"),
        # 5. Trace normalization & EMA ablations (Problem #2: Adam vs SGD)
        replace(f, name="no_l1norm(adam)", l1norm=False),
        replace(f, name="with_l1norm(sgd)", optimizer="sgd", l1norm=True),
        replace(f, name="no_l1norm(sgd)", optimizer="sgd", l1norm=False),
        replace(f, name="no_ema", ema=False),
        # 6. Stabilizer epsilon sensitivity & zero-val policy (Problem #4)
        replace(f, name="eps=1e-6", eps=1e-6),
        replace(f, name="eps=1e-4", eps=1e-4),
        replace(f, name="eps=1e-2", eps=1e-2),
        replace(f, name="eps=1e-1", eps=1e-1),
        replace(f, name="val_eps=1e-9", feedback="val", eps=1e-9),
        replace(f, name="val_eps=1e-4", feedback="val", eps=1e-4),
        replace(f, name="val_eps=1e-2", feedback="val", eps=1e-2),
        replace(f, name="val_eps=1e-1", feedback="val", eps=1e-1),
        replace(f, name="val_zero_neutral", feedback="val", zero_val_policy="neutral"),
    ]


def valfair_suite():
    """Validation fairness experiments addressing Plan #3 and Plan #4 (AC, mtBX, VuMh)."""
    f = V()
    return {
        "HALO": [
            replace(f, name="train_fb(default)", feedback="train"),
            replace(f, name="train_loss_fb", feedback="train_loss"),
            replace(f, name="oof5_fb", feedback="oof", oof_folds=5),
            replace(f, name="val_fb", feedback="val"),
            replace(f, name="val_fb_neutral_zeros", feedback="val", zero_val_policy="neutral"),
            replace(f, name="split_val(train_fb)", feedback="train", split_val=True),
            replace(f, name="split_val(val_fb)", feedback="val", split_val=True),
            replace(f, name="no_val_labels(train_fb,tau=1)", feedback="train", select="last", tau_mode="fixed1"),
            replace(f, name="no_val_labels(loss_fb,tau=1)", feedback="train_loss", select="last", tau_mode="fixed1"),
            replace(f, name="no_val_labels(oof_fb,tau=1)", feedback="oof", select="last", tau_mode="fixed1"),
            replace(f, name="full_data_100pct(train_fb,tau=1)", feedback="train", full_data=True, select="last", tau_mode="fixed1"),
            replace(f, name="full_data_100pct(loss_fb,tau=1)", feedback="train_loss", full_data=True, select="last", tau_mode="fixed1"),
        ],
        "*": [
            replace(f, name="standard_85pct"),
            replace(f, name="split_val", split_val=True),
            replace(f, name="no_val_labels(last,tau=1)", select="last", tau_mode="fixed1"),
            replace(f, name="full_data_100pct(last,tau=1)", full_data=True, select="last", tau_mode="fixed1"),
        ],
    }


BASELINES = [
    "CE", "WCE", "CB", "Focal", "LDAM", "DBM", "ALPA", "RobustFocal", "DCAL",
    "LA", "BalancedSoftmax", "LDAM-DRW", "cRT", "LWS"
]
HALO_LIKE = ("HALO",)
SIMPLE = {
    "CE": H.CrossEntropyLossBaseline,
    "WCE": H.WCELoss,
    "CB": H.ClassBalancedLoss,
    "Focal": H.FocalLoss,
    "ALPA": H.ALPALoss,
    "RobustFocal": H.RobustFocalLoss,
    "Dice": H.DiceLoss,
    "DWB": H.DWBLoss,
    "cRT": H.CrossEntropyLossBaseline,
    "LWS": H.CrossEntropyLossBaseline,
}


def plan(suites, methods_filter):
    runs = []
    for s in suites:
        if s == "baselines":
            runs += [(m, V()) for m in BASELINES] + [("HALO", V())]
        elif s == "ablation":
            runs += [("HALO", v) for v in ablation_suite()]
        elif s == "valfair":
            vf = valfair_suite()
            runs += [("HALO", v) for v in vf["HALO"]]
            for m in ("CE", "WCE", "CB", "LA", "BalancedSoftmax", "LDAM-DRW"):
                runs += [(m, v) for v in vf["*"]]
        elif s == "tune_margin":
            for m in ("LDAM", "LDAM-DRW", "DBM"):
                for cos in (False, True):
                    for sc in (1.0, 5.0, 10.0, 30.0):
                        tag = f"{'cos' if cos else 'lin'}_s={sc:g}"
                        runs.append((m, V(name=tag, scale=sc, cosine_head=cos)))
        else:
            raise ValueError(f"Unknown suite: {s}")
    seen, out = set(), []
    for m, v in runs:
        if (m, v.name) not in seen and (not methods_filter or m in methods_filter):
            seen.add((m, v.name))
            out.append((m, v))
    return out


# =============================================================================
# 2. METRICS & CALIBRATION (RAW, TAU*, TEMPERATURE SCALING)
# =============================================================================
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def softmax_np(z):
    z_shift = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=1, keepdims=True)


def per_class_f1(t, p):
    return f1_score(t, p, labels=list(range(cfg.NUM_CLASSES)), average=None, zero_division=0)


def ece_np(probs, t, n_bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs((pred[m] == t[m]).mean() - conf[m].mean())
    return float(ece)


def metrics(logits, t):
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    f1 = per_class_f1(t, pred)
    counts = np.array(cfg.CLASS_COUNTS)
    order = np.argsort(-counts)  # descending: head -> medium -> tail
    C = len(counts)
    k1, k2 = max(1, C // 3), max(2, (2 * C) // 3)
    head_idx, med_idx, tail_idx = order[:k1], order[k1:k2], order[k2:]
    return dict(
        f1=float(f1.mean()),
        acc=float((pred == t).mean()),
        ece=float(ece_np(probs, t)),
        head_f1=float(f1[head_idx].mean()) if len(head_idx) else 0.0,
        med_f1=float(f1[med_idx].mean()) if len(med_idx) else 0.0,
        tail_f1=float(f1[tail_idx].mean()) if len(tail_idx) else 0.0,
    )


def search_tau(val_logits, val_t, log_prior, tau_grid):
    best_tau, best_f1 = 0.0, -1.0
    for tau in tau_grid:
        f = per_class_f1(val_t, (val_logits - tau * log_prior).argmax(axis=1)).mean()
        if f > best_f1:
            best_f1, best_tau = float(f), float(tau)
    return best_tau, best_f1


def search_temperature(val_logits, val_t):
    """Finds scalar temperature T > 0 minimizing validation NLL (for fair ECE comparison, Plan #7)."""
    z_t = torch.tensor(val_logits, dtype=torch.float32)
    y_t = torch.tensor(val_t, dtype=torch.long)
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.arange(0.2, 2.0, 0.05), np.arange(2.0, 5.05, 0.25)]):
        nll = F.cross_entropy(z_t / float(T), y_t).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


@torch.no_grad()
def collect(model, loader):
    model.eval()
    L, T = [], []
    for x, y in loader:
        L.append(model(x.to(cfg.DEVICE)).float().cpu().numpy())
        T.append(y.numpy())
    L, T = np.concatenate(L), np.concatenate(T)
    m = T >= 0
    return L[m], T[m]


@torch.no_grad()
def collect_feats(model, loader):
    model.eval()
    X, T = [], []
    for x, y in loader:
        X.append(model.backbone(x.to(cfg.DEVICE)).float().cpu())
        T.append(y)
    X, T = torch.cat(X), torch.cat(T)
    m = T >= 0
    return X[m], T[m]


def get_labels(loader):
    return np.concatenate([y.numpy() for _, y in loader])


# =============================================================================
# 3. FLEXIBLE HALO WEIGHT MANAGER & NEW BASELINES
# =============================================================================
class FlexWeights:
    """Configurable HALO weight manager supporting all ablation & validation-fairness switches."""
    def __init__(self, counts, v: V, obs_counts=None):
        self.v = v
        self.c = len(counts)
        ratio = counts.max() / np.maximum(counts, 1.0)
        self.w_rep = ratio ** v.beta_rep
        self.w_pen = ratio ** v.beta_pen
        self.w_dyn = self.w_pen.copy()
        self.init = False
        self.obs_counts = obs_counts  # class counts in the feedback split (to detect 0-val classes)
        self.signal_smooth = np.ones(self.c) / self.c
        self.sigma2 = np.ones(self.c) * 0.01
        self.hist = []

    def _norm(self, w):
        return self.c * w / (w.sum() + 1e-9) if self.v.l1norm else w.copy()

    def update(self, signal, is_loss=False):
        """
        Updates dynamic penalty weights W_dyn.
        If `is_loss=False`, `signal` is per-class F1 in [0, 1] (higher is better).
        If `is_loss=True`, `signal` is per-class CE loss >= 0 (higher means worse class error).
        """
        v = self.v
        self.init = True
        mult = np.ones(self.c, dtype=np.float64)
        if signal is not None and v.penalty != "none":
            sig = np.asarray(signal, dtype=np.float64)
            if v.ema:
                inv = sig - self.signal_smooth
                self.sigma2 = 0.9 * self.sigma2 + 0.1 * (inv ** 2)
                lr = 0.1 / (0.1 + self.sigma2 / (self.sigma2.mean() + 1e-9))
                self.signal_smooth = self.signal_smooth + lr * inv
            else:
                self.signal_smooth = sig.copy()

            s = self.signal_smooth
            if is_loss:
                # For loss feedback: failure ratio M_c = (loss_c + eps) / (min_j loss_j + eps) >= 1
                s_min = s.min()
                M_c = (s + v.eps) / (s_min + v.eps)
                gap = (s - s_min) / (s.max() + v.eps)
            else:
                # For F1 feedback: failure ratio M_c = (max_j F1_j + eps) / (F1_c + eps) >= 1
                s_max = s.max()
                if s_max > 0:
                    M_c = (s_max + v.eps) / (s + v.eps)
                    gap = (s_max - s) / (s_max + v.eps)
                else:
                    M_c = np.ones(self.c, dtype=np.float64)
                    gap = np.zeros(self.c, dtype=np.float64)

            if v.penalty == "log":
                mult = 1.0 + np.log(np.maximum(M_c, 1.0))
            elif v.penalty == "linear":
                mult = 1.0 + gap
            elif v.penalty == "ratio":
                mult = M_c
            else:
                raise ValueError(f"Unknown penalty: {v.penalty}")

            # Handle classes with 0 samples in the feedback split when policy == "neutral"
            if v.zero_val_policy == "neutral" and self.obs_counts is not None:
                mult[self.obs_counts == 0] = 1.0

        self.w_dyn = self.w_pen * mult

    def get(self, gamma):
        if not self.init:
            return self._norm(self.w_rep)
        w_act = (1.0 - gamma) * self.w_rep + gamma * self.w_dyn
        normed = self._norm(w_act)
        self.hist.append(normed.copy())
        return normed


def gate_value(v: V, A: float, ep: int, total_epochs: int) -> float:
    if v.gate == "poly":
        return float(A) ** v.p
    if v.gate == "discrete_epoch":
        return 1.0 if ep > (total_epochs // 2) else 0.0
    if v.gate == "discrete_acc":
        return 1.0 if A >= v.gate_thresh else 0.0
    if v.gate == "rep_only":
        return 0.0
    if v.gate == "dyn_only":
        return 1.0
    raise ValueError(f"Unknown gate: {v.gate}")


class CosineClassifier(nn.Module):
    """L2-normalized cosine similarity linear head (standard for LDAM / margin losses)."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm)


def replace_with_cosine_head(model):
    """Replaces the final Linear layer in model.classifier with CosineClassifier."""
    if isinstance(model.classifier, nn.Linear):
        model.classifier = CosineClassifier(model.classifier.in_features, cfg.NUM_CLASSES)
    elif isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier.children())
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], nn.Linear):
                layers[i] = CosineClassifier(layers[i].in_features, cfg.NUM_CLASSES)
                break
        model.classifier = nn.Sequential(*layers)
    return model


class LogitAdjustedLoss(nn.Module):
    """Menon et al. (ICLR 2021) Logit-Adjusted training loss: CE(z + tau * log pi, y)."""
    def __init__(self, tau=1.0):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        pi = c / c.sum()
        self.offset = torch.tensor(tau * np.log(pi + 1e-12), dtype=torch.float32, device=cfg.DEVICE)

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.offset, targets)


class BalancedSoftmaxLoss(nn.Module):
    """Ren et al. (NeurIPS 2020) Balanced Meta-Softmax loss: CE(z + log N_c, y)."""
    def __init__(self):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        self.log_n = torch.tensor(np.log(np.maximum(c, 1.0)), dtype=torch.float32, device=cfg.DEVICE)

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.log_n, targets)


class LDAMDRWLoss(nn.Module):
    """Cao et al. (NeurIPS 2019) LDAM loss with Deferred Re-Weighting (DRW) at 80% of training."""
    def __init__(self, s=30.0, max_m=0.5, beta=0.9999):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        m = 1.0 / np.sqrt(np.sqrt(c))
        m = m * (max_m / m.max())
        w = (1.0 - beta) / (1.0 - np.power(beta, c) + 1e-8)
        w = w / w.sum() * len(c)
        self.m = torch.tensor(m, dtype=torch.float32, device=cfg.DEVICE)
        self.w = torch.tensor(w, dtype=torch.float32, device=cfg.DEVICE)
        self.s = s

    def forward(self, logits, targets, drw=False):
        oh = F.one_hot(targets, logits.size(1)).bool()
        out = torch.where(oh, logits - self.m[targets][:, None], logits)
        return F.cross_entropy(self.s * out, targets, weight=self.w if drw else None)


class LWSHead(nn.Module):
    """Kang et al. (ICLR 2020) Learnable Weight Scaling (LWS) on a frozen linear classifier."""
    def __init__(self, clf):
        super().__init__()
        self.clf = clf
        for p in self.clf.parameters():
            p.requires_grad_(False)
        self.scale = nn.Parameter(torch.ones(cfg.NUM_CLASSES))

    def forward(self, f):
        return self.clf(f) * self.scale


def build_criterion(method, v: V):
    if method == "LA":
        return LogitAdjustedLoss(tau=1.0).to(cfg.DEVICE)
    if method == "BalancedSoftmax":
        return BalancedSoftmaxLoss().to(cfg.DEVICE)
    if method == "LDAM":
        return H.LDAMLoss(s=v.scale or 30.0).to(cfg.DEVICE)
    if method == "DBM":
        return H.DBMLoss(scale=v.scale or 30.0).to(cfg.DEVICE)
    if method == "LDAM-DRW":
        return LDAMDRWLoss(s=v.scale or 30.0).to(cfg.DEVICE)
    if method == "DCAL":
        return H.PublishedDCALoss().to(cfg.DEVICE)
    if method in SIMPLE:
        return SIMPLE[method]().to(cfg.DEVICE)
    return None


# =============================================================================
# 4. STAGE-2 DECOUPLED RE-TRAINING (cRT & LWS)
# =============================================================================
def two_stage(model, method, loaders, v: V, args):
    """Decoupled Stage-2 training (Kang et al., ICLR 2020) on frozen Stage-1 features."""
    train_loader, sel_loader, tau_loader, test_loader = loaders
    dev, C = cfg.DEVICE, cfg.NUM_CLASSES
    Xtr, Ttr = collect_feats(model, train_loader)
    eval_feats = {"test": collect_feats(model, test_loader)}
    if sel_loader is not None:
        eval_feats["sel"] = collect_feats(model, sel_loader)
    if tau_loader is not None:
        eval_feats["tau"] = eval_feats["sel"] if tau_loader is sel_loader else collect_feats(model, tau_loader)

    if method == "cRT":
        head = copy.deepcopy(model.classifier)
        for m in head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    else:
        head = LWSHead(copy.deepcopy(model.classifier))
    head = head.to(dev)

    cnt = np.bincount(Ttr.numpy(), minlength=C).astype(np.float64)
    sw = torch.tensor(1.0 / np.maximum(cnt[Ttr.numpy()], 1.0), dtype=torch.double)
    dl = DataLoader(
        TensorDataset(Xtr, Ttr),
        batch_size=256,
        sampler=WeightedRandomSampler(sw, num_samples=len(Ttr), replacement=True),
    )
    opt = optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

    def predict(X):
        head.eval()
        with torch.no_grad():
            return head(X.to(dev)).float().cpu().numpy()

    best, best_state = -1.0, None
    for _ in range(args.stage2_epochs):
        head.train()
        for x, y in dl:
            opt.zero_grad()
            F.cross_entropy(head(x.to(dev)), y.to(dev)).backward()
            opt.step()
        if v.select == "best" and "sel" in eval_feats:
            X, T = eval_feats["sel"]
            f = per_class_f1(T.numpy(), predict(X).argmax(axis=1)).mean()
            if f > best:
                best, best_state = float(f), copy.deepcopy(head.state_dict())
    if best_state is not None:
        head.load_state_dict(best_state)
    return {k: (predict(X), T.numpy()) for k, (X, T) in eval_feats.items()}


# =============================================================================
# 5. TRAINING LOOP (SUPPORTING TRAIN / LOSS / OOF / VAL FEEDBACK)
# =============================================================================
def make_oof_folds(train_loader, k_folds, seed):
    """Partitions the training dataset into K stratified folds for out-of-fold (OOF) feedback."""
    ds = train_loader.dataset
    labels = get_labels(train_loader)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k_folds)]
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        for i, ix in enumerate(idx):
            folds[i % k_folds].append(int(ix))
    bs, nw = cfg.BATCH_SIZE, cfg.NUM_WORKERS
    fold_pairs = []
    all_idx = set(range(len(labels)))
    for k in range(k_folds):
        oof_idx = sorted(folds[k])
        sub_tr_idx = sorted(all_idx - set(oof_idx))
        if not sub_tr_idx or not oof_idx:
            sub_tr_idx = list(range(len(labels)))
            oof_idx = list(range(len(labels)))
        tr_k = DataLoader(Subset(ds, sub_tr_idx), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
        oof_k = DataLoader(Subset(ds, oof_idx), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        fold_pairs.append((tr_k, oof_k))
    return fold_pairs


def train_model(method, v: V, seed: int, loaders, args):
    train_loader, sel_loader, tau_loader, test_loader = loaders
    dev, C, E = cfg.DEVICE, cfg.NUM_CLASSES, cfg.EPOCHS
    set_seed(seed)

    model = H.get_model(cfg.BACKBONE)
    if v.cosine_head:
        model = replace_with_cosine_head(model)
    model = model.to(dev)

    counts = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
    crit = build_criterion(method, v)
    is_halo = method in HALO_LIKE

    obs_counts = None
    if is_halo and v.feedback == "val" and sel_loader is not None:
        obs_counts = np.bincount(get_labels(sel_loader), minlength=C)

    wm = FlexWeights(counts, v, obs_counts=obs_counts) if is_halo else None
    dcal_wm = H.DynamicWeightManager(use_ema=True) if method == "DCAL" else None
    if is_halo and v.feedback == "none":
        wm.update(None)

    oof_pairs = make_oof_folds(train_loader, v.oof_folds, seed) if (is_halo and v.feedback == "oof") else None
    drw_start = int(0.8 * E)

    if v.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE * 10.0, momentum=0.9, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-6)

    need_sel_eval = sel_loader is not None and (v.select == "best" or (is_halo and v.feedback == "val"))
    best_f1, best_state, acc_hist = -1.0, None, [0.0]

    for ep in range(1, E + 1):
        model.train()
        w_active = None
        if is_halo:
            gamma = gate_value(v, acc_hist[-1], ep, E)
            w_active = torch.as_tensor(wm.get(gamma), dtype=torch.float32, device=dev)
        elif method == "DCAL":
            w_active = torch.as_tensor(dcal_wm.get_active_weights(0.0), dtype=torch.float32, device=dev)

        active_train_loader = oof_pairs[(ep - 1) % len(oof_pairs)][0] if oof_pairs is not None else train_loader
        tp, tt, t_losses = [], [], []

        for x, y in active_train_loader:
            y_dev = y.to(dev)
            opt.zero_grad()
            out = model(x.to(dev))

            if is_halo:
                ce_vec = F.cross_entropy(out, y_dev, reduction="none", label_smoothing=v.label_smoothing)
                loss = (ce_vec * w_active[y_dev]).mean()
                tp.append(out.argmax(dim=1).detach().cpu())
                tt.append(y)
                if v.feedback == "train_loss":
                    with torch.no_grad():
                        raw_ce = F.cross_entropy(out.detach(), y_dev, reduction="none")
                        t_losses.append(raw_ce.cpu())
            elif method == "DCAL":
                loss = crit(out, y_dev, w_active=w_active)
                tp.append(out.argmax(dim=1).detach().cpu())
                tt.append(y)
            elif method == "LDAM-DRW":
                loss = crit(out, y_dev, drw=(ep > drw_start))
            else:
                loss = crit(out, y_dev)

            loss.backward()
            opt.step()

        sched.step()

        # Update training accuracy history & training-derived feedback signals
        if is_halo or method == "DCAL":
            tp_np = torch.cat(tp).numpy()
            tt_np = torch.cat(tt).numpy()
            acc_hist.append(float((tp_np == tt_np).mean()))

            if method == "DCAL":
                dcal_wm.update(per_class_f1(tt_np, tp_np), np.zeros(C))
            elif is_halo:
                if v.feedback == "train":
                    wm.update(per_class_f1(tt_np, tp_np), is_loss=False)
                elif v.feedback == "train_loss":
                    loss_np = torch.cat(t_losses).numpy()
                    cls_loss = np.zeros(C, dtype=np.float64)
                    for c in range(C):
                        m = tt_np == c
                        cls_loss[c] = loss_np[m].mean() if m.any() else 0.0
                    wm.update(cls_loss, is_loss=True)
                elif v.feedback == "oof":
                    _, oof_loader = oof_pairs[(ep - 1) % len(oof_pairs)]
                    Lo, To = collect(model, oof_loader)
                    wm.update(per_class_f1(To, Lo.argmax(axis=1)), is_loss=False)

        if need_sel_eval:
            L, T = collect(model, sel_loader)
            fv = per_class_f1(T, L.argmax(axis=1))
            if is_halo and v.feedback == "val":
                wm.update(fv, is_loss=False)
            if v.select == "best" and fv.mean() > best_f1:
                best_f1 = float(fv.mean())
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

        if ep % 10 == 0 or ep == E:
            print(f"    [{method}/{v.name}] epoch {ep:3d}/{E} | val_best_f1={best_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    if method in ("cRT", "LWS"):
        out = two_stage(model, method, loaders, v, args)
    else:
        out = {"test": collect(model, test_loader)}
        if sel_loader is not None:
            out["sel"] = collect(model, sel_loader)
        if tau_loader is not None:
            out["tau"] = out["sel"] if tau_loader is sel_loader else collect(model, tau_loader)
    return out, (best_f1 if best_f1 >= 0 else float("nan"))


# =============================================================================
# 6. DATA SPLITTING MODES (STANDARD, SPLIT-VAL 50/50, 100% FULL DATA)
# =============================================================================
def prepare_loaders(base, v: V, cache):
    train_loader, val_loader, test_loader = base
    bs, nw = cfg.BATCH_SIZE, cfg.NUM_WORKERS
    if v.full_data:
        tl = DataLoader(
            ConcatDataset([train_loader.dataset, val_loader.dataset]),
            batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
        )
        return tl, None, None, test_loader
    if v.split_val:
        if "halves" not in cache:
            lab = get_labels(val_loader)
            rng = np.random.default_rng(42)
            A, B = [], []
            toggle = 0
            for c in np.unique(lab):
                idx = rng.permutation(np.where(lab == c)[0]).tolist()
                if len(idx) == 1:
                    # Alternate singleton validation classes so neither half is starved
                    (A if toggle % 2 == 0 else B).extend(idx)
                    toggle += 1
                else:
                    k = len(idx) // 2
                    A.extend(idx[:k])
                    B.extend(idx[k:])
            if not B:
                B = list(A)
            mk = lambda ids: DataLoader(
                Subset(val_loader.dataset, sorted(ids)),
                batch_size=bs, shuffle=False, num_workers=nw,
            )
            cache["halves"] = (mk(A), mk(B))
        a, b = cache["halves"]
        return train_loader, a, b, test_loader
    return train_loader, val_loader, val_loader, test_loader


def report_data(base, name):
    tr, va, te = base
    C = cfg.NUM_CLASSES
    cnt = {
        k: np.bincount(lab[lab >= 0], minlength=C)
        for k, lab in (("train", get_labels(tr)), ("val", get_labels(va)), ("test", get_labels(te)))
    }
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    path = os.path.join(cfg.SAVE_DIR, f"class_counts_{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "val", "test"])
        for c in range(C):
            w.writerow([c, int(cnt["train"][c]), int(cnt["val"][c]), int(cnt["test"][c])])
    order = np.argsort(cnt["train"])
    print(f"\n[{name}] Rarest 10 classes (class_id: train / val / test):")
    for c in order[:10]:
        print(f"   class {c:4d}: {cnt['train'][c]:4d} / {cnt['val'][c]:4d} / {cnt['test'][c]:4d}")
    print(
        f"[{name}] Summary: total={cnt['train'].sum()}/{cnt['val'].sum()}/{cnt['test'].sum()} | "
        f"train min/max={cnt['train'].min()}/{cnt['train'].max()} | "
        f"classes with <=2 train: {int((cnt['train'] <= 2).sum())}/{C} | "
        f"classes with 0 val: {int((cnt['val'] == 0).sum())}/{C} | "
        f"classes with 0 test: {int((cnt['test'] == 0).sum())}/{C}\n"
        f"  -> Saved per-class counts to {path}"
    )


# =============================================================================
# 7. EVALUATION STAGES & MAIN CLI DRIVER
# =============================================================================
def finish(out, v: V, tau_grid):
    """Computes test metrics at 3 stages: raw (tau=0), tau-calibrated (+tau*), and temperature-scaled (T*)."""
    counts = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
    log_prior = np.log(counts / counts.sum() + 1e-12)
    Lt, Tt = out["test"]

    rows = [("raw", 0.0, metrics(Lt, Tt))]

    if v.tau_mode == "search" and "tau" in out:
        tau_star, _ = search_tau(*out["tau"], log_prior, tau_grid)
    else:
        tau_star = 1.0
    rows.append(("tau", tau_star, metrics(Lt - tau_star * log_prior, Tt)))

    if "tau" in out:
        T_star = search_temperature(*out["tau"])
        rows.append(("temp_scale", T_star, metrics(Lt / T_star, Tt)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="HALO Revision Experiment Runner")
    ap.add_argument("--datasets", nargs="+", default=["cifar10_lt"])
    ap.add_argument("--backbones", nargs="+", default=["resnet18"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--suite", nargs="+", default=["baselines"],
        choices=["baselines", "ablation", "valfair", "tune_margin"],
    )
    ap.add_argument("--methods", nargs="*", default=None, help="Restrict execution to specific method names")
    ap.add_argument("--variants", nargs="*", default=None, help="Restrict execution to specific variant names")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stage2_epochs", type=int, default=30)
    ap.add_argument("--tau_max", type=float, default=3.0, help="Max tau for Phase-3 grid search (paper Eq. 23: 3.0)")
    ap.add_argument("--tau_step", type=float, default=0.05, help="Step size for Phase-3 grid search (paper: 0.05)")
    ap.add_argument("--out", default=None, help="CSV output path (default: <SAVE_DIR>/runs.csv)")
    ap.add_argument("--report_data", action="store_true", help="Only report per-class train/val/test counts and exit")
    args = ap.parse_args()

    if args.epochs:
        cfg.EPOCHS = args.epochs
    tau_grid = np.round(np.arange(0.0, args.tau_max + 1e-9, args.tau_step), 4)

    out_path = args.out or os.path.join(cfg.SAVE_DIR, "runs.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["backbone"], r["seed"], r["method"], r["variant"]))
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    todo = plan(args.suite, args.methods)
    if args.variants:
        todo = [(m, v) for m, v in todo if v.name in args.variants]

    for ds in args.datasets:
        for bb in args.backbones:
            cfg.DATASET = ds
            cfg.set_backbone(bb)
            base = H.setup_data()
            if args.report_data:
                report_data(base, ds)
                break
            cache = {}
            for seed in args.seeds:
                for method, v in todo:
                    key = (ds, bb, str(seed), method, v.name)
                    if key in done:
                        continue
                    print(f"\n=== {ds} | {bb} | seed {seed} | {method} [{v.name}] ===", flush=True)
                    t0 = time.time()
                    loaders = prepare_loaders(base, v, cache)
                    out, sel_f1 = train_model(method, v, seed, loaders, args)
                    rows = finish(out, v, tau_grid)
                    elapsed = round(time.time() - t0, 1)
                    with open(out_path, "a", newline="") as f:
                        w = csv.DictWriter(f, FIELDS)
                        for stage, param_val, m in rows:
                            w.writerow(dict(
                                dataset=ds, backbone=bb, seed=seed, method=method,
                                variant=v.name, stage=stage, tau=param_val,
                                sel_f1=round(sel_f1, 6) if not np.isnan(sel_f1) else "",
                                epochs=cfg.EPOCHS, seconds=elapsed, **m
                            ))
                    done.add(key)
                    summary_str = " | ".join(
                        f"{s}: F1={m['f1']:.4f} TailF1={m['tail_f1']:.4f} Acc={m['acc']:.4f} ECE={m['ece']:.4f} ({p:.2f})"
                        for s, p, m in rows
                    )
                    print("   " + summary_str, flush=True)


if __name__ == "__main__":
    main()
