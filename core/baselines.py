import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import math


class CrossEntropyLossBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, logits, targets): return self.criterion(logits, targets)

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.9999):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        w = (1.0 - beta) / ((1.0 - np.power(beta, c)) + 1e-8)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(w / w.sum() * len(c)).to(cfg.DEVICE))
    def forward(self, logits, targets): return self.criterion(logits, targets)

class DWBLoss(nn.Module):
    def __init__(self, base_weight_offset=0.5):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        self.class_weights = torch.FloatTensor(np.log(c.max() / c) + base_weight_offset).to(cfg.DEVICE)
    def forward(self, logits, targets):
        p_t = (F.softmax(logits, dim=1) * F.one_hot(targets, cfg.NUM_CLASSES).float()).sum(1).clamp(min=1e-7)
        return (torch.pow(self.class_weights[targets].clamp(min=1e-3), (1.0 - p_t).clamp(max=5.0)) * (-torch.log(p_t))).mean()

class WCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float32)
        w = c.max() / c
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(w).to(cfg.DEVICE))
    def forward(self, logits, targets): return self.criterion(logits, targets)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class LDAMLoss(nn.Module):
    def __init__(self, max_m=0.5, s=30):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float32)
        m_list = 1.0 / np.sqrt(np.sqrt(c))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).to(cfg.DEVICE)
        self.s = s
    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = logits - batch_m
        output = torch.where(index, x_m, logits)
        return F.cross_entropy(self.s * output, targets)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        intersection = torch.sum(probs * targets_one_hot, dim=1)
        cardinality = torch.sum(probs ** 2, dim=1) + torch.sum(targets_one_hot ** 2, dim=1)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice).mean()

class DBMLoss(nn.Module):
    def __init__(self, scale=30.0, base_m=0.5):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float32)
        self.mc = torch.FloatTensor(base_m * (c.max() / c)).to(cfg.DEVICE)
        self.s = scale
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(1, targets.view(-1,1)).view(-1)
        mi = (1.0 - true_probs).detach() * 0.2
        
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        
        batch_mc = self.mc[targets]
        margin = batch_mc + mi
        
        output = torch.where(index, logits - margin.view(-1,1), logits)
        return F.cross_entropy(self.s * output, targets)

class ALPALoss(nn.Module):
    def __init__(self, a0=1.0, a1=2.0, b1=1.5):
        super().__init__()
        self.a0 = a0
        self.a1 = a1
        self.b1 = b1
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        weight = (self.a0 + self.a1 * (1 - pt)) / (1 + self.b1 * (1 - pt))
        return (weight * ce_loss).mean()

class RobustFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, margin=0.2):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(1, targets.view(-1,1)).view(-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        hill_weight = (true_probs < (1 - self.margin)).float()
        focal_weight = ((1 - true_probs) ** self.gamma)
        
        loss = focal_weight * hill_weight * ce_loss
        return loss.mean()

class PublishedDCALoss(nn.Module):
    def __init__(self, m=0.3, s=30.0, lambda_mdca=0.1):
        super().__init__()
        self.m = m
        self.s = s
        self.lambda_mdca = lambda_mdca
        
    def forward(self, logits, targets, w_active=None):
        norms = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        cosine = logits / norms
        
        index = torch.zeros_like(cosine, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        cosine_m = cosine - self.m
        
        output = torch.where(index, cosine_m, cosine) * self.s
        
        ce_loss = F.cross_entropy(output, targets, weight=w_active)
        
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1)[0]
        preds = probs.argmax(dim=1)
        correct = (preds == targets).float()
        
        mdca_loss = 0.0
        unique_classes = torch.unique(targets)
        for c in unique_classes:
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_conf = conf[class_mask].mean()
                class_acc = correct[class_mask].mean()
                weight = w_active[c] if w_active is not None else 1.0
                mdca_loss += weight * torch.abs(class_conf - class_acc)
                
        mdca_loss = mdca_loss / max(1, len(unique_classes))
        
        return ce_loss + self.lambda_mdca * mdca_loss


class LogitAdjustedLoss(nn.Module):
    """Menon et al. (ICLR 2021) Logit-Adjusted training loss: CE(z + tau * log pi, y)."""
    def __init__(self, tau=1.0, class_counts=None, device=None):
        super().__init__()
        c = np.array(class_counts if class_counts is not None else cfg.CLASS_COUNTS, dtype=np.float64)
        pi = c / c.sum()
        dev = device if device is not None else cfg.DEVICE
        self.offset = torch.tensor(tau * np.log(pi + 1e-12), dtype=torch.float32, device=dev)

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.offset, targets)


class BalancedSoftmaxLoss(nn.Module):
    """Ren et al. (NeurIPS 2020) Balanced Meta-Softmax loss: CE(z + log N_c, y)."""
    def __init__(self, class_counts=None, device=None):
        super().__init__()
        c = np.array(class_counts if class_counts is not None else cfg.CLASS_COUNTS, dtype=np.float64)
        dev = device if device is not None else cfg.DEVICE
        self.log_n = torch.tensor(np.log(np.maximum(c, 1.0)), dtype=torch.float32, device=dev)

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.log_n, targets)


class LDAMDRWLoss(nn.Module):
    """Cao et al. (NeurIPS 2019) LDAM loss with Deferred Re-Weighting (DRW) at 80% of training."""
    def __init__(self, s=30.0, max_m=0.5, beta=0.9999, class_counts=None, device=None):
        super().__init__()
        c = np.array(class_counts if class_counts is not None else cfg.CLASS_COUNTS, dtype=np.float64)
        dev = device if device is not None else cfg.DEVICE
        m = 1.0 / np.sqrt(np.sqrt(c))
        m = m * (max_m / m.max())
        w = (1.0 - beta) / (1.0 - np.power(beta, c) + 1e-8)
        w = w / w.sum() * len(c)
        self.m = torch.tensor(m, dtype=torch.float32, device=dev)
        self.w = torch.tensor(w, dtype=torch.float32, device=dev)
        self.s = s

    def forward(self, logits, targets, drw=False):
        oh = F.one_hot(targets, logits.size(1)).bool()
        out = torch.where(oh, logits - self.m[targets][:, None], logits)
        return F.cross_entropy(self.s * out, targets, weight=self.w if drw else None)


class LWSHead(nn.Module):
    """Kang et al. (ICLR 2020) Learnable Weight Scaling (LWS) on a frozen linear classifier."""
    def __init__(self, clf, num_classes=None):
        super().__init__()
        self.clf = clf
        for p in self.clf.parameters():
            p.requires_grad_(False)
        C = num_classes if num_classes is not None else cfg.NUM_CLASSES
        self.scale = nn.Parameter(torch.ones(C))

    def forward(self, f):
        return self.clf(f) * self.scale


class CosineClassifier(nn.Module):
    """L2-normalized cosine similarity linear head (standard for LDAM / margin losses)."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
