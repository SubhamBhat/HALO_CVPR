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


from core.baselines import *
from core.halo_loss import *
from data.datasets import *

class Config:
    # -------------------------------------------------------------------------
    # DATASET SELECTION
    # -------------------------------------------------------------------------
    # Available datasets:
    #   "cifar10_lt"      - CIFAR-10 Long-Tail (10 classes, 50:1 synthetic imbalance, 32x32)
    #   "cifar100_lt"     - CIFAR-100 Long-Tail (100 classes, 50:1 synthetic imbalance, 32x32)
    #   "oxford_pet"      - Oxford-IIIT Pet (37 breeds, 50:1 synthetic imbalance, 224x224)
    #   "cub200"          - CUB-200-2011 Birds (200 classes, 50:1 synthetic imbalance, 224x224)
    #   "stanford_cars"   - Stanford Cars (196 classes, 50:1 synthetic imbalance, 224x224)
    #   "ham10000"        - HAM10000 Skin Lesions (7 classes, NATURAL ~58:1 imbalance, 224x224)
    #
    # Kaggle Upload Guide (To save time & avoid internet downloads):
    #   cifar10_lt      → Search & add "cifar-10-python" from Kaggle Datasets
    #   cifar100_lt     → Search & add "cifar-100-python" from Kaggle Datasets
    #   oxford_pet      → Search & add "oxford-iiit-pet" from Kaggle Datasets
    #   cub200          → Search & add "CUB_200_2011" from Kaggle Datasets
    #   stanford_cars   → Search & add "stanford-cars-dataset" from Kaggle Datasets
    #   ham10000        → Search & add "skin-cancer-mnist-ham10000" from Kaggle Datasets
    # -------------------------------------------------------------------------
    DATASETS = ["cifar10_lt", "cifar100_lt", "oxford_pet", "cub200", "stanford_cars"]
    DATASET = "cifar10_lt"  # Default
    
    # -------------------------------------------------------------------------
    # BACKBONE & HARDWARE CONFIGURATION
    # -------------------------------------------------------------------------
    # Available backbones (all ImageNet pre-trained):
    #   "squeezenet"       -  1.2M params | Batch 128 | Fire Modules
    #   "shufflenet"       -  2.3M params | Batch 128 | Channel Shuffle
    #   "mobilenet_v2"     -  2.2M params | Batch 128 | Depthwise Separable
    #   "efficientnet_b0"  -  5.3M params | Batch  64 | Compound Scaling
    #   "densenet121"      -  8.0M params | Batch  32 | Dense Connections
    #   "resnet18"         - 11.7M params | Batch 128 | Residual Blocks
    #   "resnet50"         - 25.6M params | Batch  32 | Deep Residuals
    #   "resnext50_32x4d"  - 25.0M params | Batch  32 | Grouped Convolutions
    #   "convnext_tiny"    - 28.0M params | Batch  64 | Modern/LayerNorm
    #   "alexnet"          - 61.1M params | Batch 128 | Legacy FC (needs 224x224)
    #   "vgg16"            - 138M params  | Batch  64 | Deep Sequential (needs 224x224)
    #   "googlenet"        -  6.8M params | Batch  64 | Inception Modules (needs 224x224)
    # -------------------------------------------------------------------------
    KAGGLE_INPUT = "/kaggle/input"
    SAVE_DIR = "/kaggle/working/results"
    
    # Run these backbones sequentially (Top 3 diverse architectures)
    BACKBONES = ["resnet18", "densenet121", "mobilenet_v2", "efficientnet_b0", "squeezenet"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    IMAGE_SIZE = 224  # All torchvision pretrained models expect 224x224
    
    def __init__(self):
        self.set_backbone(self.BACKBONES[0])
        
    def set_backbone(self, backbone_name):
        self.BACKBONE = backbone_name
        # Auto-set optimal batch size for Kaggle 16GB GPUs based on model
        sizes = {
            "squeezenet": 128, "shufflenet": 128, "mobilenet_v2": 128, "alexnet": 128,
            "resnet18": 128, "vgg16": 64, "googlenet": 64, "convnext_tiny": 64,
            "efficientnet_b0": 64, "densenet121": 32, "resnet50": 32, "resnext50_32x4d": 32
        }
        self.BATCH_SIZE = sizes.get(self.BACKBONE, 64)
        
    # These are defaults for HAM10000; auto-updated by setup_data()
    NUM_CLASSES = 7
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    CLASS_COUNTS = [327, 514, 1099, 115, 1113, 6705, 142] 
    
    # -------------------------------------------------------------------------
    # CIFAR-10 LONG-TAIL CONFIGURATION (Experiment 1)
    # -------------------------------------------------------------------------
    CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                           "dog", "frog", "horse", "ship", "truck"]
    CIFAR10_SAMPLES_PER_CLASS = [5000, 5000, 5000, 5000, 5000,
                                  500,  500,  500,  100,  100]
    
    # -------------------------------------------------------------------------
    # CIFAR-100 LONG-TAIL CONFIGURATION
    # -------------------------------------------------------------------------
    CIFAR100_IMB_FACTOR = 1.0 / 50.0  # 50:1 imbalance ratio
    CIFAR100_MAX_SAMPLES = 500        # Max samples for the head class
    
    # -------------------------------------------------------------------------
    # OTHER DATASETS SYNTHETIC IMBALANCE CONFIGURATIONS (50:1)
    # -------------------------------------------------------------------------
    OXFORD_PET_IMB_FACTOR = 1.0 / 50.0
    OXFORD_PET_MAX_SAMPLES = 100
    
    CUB200_IMB_FACTOR = 1.0 / 50.0
    CUB200_MAX_SAMPLES = 30
    
    STANFORD_CARS_IMB_FACTOR = 1.0 / 50.0
    STANFORD_CARS_MAX_SAMPLES = 40
    
    # -------------------------------------------------------------------------
    # HAM10000 REDUCTION
    # -------------------------------------------------------------------------
    HAM_REDUCTION_FACTOR = 0.3  # Keep 30% of training data proportionately
    # -------------------------------------------------------------------------
    # DCAL (Now 100% Parameter-Free & Data-Driven)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # EMA ABLATION (Experiment 3)
    # When use_ema=False, raw F1/ECE values drive the weight update.
    # When use_ema=True (default DCAL), EMA-smoothed values are used.
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # RL PENALTY SCHEDULER (Experiment 4)
    # -------------------------------------------------------------------------
    RL_EPSILON = 0.3
    RL_LR = 0.1
    RL_GAMMA = 0.9
    RL_ACTIONS = [0.0, 0.01, 0.03, 0.05, 0.1]
    
    # -------------------------------------------------------------------------
    # HALO IMPROVEMENTS (Experiment 6)
    # Combines: Cosine LR, Lower EMA (0.4), Constant λ, DRW, Logit Adj.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # METHODS TO RUN
    # -------------------------------------------------------------------------    # Methods to run sequentially in the pipeline.
    METHODS_TO_RUN = ["HALO"]

cfg = Config()

class BackboneWithFeatures(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x):
        return self.classifier(self.backbone(x))

def get_model(name):
    """Instantiates the requested backbone with a custom classification head."""
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        classifier = nn.Linear(m.fc.in_features, cfg.NUM_CLASSES)
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        classifier = nn.Linear(m.fc.in_features, cfg.NUM_CLASSES)
    elif name == 'alexnet':
        m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        fc_layers = list(m.classifier.children())[:-1]
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(), *fc_layers)
        classifier = nn.Linear(4096, cfg.NUM_CLASSES)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, cfg.NUM_CLASSES))
    elif name == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, cfg.NUM_CLASSES))
    elif name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        classifier = nn.Linear(m.classifier.in_features, cfg.NUM_CLASSES)
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        classifier = nn.Linear(m.classifier[2].in_features, cfg.NUM_CLASSES)
    elif name == 'googlenet':
        m = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, transform_input=False)
        backbone = nn.Sequential(*list(m.children())[:-2], nn.AdaptiveAvgPool2d(1), nn.Flatten())
        classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, cfg.NUM_CLASSES))
    elif name == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        fc_layers = list(m.classifier.children())[:-1]
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(), *fc_layers)
        classifier = nn.Linear(4096, cfg.NUM_CLASSES)
    elif name == 'resnext50_32x4d':
        m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        classifier = nn.Linear(m.fc.in_features, cfg.NUM_CLASSES)
    elif name == 'squeezenet':
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        classifier = nn.Linear(512, cfg.NUM_CLASSES)
    elif name == 'shufflenet':
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        children = list(m.children())
        backbone = nn.Sequential(*children[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())
        classifier = nn.Linear(m.fc.in_features, cfg.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    return BackboneWithFeatures(backbone, classifier)

def evaluate(model, loader):
    """Evaluates model and computes F1, ECE, Silhouette, and Shannon Entropy."""
    model.eval()
    feats, logits, tgts = [], [], []
    with torch.no_grad():
        for x, y in loader:
            f = model.backbone(x.to(cfg.DEVICE))
            lg = model.classifier(f)
            feats.append(f.cpu().numpy()); logits.append(lg.cpu().numpy()); tgts.append(y.numpy())
    feats, logits, tgts = np.concatenate(feats), np.concatenate(logits), np.concatenate(tgts)

    # Filter out any unlabeled samples (target == -1) from Stanford Cars test set
    valid_mask = tgts >= 0
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  [evaluate] Filtering {n_invalid} samples with invalid label (-1) from test set.")
        feats, logits, tgts = feats[valid_mask], logits[valid_mask], tgts[valid_mask]

    val_loss = F.cross_entropy(torch.FloatTensor(logits), torch.LongTensor(tgts)).item()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    
    # 1. Global Accuracy
    acc = accuracy_score(tgts, preds)
    
    # 2. Per-class F1 & ECE
    f1s = np.zeros(cfg.NUM_CLASSES)
    eces = np.zeros(cfg.NUM_CLASSES)
    rep = classification_report(tgts, preds, output_dict=True, zero_division=0)
    for c in range(cfg.NUM_CLASSES):
        f1s[c] = rep.get(str(c), {}).get('f1-score', 0)
        mask = (preds == c)
        if mask.sum() > 0:
            c_conf, c_acc = probs[mask, c], (tgts[mask] == c).astype(float)
            bins = np.linspace(0, 1, 16)
            for i in range(15):
                bm = (c_conf > bins[i]) & (c_conf <= bins[i+1])
                if bm.sum() > 0: eces[c] += bm.mean() * abs(c_acc[bm].mean() - c_conf[bm].mean())
                
    # 3. Global ECE
    confidences = probs.max(axis=1)
    global_ece = 0.0
    bins = np.linspace(0, 1, 16)
    for i in range(15):
        bm = (confidences > bins[i]) & (confidences <= bins[i+1])
        if bm.sum() > 0:
            bin_acc = (preds[bm] == tgts[bm]).mean()
            bin_conf = confidences[bm].mean()
            global_ece += (bm.sum() / len(confidences)) * abs(bin_acc - bin_conf)
            
    # 4. Silhouette Score
    sil_score = silhouette_score(feats, tgts) if len(np.unique(tgts)) > 1 else 0.0
    
    # 5. Shannon Entropy Stability Analysis (Experiment 5)
    log_probs = np.log(probs + 1e-12)
    per_sample_entropy = -np.sum(probs * log_probs, axis=1)
    
    mean_entropy = per_sample_entropy.mean()
    
    class_entropies = np.zeros(cfg.NUM_CLASSES)
    for c in range(cfg.NUM_CLASSES):
        cmask = (tgts == c)
        if cmask.sum() > 0:
            class_entropies[c] = per_sample_entropy[cmask].mean()
    
    correct_mask = (preds == tgts)
    H_correct = per_sample_entropy[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
    H_wrong = per_sample_entropy[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0.0
    entropy_sep = H_wrong - H_correct
    
    return (f1s, eces, feats, probs, tgts, acc, global_ece, sil_score,
            mean_entropy, class_entropies, entropy_sep, val_loss)

def run_pipeline():
    """Main execution loop that trains all methods and compares them."""
    train_loader, val_loader, test_loader = setup_data()
    results = {}
    class_entropies_all = {}
    
    for method in cfg.METHODS_TO_RUN:
        print(f"\n{'='*60}\nTraining {method} with {cfg.BACKBONE}\n{'='*60}")
        
        model = get_model(cfg.BACKBONE).to(cfg.DEVICE)
        
        # Setup loss / managers
        crit = None
        dw_manager = None
        rl_scheduler = None
        is_dcal = method in ("DCAL", "DCAL_NO_EMA", "DCAL_RL", "HALO")
        is_v2 = (method == "HALO")
        
        if method == "CE":
            crit = CrossEntropyLossBaseline().to(cfg.DEVICE)
        elif method == "WCE":
            crit = WCELoss().to(cfg.DEVICE)
        elif method == "Focal":
            crit = FocalLoss().to(cfg.DEVICE)
        elif method == "LDAM":
            crit = LDAMLoss().to(cfg.DEVICE)
        elif method == "CB":
            crit = ClassBalancedLoss().to(cfg.DEVICE)
        elif method == "DWB":
            crit = DWBLoss().to(cfg.DEVICE)
        elif method == "Dice":
            crit = DiceLoss().to(cfg.DEVICE)
        elif method == "DBM":
            crit = DBMLoss().to(cfg.DEVICE)
        elif method == "ALPA":
            crit = ALPALoss().to(cfg.DEVICE)
        elif method == "RobustFocal":
            crit = RobustFocalLoss().to(cfg.DEVICE)
        elif method == "LA":
            crit = LogitAdjustedLoss().to(cfg.DEVICE)
        elif method == "BalancedSoftmax":
            crit = BalancedSoftmaxLoss().to(cfg.DEVICE)
        elif method == "LDAM-DRW":
            crit = LDAMDRWLoss().to(cfg.DEVICE)
        elif method == "DCAL":
            crit = PublishedDCALoss().to(cfg.DEVICE)
            dw_manager = DynamicWeightManager(use_ema=True)
        elif method in ("DCAL_NO_EMA", "DCAL_RL", "HALO"):
            crit = HALOLoss().to(cfg.DEVICE)
            dw_manager = DynamicWeightManager(use_ema=(method != "DCAL_NO_EMA"))
            if method == "DCAL_RL": rl_scheduler = RLPenaltyScheduler()

        # V5 Optimizer Setup (Added weight_decay for tiny datasets like Stanford Cars)
        opt = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=1e-6)
            
        best_f1 = 0.0
        current_macro_f1 = 0.0
        train_acc_history = [0.0]  # Start with 0 accuracy
        
        for ep in range(1, cfg.EPOCHS + 1):
            model.train()
            
            # Gamma represents the network's current convergence state (User's original train^2 algo)
            gamma = (train_acc_history[-1]) ** 2 if is_v2 else 0.0
            
            # --- Get dynamic weights ---
            w_active = None
            if is_dcal and dw_manager is not None:
                w_active = torch.as_tensor(dw_manager.get_active_weights(gamma), dtype=torch.float32, device=cfg.DEVICE)
            
            # --- Accumulate training F1/ECE for DCAL weight updates ---
            train_logits_list = []
            train_targets_list = []
            
            # --- Batch Loop ---
            for x, y in train_loader:
                opt.zero_grad()
                out = model(x.to(cfg.DEVICE))
                
                if is_dcal:
                    loss = crit(out, y.to(cfg.DEVICE), w_active=w_active, gamma=gamma)
                    train_logits_list.append(out.detach().cpu().numpy())
                    train_targets_list.append(y.numpy())
                elif method == "LDAM-DRW":
                    loss = crit(out, y.to(cfg.DEVICE), drw=(ep > int(0.8 * cfg.EPOCHS)))
                else:
                    loss = crit(out, y.to(cfg.DEVICE))
                    
                loss.backward(); opt.step()
            
            # Step the LR scheduler
            scheduler.step()
            
            # --- DCAL: Update dynamic weights using TRAINING metrics ---
            if is_dcal and dw_manager is not None and len(train_logits_list) > 0:
                t_logits = np.concatenate(train_logits_list)
                t_tgts = np.concatenate(train_targets_list)
                t_probs = np.exp(t_logits - t_logits.max(axis=1, keepdims=True))
                t_probs = t_probs / t_probs.sum(axis=1, keepdims=True)
                t_preds = t_probs.argmax(axis=1)
                
                # Calculate True Training Accuracy for gating
                train_acc = (t_preds == t_tgts).mean()
                train_acc_history.append(train_acc)
                
                t_f1s = np.zeros(cfg.NUM_CLASSES)
                t_eces = np.zeros(cfg.NUM_CLASSES)
                rep = classification_report(t_tgts, t_preds, output_dict=True, zero_division=0)
                for c in range(cfg.NUM_CLASSES):
                    t_f1s[c] = rep.get(str(c), {}).get('f1-score', 0)
                    mask = (t_preds == c)
                    if mask.sum() > 0:
                        c_conf = t_probs[mask, c]
                        c_acc = (t_tgts[mask] == c).astype(float)
                        bins = np.linspace(0, 1, 16)
                        for i in range(15):
                            bm = (c_conf > bins[i]) & (c_conf <= bins[i+1])
                            if bm.sum() > 0:
                                t_eces[c] += bm.mean() * abs(c_acc[bm].mean() - c_conf[bm].mean())
                
                # Update dynamic weights with F1-driven momentum
                dw_manager.update(t_f1s, t_eces)
            
            # --- Evaluate on Validation Set ---
            eval_out = evaluate(model, val_loader)
            f1s, eces = eval_out[0], eval_out[1]
            val_acc, val_ece, val_sil = eval_out[5], eval_out[6], eval_out[7]
            val_entropy, val_entropy_sep = eval_out[8], eval_out[10]
            
            # Checkpoint
            macro_f1 = f1s.mean()
            is_best = macro_f1 > best_f1
            if is_best:
                best_f1 = macro_f1
                torch.save(model.state_dict(), f"{cfg.SAVE_DIR}/best_{method}.pth")
                
            # Always print if it's a new best
            if is_best:
                best_str = "New Best "
                # Log string formatting
                stage_str = f" [Gamma: {gamma:.2f}, TrainAcc: {train_acc_history[-1]:.2f}]" if is_v2 else ""
                print(f"Ep {ep:02d} | {best_str}Macro F1: {macro_f1:.4f} | Acc: {val_acc:.4f} | ECE: {val_ece:.4f} | Entropy: {val_entropy:.4f} | Sep: {val_entropy_sep:.4f}{stage_str}")
                
                # No explicit stage triggers needed. The weights adapt seamlessly via TrainAcc Gating.
                
        # --- Final Test Set Evaluation ---
        print(f"\nEvaluating Best {method} Model on Test Set...")
        model.load_state_dict(torch.load(f"{cfg.SAVE_DIR}/best_{method}.pth"))
        test_out = evaluate(model, test_loader)
        test_f1s, test_eces = test_out[0], test_out[1]
        test_feats, test_probs, test_tgts = test_out[2], test_out[3], test_out[4]
        test_acc, test_ece, test_sil = test_out[5], test_out[6], test_out[7]
        test_entropy, test_class_ent, test_ent_sep = test_out[8], test_out[9], test_out[10]

        raw_f1, raw_acc, raw_ece = float(test_f1s.mean()), float(test_acc), float(test_ece)
        best_tau = 0.0
        
        # Shared Post-Hoc Grid Search for optimal tau across ALL methods (Plan #1)
        if getattr(cfg, "CALIBRATE_ALL_METHODS", True) or is_v2:
            print("  Searching for optimal tau via grid search on validation set...")
            
            # Collect raw validation logits
            model.eval()
            val_lg_list, val_tg_list = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    val_lg_list.append(model(x.to(cfg.DEVICE)).cpu().numpy())
                    val_tg_list.append(y.numpy())
            val_logits_np = np.concatenate(val_lg_list)
            val_tgts_np = np.concatenate(val_tg_list)
            
            # Precompute log priors
            class_priors = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
            class_priors = class_priors / class_priors.sum()
            log_priors_np = np.log(class_priors + 1e-12)
            
            # Grid search: find tau that maximizes MACRO F1 on val set 
            best_tau, best_val_f1 = 0.0, 0.0
            
            for tau_candidate in np.arange(0.0, 3.05, 0.05):
                offset = tau_candidate * (-log_priors_np)  # Boost minority classes
                adj_lg = val_logits_np + offset
                adj_preds = adj_lg.argmax(axis=1)
                rep = classification_report(val_tgts_np, adj_preds, output_dict=True, zero_division=0)
                macro_f1 = rep['macro avg']['f1-score']
                if macro_f1 > best_val_f1:
                    best_val_f1 = macro_f1
                    best_tau = float(tau_candidate)
            
            print(f"  Optimal tau={best_tau:.2f} (val macro F1={best_val_f1:.4f})")
            
            # Apply optimal tau to TEST set
            test_lg_list, test_tg_list = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    test_lg_list.append(model(x.to(cfg.DEVICE)).cpu().numpy())
                    test_tg_list.append(y.numpy())
            raw_test_logits = np.concatenate(test_lg_list)
            raw_test_tgts = np.concatenate(test_tg_list)
            valid_m = raw_test_tgts >= 0
            raw_test_logits, raw_test_tgts = raw_test_logits[valid_m], raw_test_tgts[valid_m]
            
            optimal_offset = best_tau * (-log_priors_np)
            adj_logits = raw_test_logits + optimal_offset
            adj_probs = np.exp(adj_logits - adj_logits.max(axis=1, keepdims=True))
            adj_probs = adj_probs / adj_probs.sum(axis=1, keepdims=True)
            adj_preds = adj_probs.argmax(axis=1)
            
            # Recompute all metrics with adjusted predictions
            adj_acc = accuracy_score(raw_test_tgts, adj_preds)
            adj_rep = classification_report(raw_test_tgts, adj_preds, output_dict=True, zero_division=0)
            adj_f1s = np.array([adj_rep.get(str(c), {}).get('f1-score', 0) for c in range(cfg.NUM_CLASSES)])
            
            adj_conf = adj_probs.max(axis=1)
            adj_ece = 0.0
            bins = np.linspace(0, 1, 16)
            for i in range(15):
                bm = (adj_conf > bins[i]) & (adj_conf <= bins[i+1])
                if bm.sum() > 0:
                    adj_ece += (bm.sum() / len(adj_conf)) * abs((adj_preds[bm] == raw_test_tgts[bm]).mean() - adj_conf[bm].mean())
            
            adj_log_probs = np.log(adj_probs + 1e-12)
            adj_entropy_per = -np.sum(adj_probs * adj_log_probs, axis=1)
            adj_entropy = adj_entropy_per.mean()
            adj_correct = (adj_preds == raw_test_tgts)
            adj_H_c = adj_entropy_per[adj_correct].mean() if adj_correct.sum() > 0 else 0.0
            adj_H_w = adj_entropy_per[~adj_correct].mean() if (~adj_correct).sum() > 0 else 0.0
            adj_ent_sep = adj_H_w - adj_H_c
            
            # Override test results with adjusted ones
            test_f1s = adj_f1s
            test_acc = adj_acc
            test_ece = adj_ece
            test_entropy = adj_entropy
            test_ent_sep = adj_ent_sep
            test_probs = adj_probs
            test_tgts = raw_test_tgts
            
            for c in range(cfg.NUM_CLASSES):
                cmask = (raw_test_tgts == c)
                if cmask.sum() > 0:
                    test_class_ent[c] = adj_entropy_per[cmask].mean()
        
        print(f"[{method} Final] Raw F1: {raw_f1:.4f} | +tau* F1: {test_f1s.mean():.4f} (tau*={best_tau:.2f}) | "
              f"Acc: {test_acc:.4f} | ECE: {test_ece:.4f} | Entropy: {test_entropy:.4f} | Sep: {test_ent_sep:.4f}")
        
        results[method] = {
            "features": test_feats, "probs": test_probs, "targets": test_tgts,
            "raw_f1": raw_f1, "raw_acc": raw_acc, "raw_ece": raw_ece, "tau": best_tau,
            "f1": test_f1s.mean(), "acc": test_acc, "ece": test_ece, "sil": test_sil,
            "entropy": test_entropy, "entropy_sep": test_ent_sep
        }
        class_entropies_all[method] = test_class_ent
        
        # Save weight trajectory for DCAL variants
        if dw_manager is not None and hasattr(dw_manager, 'wd_history'):
            DCALPlotter.plot_weight_trajectory(dw_manager.wd_history, method)
        
        # Save RL lambda schedule
        if rl_scheduler is not None and hasattr(rl_scheduler, 'lambda_history'):
            DCALPlotter.plot_rl_lambda_schedule(rl_scheduler.lambda_history)
            
        # Add CM and Class Report
        print(f"\n--- Class-wise Report for {method} ---")
        y_pred = test_probs.argmax(axis=1)
        if cfg.NUM_CLASSES <= 50:
            print(classification_report(test_tgts, y_pred, target_names=cfg.CLASS_NAMES, zero_division=0))
        else:
            print(f"Skipping detailed class report (dataset has {cfg.NUM_CLASSES} classes). Summary metrics above.")
            
        DCALPlotter.plot_confusion_matrix(test_tgts, y_pred, method)
        
    # =========================================================================
    # GENERATE ALL PLOTS
    # =========================================================================
    print("\nGenerating Plots...")
    
    our_method = "DCAL" if "DCAL" in results else list(results.keys())[-1]
    baselines = [m for m in ["CE", "CB", "DWB"] if m in results]
    
    for b in baselines:
        DCALPlotter.plot_tsne(results[b]["features"], results[our_method]["features"],
                              results[our_method]["targets"], b, our_method)
        DCALPlotter.plot_reliability(results[our_method]["targets"], results[b]["probs"],
                                     results[our_method]["probs"], b, our_method)
    
    # Per-class entropy comparison (Experiment 5)
    DCALPlotter.plot_class_entropy(class_entropies_all)
        
    # =========================================================================
    # FINAL RESULTS TABLE
    # =========================================================================
    print(f"\n{'='*110}")
    print(f"{'FINAL EXPERIMENTAL RESULTS SUMMARY (RAW vs +TAU*)':^110}")
    print(f"{'='*110}")
    print(f"{'Method':<16} | {'Raw F1':<8} | {'+tau* F1':<9} | {'tau*':<5} | {'Accuracy':<9} | {'ECE':<9} | {'Silhouette':<10} | {'Entropy':<9} | {'Ent.Sep.'}")
    print(f"{'-'*110}")
    for m in cfg.METHODS_TO_RUN:
        if m in results:
            r = results[m]
            print(f"{m:<16} | {r['raw_f1']:.4f}   | {r['f1']:.4f}    | {r['tau']:.2f}  | {r['acc']:.4f}    | {r['ece']:.4f}    | "
                  f"{r['sil']:.4f}     | {r['entropy']:.4f}    | {r['entropy_sep']:.4f}")
    print(f"{'='*110}")
    
    print(f"\nDone with {cfg.BACKBONE}! Check {cfg.SAVE_DIR}")


if __name__ == '__main__':
    for d_name in cfg.DATASETS:
        cfg.DATASET = d_name
        for b_name in cfg.BACKBONES:
            cfg.set_backbone(b_name)
            run_pipeline()

