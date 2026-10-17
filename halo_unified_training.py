import os
import glob
import math
import copy
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import torchvision.models as models

# Fix for Docker /dev/shm limits while keeping high num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
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
    SAVE_DIR = os.environ.get("HALO_SAVE_DIR") or ("/kaggle/working/results" if os.path.isdir("/kaggle/working") else "./results")
    
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
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# =============================================================================
# 2A. HAM10000 DATASET LOADING
# =============================================================================
class HAM10000Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_transforms(allow_vflip=False):
    """Defines image augmentations based on ImageNet standards."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf_list = [
        transforms.Resize((cfg.IMAGE_SIZE + 32, cfg.IMAGE_SIZE + 32)),
        transforms.RandomCrop(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
    ]
    if allow_vflip:
        train_tf_list.append(transforms.RandomVerticalFlip())
        
    train_tf_list.extend([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_tf = transforms.Compose(train_tf_list)
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf

def setup_ham10000():
    """Finds HAM10000 images on Kaggle, splits data, and builds DataLoaders."""
    print("Setting up HAM10000 data...")
    
    meta_path = None
    for pat in ["**/HAM10000_metadata*", "**/hmnist_metadata*"]:
        candidates = glob.glob(os.path.join(cfg.KAGGLE_INPUT, pat), recursive=True)
        for p in candidates:
            if "image_id" in pd.read_csv(p, nrows=1).columns:
                meta_path = p
                break
        if meta_path: break
    if not meta_path: raise FileNotFoundError("HAM10000 metadata not found!")
    
    print(f"  Metadata: {meta_path}")
    df = pd.read_csv(meta_path)
    print(f"  Metadata rows: {len(df)}")
    
    # Search for images ONLY in the same dataset folder (not all of /kaggle/input)
    dataset_root = os.path.dirname(meta_path)
    image_map = {}
    for root, dirs, files in os.walk(dataset_root):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_map[os.path.splitext(fname)[0]] = os.path.join(root, fname)
    print(f"  Images found: {len(image_map)}")
            
    df["image_path"] = df["image_id"].map(image_map)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df["label"] = df["dx"].map(cfg.CLASS_LABELS)
    print(f"  Matched rows: {len(df)}")
    
    cfg.CLASS_COUNTS = [df[df["label"] == i].shape[0] for i in range(cfg.NUM_CLASSES)]
    print(f"  Class counts: {dict(zip(cfg.CLASS_NAMES, cfg.CLASS_COUNTS))}")
    
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    
    # Proportional reduction of training set to speed up experiments
    if hasattr(cfg, 'HAM_REDUCTION_FACTOR') and cfg.HAM_REDUCTION_FACTOR < 1.0:
        reduced_dfs = []
        for label_val in range(cfg.NUM_CLASSES):
            class_df = train_df[train_df["label"] == label_val]
            n_keep = max(1, int(len(class_df) * cfg.HAM_REDUCTION_FACTOR))
            reduced_dfs.append(class_df.sample(n=n_keep, random_state=42))
        train_df = pd.concat(reduced_dfs).reset_index(drop=True)
        print(f"  HAM10000 reduced to {len(train_df)} training samples ({cfg.HAM_REDUCTION_FACTOR*100:.0f}%)")
    
    train_df, val_df = train_test_split(train_df, test_size=0.15/0.85, stratify=train_df["label"], random_state=42)
    print(f"  Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    train_tf, eval_tf = get_transforms(allow_vflip=True)
    mk_dl = lambda d, tf, shuf: DataLoader(
        HAM10000Dataset(d["image_path"].tolist(), d["label"].tolist(), tf),
        batch_size=cfg.BATCH_SIZE, shuffle=shuf, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    print("  DataLoaders ready.\n")
    return mk_dl(train_df, train_tf, True), mk_dl(val_df, eval_tf, False), mk_dl(test_df, eval_tf, False)

# =============================================================================
# 2B. CIFAR-10 LONG-TAIL DATASET LOADING (Experiment 1)
# =============================================================================
def setup_cifar10_lt():
    """Creates an artificially imbalanced CIFAR-10 dataset (50:1 ratio)."""
    print("Setting up CIFAR-10 Long-Tail data...")
    
    cfg.NUM_CLASSES = 10
    cfg.CLASS_NAMES = cfg.CIFAR10_CLASS_NAMES
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    # AlexNet, VGG, and GoogLeNet physically crash on 32x32 due to their rigid pooling/kernel sizes.
    if cfg.BACKBONE in ["alexnet", "vgg16", "googlenet"]:
        cfg.IMAGE_SIZE = 224
    else:
        cfg.IMAGE_SIZE = 32
        
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    
    # If 32x32, padding=4. If 224x224, padding=28
    pad_size = 4 if cfg.IMAGE_SIZE == 32 else 28
    
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomCrop(cfg.IMAGE_SIZE, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Try loading from Kaggle dataset first (no internet needed)
    # Add "cifar-10-python" or "cifar10" dataset in Kaggle to use this
    cifar_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/data_batch_1"), recursive=True):
        if os.path.isfile(candidate):
            cifar_local = os.path.dirname(candidate)
            break
    
    root = "/tmp/cifar10_data"
    dl = True
    if cifar_local:
        print(f"  Loading CIFAR-10 from local: {cifar_local}")
        dl = False
        target_dir = os.path.join(root, "cifar-10-batches-py")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(cifar_local):
                src = os.path.join(cifar_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local CIFAR-10 found, downloading (enable Internet in Kaggle Settings)...")
    
    full_train = tv_datasets.CIFAR10(root=root, train=True, download=dl)
    full_test = tv_datasets.CIFAR10(root=root, train=False, download=dl)
    
    targets = np.array(full_train.targets)
    selected_indices = []
    for c in range(10):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = cfg.CIFAR10_SAMPLES_PER_CLASS[c]
        chosen = np.random.choice(class_indices, size=min(n_keep, len(class_indices)), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets[train_idx]
    cfg.CLASS_COUNTS = [int((train_targets == c).sum()) for c in range(10)]
    print(f"  Long-Tail Class counts (train): {dict(zip(cfg.CLASS_NAMES, cfg.CLASS_COUNTS))}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = TransformSubset(full_test, list(range(len(full_test))), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

# =============================================================================
# 2C. CIFAR-100 LONG-TAIL DATASET LOADING
# =============================================================================
def setup_cifar100_lt():
    """Creates an artificially imbalanced CIFAR-100 dataset."""
    print("Setting up CIFAR-100 Long-Tail data...")
    
    cfg.NUM_CLASSES = 100
    cfg.CLASS_NAMES = [f"class_{i}" for i in range(100)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    if cfg.BACKBONE in ["alexnet", "vgg16", "googlenet"]:
        cfg.IMAGE_SIZE = 224
    else:
        cfg.IMAGE_SIZE = 32
    
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    pad_size = 4 if cfg.IMAGE_SIZE == 32 else 28
    
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomCrop(cfg.IMAGE_SIZE, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Search locally first, then download
    cifar_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/meta"), recursive=True):
        if "cifar100" in candidate.lower() or "cifar-100" in candidate.lower():
            cifar_local = os.path.dirname(candidate)
            break
            
    # Check original torchvision format just in case
    if not cifar_local:
        for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/cifar-100-python"), recursive=True):
            if os.path.isdir(candidate):
                cifar_local = os.path.dirname(candidate)
                break
    
    root = "/tmp/cifar100_data"
    dl = True
    if cifar_local:
        print(f"  Loading CIFAR-100 from local: {cifar_local}")
        dl = False
        # Torchvision strictly expects the 'cifar-100-python' directory name inside the root
        target_dir = os.path.join(root, "cifar-100-python")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(cifar_local):
                src = os.path.join(cifar_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local CIFAR-100 found, downloading...")
    
    full_train = tv_datasets.CIFAR100(root=root, train=True, download=dl)
    full_test = tv_datasets.CIFAR100(root=root, train=False, download=dl)
    
    # Create exponential decay long-tail distribution
    targets = np.array(full_train.targets)
    selected_indices = []
    imb_factor = cfg.CIFAR100_IMB_FACTOR
    max_num = cfg.CIFAR100_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_targets == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = TransformSubset(full_test, list(range(len(full_test))), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

# =============================================================================
# 2D. OXFORD-IIIT PET DATASET LOADING
# =============================================================================
def setup_oxford_pet():
    """Loads Oxford-IIIT Pet dataset (37 breeds, natural imbalance)."""
    print("Setting up Oxford-IIIT Pet data...")
    
    cfg.NUM_CLASSES = 37
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"breed_{i}" for i in range(37)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # Search locally first
    pet_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/annotations"), recursive=True):
        if os.path.isdir(candidate):
            pet_local = os.path.dirname(candidate)
            break
    
    root = "/tmp/oxford_pet"
    dl = True
    if pet_local:
        print(f"  Loading Oxford Pet from local: {pet_local}")
        dl = False
        target_dir = os.path.join(root, "oxford-iiit-pet")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(pet_local):
                src = os.path.join(pet_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local Oxford Pet found, downloading...")
    
    full_train = tv_datasets.OxfordIIITPet(root=root, split="trainval", download=dl)
    full_test = tv_datasets.OxfordIIITPet(root=root, split="test", download=dl)
    
    # Create exponential decay long-tail distribution
    targets = np.array([t for _, t in full_train])
    selected_indices = []
    imb_factor = cfg.OXFORD_PET_IMB_FACTOR
    max_num = cfg.OXFORD_PET_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets
    
    train_tgts = train_targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    class FullDatasetTransform(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = FullDatasetTransform(full_test, eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

# =============================================================================
# 2E. CUB-200-2011 DATASET LOADING
# =============================================================================
def setup_cub200():
    """Loads CUB-200-2011 fine-grained bird classification (200 classes)."""
    print("Setting up CUB-200-2011 data...")
    
    cfg.NUM_CLASSES = 200
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"bird_{i}" for i in range(200)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # Search for CUB dataset in Kaggle inputs
    cub_root = None
    is_imagefolder = False
    
    # 1. Search for ImageFolder structure (like muteks/cub-200-2011)
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/train"), recursive=True):
        if "cub" in candidate.lower() or "bird" in candidate.lower():
            parent = os.path.dirname(candidate)
            if os.path.exists(os.path.join(parent, "test")):
                cub_root = parent
                is_imagefolder = True
                break
                
    # 2. Search for original Caltech metadata format
    if cub_root is None:
        for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/images.txt"), recursive=True):
            parent = os.path.dirname(candidate)
            if os.path.exists(os.path.join(parent, "image_class_labels.txt")):
                cub_root = parent
                break
    
    if cub_root is None:
        print("  No local CUB-200 found. Attempting download...")
        cub_root = "/tmp/cub200"
        os.makedirs(cub_root, exist_ok=True)
        import urllib.request, tarfile
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        tar_path = os.path.join(cub_root, "CUB_200_2011.tgz")
        if not os.path.exists(os.path.join(cub_root, "CUB_200_2011")):
            print("  Downloading CUB-200-2011...")
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(cub_root)
        cub_root = os.path.join(cub_root, "CUB_200_2011")
    
    print(f"  Loading CUB-200 from: {cub_root}")
    
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    if is_imagefolder:
        from torchvision import datasets
        train_ds_temp = datasets.ImageFolder(os.path.join(cub_root, "train"))
        test_ds_temp = datasets.ImageFolder(os.path.join(cub_root, "test"))
        for path, label in train_ds_temp.samples:
            train_paths.append(path)
            train_labels.append(label)
        for path, label in test_ds_temp.samples:
            test_paths.append(path)
            test_labels.append(label)
    else:
        # Parse CUB-200 metadata files
        with open(os.path.join(cub_root, "images.txt")) as f:
            id_to_path = {int(line.split()[0]): line.split()[1] for line in f.readlines()}
        with open(os.path.join(cub_root, "image_class_labels.txt")) as f:
            id_to_label = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f.readlines()}
        with open(os.path.join(cub_root, "train_test_split.txt")) as f:
            id_to_split = {int(line.split()[0]): int(line.split()[1]) for line in f.readlines()}
        
        img_dir = os.path.join(cub_root, "images")
        
        for img_id, rel_path in id_to_path.items():
            full_path = os.path.join(img_dir, rel_path)
            label = id_to_label[img_id]
            if id_to_split[img_id] == 1:
                train_paths.append(full_path)
                train_labels.append(label)
            else:
                test_paths.append(full_path)
                test_labels.append(label)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # Create exponential decay long-tail distribution
    selected_indices = []
    imb_factor = cfg.CUB200_IMB_FACTOR
    max_num = cfg.CUB200_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(train_labels == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = train_labels[selected_indices]
    
    # Split train into train/val
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_tgts = train_labels[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class SubsetByIndex(Dataset):
        def __init__(self, paths, labels, indices, transform):
            self.paths = [paths[i] for i in indices]
            self.labels = [labels[i] for i in indices]
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
    
    class ImageListDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
    
    train_ds = SubsetByIndex(train_paths, train_labels.tolist(), train_idx, train_tf)
    val_ds = SubsetByIndex(train_paths, train_labels.tolist(), val_idx, eval_tf)
    test_ds = ImageListDataset(test_paths, test_labels.tolist(), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

# =============================================================================
# 2F. STANFORD CARS DATASET LOADING
# =============================================================================
class StanfordCarsRawDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        import scipy.io
        import pandas as pd
        
        # Determine image dir
        img_dir_name = "cars_train" if split == "train" else "cars_test"
        img_dir = None
        
        # Deep search for the actual directory containing the JPGs (bypasses nested folders)
        for candidate in glob.glob(os.path.join(root_dir, f"**/{img_dir_name}/**/*.jpg"), recursive=True):
            img_dir = os.path.dirname(candidate)
            break
            
        if not img_dir:
            for candidate in glob.glob(os.path.join(root_dir, f"**/{img_dir_name}"), recursive=True):
                if os.path.isdir(candidate):
                    img_dir = candidate
                    break
        
        if img_dir is None:
            raise FileNotFoundError(f"Could not find {img_dir_name} directory in {root_dir}")
            
        # Try to find CSV annotations first
        csv_path = None
        for candidate in glob.glob(os.path.join(root_dir, f"**/anno_{split}.csv"), recursive=True):
            csv_path = candidate
            break
            
        if csv_path:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Kaggle CSVs sometimes just have the filename
                fname = row.iloc[0]
                # Label is usually 1-indexed in Stanford Cars
                label = int(row.iloc[-1]) - 1
                self.samples.append((os.path.join(img_dir, fname), label))
        else:
            # Fallback to .mat
            mat_path = None
            mat_name = "cars_train_annos.mat" if split == "train" else "cars_test_annos_withlabels.mat"
            for candidate in glob.glob(os.path.join(root_dir, f"**/{mat_name}"), recursive=True):
                mat_path = candidate
                break
                
            if not mat_path:
                # Some test mats are named cars_test_annos.mat (without labels). We need labels!
                for candidate in glob.glob(os.path.join(root_dir, f"**/cars_test_annos.mat"), recursive=True):
                    mat_path = candidate
                    break
                    
            if not mat_path:
                raise FileNotFoundError(f"Could not find annotations for {split}")
                
            annos = scipy.io.loadmat(mat_path)
            if 'annotations' in annos:
                for anno in annos['annotations'][0]:
                    fname = anno[-1][0]
                    # Check if label exists (test set might not have it in some versions)
                    if len(anno) >= 5:
                        label = int(anno[4][0][0]) - 1
                        self.samples.append((os.path.join(img_dir, fname), label))
                    else:
                        self.samples.append((os.path.join(img_dir, fname), 0))
            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

def setup_stanford_cars():
    """Loads Stanford Cars fine-grained classification (196 classes)."""
    print("Setting up Stanford Cars data...")
    
    cfg.NUM_CLASSES = 196
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"car_{i}" for i in range(196)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # 1. Check if user attached dataset to Kaggle
    kaggle_root = None
    if getattr(cfg, 'IS_KAGGLE', True) or os.path.exists("/kaggle/input"):
        base_dir = getattr(cfg, 'KAGGLE_INPUT', "/kaggle/input")
        # Look for imagefolder format first
        for candidate in glob.glob(os.path.join(base_dir, "**/stanford-cars/train"), recursive=True):
            kaggle_root = os.path.dirname(candidate)
            break
        # Look for raw format
        if not kaggle_root:
            for candidate in glob.glob(os.path.join(base_dir, "**/cars_train"), recursive=True):
                kaggle_root = os.path.dirname(os.path.dirname(candidate)) # Go up past cars_train/cars_train if nested
                if not os.path.exists(os.path.join(kaggle_root, "cars_train")):
                     kaggle_root = os.path.dirname(candidate)
                break

    full_train = None
    full_test = None
    
    if kaggle_root:
        print(f"  Found Stanford Cars dataset locally at: {kaggle_root}")
        if os.path.exists(os.path.join(kaggle_root, "train")):
            print("  Using ImageFolder format.")
            full_train = tv_datasets.ImageFolder(os.path.join(kaggle_root, "train"))
            full_test = tv_datasets.ImageFolder(os.path.join(kaggle_root, "test"))
        else:
            print("  Using Raw/CSV/MAT format.")
            full_train = StanfordCarsRawDataset(kaggle_root, split="train")
            full_test = StanfordCarsRawDataset(kaggle_root, split="test")
    else:
        # Fallback to downloading
        root = "/tmp/stanford_cars"
        target_dir = os.path.join(root, "stanford-cars")
        if not os.path.exists(os.path.join(target_dir, "train")):
            print("  Downloading Stanford Cars from Fast.ai AWS mirror...")
            os.makedirs(root, exist_ok=True)
            import urllib.request, tarfile
            url = "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"
            tar_path = os.path.join(root, "stanford-cars.tgz")
            if not os.path.exists(tar_path):
                urllib.request.urlretrieve(url, tar_path)
            print("  Extracting Stanford Cars...")
            with tarfile.open(tar_path) as tf:
                tf.extractall(root)
        
        full_train = tv_datasets.ImageFolder(os.path.join(target_dir, "train"))
        full_test = tv_datasets.ImageFolder(os.path.join(target_dir, "test"))
    
    # Create exponential decay long-tail distribution
    if hasattr(full_train, 'targets'):
        targets = np.array(full_train.targets)
    else:
        targets = np.array([s[1] for s in full_train.samples])
        
    selected_indices = []
    imb_factor = cfg.STANFORD_CARS_IMB_FACTOR
    max_num = cfg.STANFORD_CARS_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    # 3-way stratified split from labeled training data
    # (Official Stanford Cars test set is unlabeled in Kaggle snapshots → all labels = -1)
    train_val_idx, test_idx = train_test_split(
        selected_indices, test_size=0.15, stratify=sel_targets, random_state=42)
    tv_targets = targets[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15/0.85, stratify=tv_targets, random_state=42)

    train_tgts = targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")

    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label

    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds   = TransformSubset(full_train, val_idx,   eval_tf)
    test_ds  = TransformSubset(full_train, test_idx,  eval_tf)

    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)


# =============================================================================
# 2G. DATASET DISPATCHER
# =============================================================================
def setup_data():
    dispatchers = {
        "cifar10_lt": setup_cifar10_lt,
        "cifar100_lt": setup_cifar100_lt,
        "oxford_pet": setup_oxford_pet,
        "cub200": setup_cub200,
        "stanford_cars": setup_stanford_cars,
        "ham10000": setup_ham10000,
    }
    if cfg.DATASET not in dispatchers:
        raise ValueError(f"Unknown dataset: {cfg.DATASET}. Choose from: {list(dispatchers.keys())}")
    return dispatchers[cfg.DATASET]()


# =============================================================================
# 3. BACKBONE REGISTRY (Experiment 2: +SqueezeNet, +ShuffleNet)
# =============================================================================
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

# =============================================================================
# 4. BASELINE LOSS FUNCTIONS
# =============================================================================
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

# =============================================================================
# 5. DCAL LOSS FUNCTION
# =============================================================================

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

class DCALoss(nn.Module):
    def __init__(self):
        super(DCALoss, self).__init__()
        # We no longer hardcode weights here; they are fully managed by DynamicWeightManager
        
    def forward(self, logits, targets, w_active=None, gamma=0.0):
        if w_active is not None:
            # Use reduction='none' + manual multiply — confirmed best on Stanford Cars
            # Increased label_smoothing to 0.2 to prevent overconfidence on 1-shot tail classes
            ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.2)
            w = w_active[targets]
            return (ce * w).mean()
        else:
            return F.cross_entropy(logits, targets, label_smoothing=0.2)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        pi = c / c.sum()
        self.offset = torch.tensor(tau * np.log(pi + 1e-12), dtype=torch.float32, device=cfg.DEVICE)
    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.offset, targets)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        c = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        self.log_n = torch.tensor(np.log(np.maximum(c, 1.0)), dtype=torch.float32, device=cfg.DEVICE)
    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.log_n, targets)


class LDAMDRWLoss(nn.Module):
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

# =============================================================================
# 6. DYNAMIC WEIGHT MANAGER (with EMA ablation — Experiment 3)
# =============================================================================
class DynamicWeightManager:
    def __init__(self, use_ema=False):
        self.c = cfg.NUM_CLASSES
        self.counts = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        
        # Cube-Root Inverse Frequency for early Representation Learning (3.0:1)
        # Sqrt (5.3x) and Linear (28x) amplify noise too much on 1-sample tail classes.
        self.w_rep = (self.counts.max() / self.counts) ** 0.33
        
        # Cube-Root for the base of the Dynamic Penalty (3.0:1)
        # Prevents massive gradient instability on 1-shot validation classes.
        self.w_pen_base = (self.counts.max() / self.counts) ** 0.33
        
        self.w_d = self._norm(self.w_rep)
        self.f_smooth = None
        self.initialized = False
        self.wd_history = []
        
    def _norm(self, w):
        return self.c * w / (w.sum() + 1e-9)
        
    def update(self, val_f1, val_ece):
        self.initialized = True
        
        if not hasattr(self, 'sigma2_f'):
            self.sigma2_f = np.ones(self.c) * 0.01
            self.f_smooth = np.ones(self.c) / self.c
            self.e_smooth = np.ones(self.c) * 0.05
            
        if getattr(self, 'use_ema', True):
            # Variance-weighted EMA smoothing (Critical for tiny val sets like Stanford Cars)
            inv = val_f1 - self.f_smooth
            self.sigma2_f = 0.9 * self.sigma2_f + 0.1 * (inv**2)
            adapt_lr = 0.1 / (0.1 + (self.sigma2_f / (self.sigma2_f.mean() + 1e-9)))
            self.f_smooth += adapt_lr * inv
            self.e_smooth = 0.9 * self.e_smooth + 0.1 * val_ece
        else:
            self.f_smooth = val_f1.copy()
            self.e_smooth = val_ece.copy()
        
        f1_max = self.f_smooth.max()
        if f1_max > 0:
            dynamic_multiplier = 1.0 + np.log((f1_max + 1e-9) / (self.f_smooth + 1e-9))
        else:
            dynamic_multiplier = np.ones_like(self.f_smooth)
            
        self.w_dynamic = self.w_pen_base * dynamic_multiplier
        
    def get_active_weights(self, gamma=0.0):
        if not self.initialized:
            return self._norm(self.w_rep)
        w_active = (1.0 - gamma) * self.w_rep + gamma * self.w_dynamic
        normed_w = self._norm(w_active)
        self.wd_history.append(normed_w.copy())
        return normed_w

# =============================================================================
# 7. RL PENALTY SCHEDULER (Experiment 4)
# =============================================================================
class RLPenaltyScheduler:
    """
    Contextual Bandit for calibration penalty scheduling.
    State:  [F1_1..F1_K, ECE_1..ECE_K, epoch/max_epoch]
    Action: λ ∈ {0, 0.01, 0.03, 0.05, 0.1}
    Reward: ΔF1 (change in validation Macro-F1)
    """
    def __init__(self):
        self.actions = cfg.RL_ACTIONS
        self.n_actions = len(self.actions)
        self.state_dim = 2 * cfg.NUM_CLASSES + 1
        self.W = np.zeros((self.n_actions, self.state_dim))
        self.prev_state = None
        self.prev_action_idx = None
        self.prev_f1 = 0.0
        self.lambda_history = []
        
    def get_state(self, f1s, eces, epoch):
        return np.concatenate([f1s, eces, [epoch / cfg.EPOCHS]])
    
    def select_action(self, state):
        if np.random.rand() < cfg.RL_EPSILON:
            idx = np.random.randint(self.n_actions)
        else:
            q_values = self.W @ state
            idx = np.argmax(q_values)
        self.lambda_history.append(self.actions[idx])
        return idx, self.actions[idx]
    
    def update(self, state, f1):
        reward = f1 - self.prev_f1
        if self.prev_state is not None:
            q_next = np.max(self.W @ state)
            target = reward + cfg.RL_GAMMA * q_next
            q_prev = self.W[self.prev_action_idx] @ self.prev_state
            self.W[self.prev_action_idx] += cfg.RL_LR * (target - q_prev) * self.prev_state
        self.prev_f1 = f1

# =============================================================================
# 8. PLOTTER
# =============================================================================
class DCALPlotter:
    @staticmethod
    def plot_tsne(feat_b, feat_d, lbls, b_name, our_name="DCAL"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        e_b, e_d = tsne.fit_transform(feat_b), tsne.fit_transform(feat_d)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        pal = sns.color_palette("tab10", len(cfg.CLASS_NAMES))
        for i, cls in enumerate(cfg.CLASS_NAMES):
            idx = (lbls == i)
            ax1.scatter(e_b[idx, 0], e_b[idx, 1], label=cls, color=pal[i], alpha=0.6, s=15)
            ax2.scatter(e_d[idx, 0], e_d[idx, 1], label=cls, color=pal[i], alpha=0.6, s=15)
        ax1.set_title(f"{b_name} Model"); ax1.axis('off')
        ax2.set_title(f"{our_name} Model"); ax2.axis('off')
        fig.legend(handles=ax1.get_legend_handles_labels()[0], labels=cfg.CLASS_NAMES,
                   loc='lower center', ncol=min(7, len(cfg.CLASS_NAMES)))
        plt.tight_layout(); plt.savefig(f"{cfg.SAVE_DIR}/tsne_{our_name}_vs_{b_name}.png", dpi=300); plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, method_name):
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"  [plot_confusion_matrix] Skipping — empty test set for {method_name}.")
            return
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 0:
            return
        plt.figure(figsize=(10, 8))
        if cfg.NUM_CLASSES > 20: # Do not plot text annotations for many classes
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES)
        plt.title(f"Confusion Matrix: {method_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{cfg.SAVE_DIR}/cm_{method_name}.png", dpi=300); plt.close()

    @staticmethod
    def plot_reliability(y_t, p_b, p_d, b_name, our_name="DCAL"):
        def cc(y, p):
            c, a = np.max(p, axis=1), (np.argmax(p, axis=1) == y)
            bins = np.linspace(0, 1, 11)
            return ([np.mean(c[(c>bins[i])&(c<=bins[i+1])]) if ((c>bins[i])&(c<=bins[i+1])).sum() > 0 else 0 for i in range(10)],
                    [np.mean(a[(c>bins[i])&(c<=bins[i+1])]) if ((c>bins[i])&(c<=bins[i+1])).sum() > 0 else 0 for i in range(10)])
        cb, ab = cc(y_t, p_b); cd, ad = cc(y_t, p_d)
        plt.figure(figsize=(8,8)); plt.plot([0,1],[0,1],'k--')
        plt.plot(cb, ab, 'ro-', label=b_name); plt.plot(cd, ad, 'bs-', label=our_name)
        plt.legend(); plt.title(f"Reliability: {our_name} vs {b_name}")
        plt.savefig(f"{cfg.SAVE_DIR}/rel_{our_name}_vs_{b_name}.png", dpi=300); plt.close()

    @staticmethod
    def plot_weight_trajectory(wd_history, method_name):
        """Plots per-class dynamic weight W_d over epochs (Experiment 3)."""
        if not wd_history: return
        arr = np.array(wd_history)
        plt.figure(figsize=(14, 6))
        for c in range(cfg.NUM_CLASSES):
            plt.plot(arr[:, c], label=cfg.CLASS_NAMES[c], linewidth=1.5)
        plt.xlabel("Epoch"); plt.ylabel("Dynamic Weight $W_d$")
        plt.title(f"Per-Class Dynamic Weight Trajectory ({method_name})")
        plt.legend(loc='best', fontsize=8); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{cfg.SAVE_DIR}/weight_trajectory_{method_name}.png", dpi=300); plt.close()
        print(f"  Saved weight trajectory plot for {method_name}")

    @staticmethod
    def plot_class_entropy(class_entropies_dict):
        """Bar chart comparing per-class entropy across methods (Experiment 5)."""
        fig, ax = plt.subplots(figsize=(14, 6))
        methods = list(class_entropies_dict.keys())
        x = np.arange(cfg.NUM_CLASSES)
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            ax.bar(x + i * width, class_entropies_dict[method], width, label=method, alpha=0.85)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(cfg.CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylabel("Mean Shannon Entropy")
        ax.set_title("Per-Class Prediction Entropy by Method")
        ax.legend(loc='upper right')
        ax.axhline(y=np.log(cfg.NUM_CLASSES), color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{cfg.SAVE_DIR}/per_class_entropy.png", dpi=300); plt.close()
        print("  Saved per-class entropy comparison plot")

    @staticmethod
    def plot_rl_lambda_schedule(lambda_history):
        """Plots the RL agent's chosen λ values over epochs (Experiment 4)."""
        if not lambda_history: return
        plt.figure(figsize=(12, 4))
        plt.step(range(1, len(lambda_history) + 1), lambda_history, where='mid',
                 linewidth=1.5, color='purple')
        plt.xlabel("Epoch"); plt.ylabel("$\\lambda$ (Calibration Penalty)")
        plt.title("RL Agent's Learned Penalty Schedule")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f"{cfg.SAVE_DIR}/rl_lambda_schedule.png", dpi=300); plt.close()
        print("  Saved RL lambda schedule plot")

# =============================================================================
# 9. EVALUATION (with Shannon Entropy — Experiment 5)
# =============================================================================
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

# =============================================================================
# 10. TRAINING & EVALUATION PIPELINE
# =============================================================================
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
            crit = DCALoss().to(cfg.DEVICE)
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

if __name__ == "__main__":
    for d_name in cfg.DATASETS:
        cfg.DATASET = d_name
        for b_name in cfg.BACKBONES:
            cfg.set_backbone(b_name)
            run_pipeline()

