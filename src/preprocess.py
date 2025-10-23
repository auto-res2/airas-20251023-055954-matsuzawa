from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

CACHE_DIR = Path(".cache/datasets").resolve()

# ---------------------------------------------------------------------------
# Transform builder
# ---------------------------------------------------------------------------

def build_transforms(pre_cfg: Dict, train: bool):
    t_list: list = []
    if train and "augmentation" in pre_cfg:
        aug = pre_cfg["augmentation"]
        if "random_crop" in aug:
            rc = aug["random_crop"]
            t_list.append(transforms.RandomCrop(rc["size"], padding=rc.get("padding", 0)))
        if "random_horizontal_flip" in aug:
            prob = aug["random_horizontal_flip"]
            t_list.append(transforms.RandomHorizontalFlip(prob))
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize(mean=pre_cfg["normalize_mean"], std=pre_cfg["normalize_std"]))
    return transforms.Compose(t_list)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(dataset_cfg: Dict, training_cfg: Dict, mode: str):
    name = dataset_cfg["name"].lower()
    if name != "cifar-10":
        raise ValueError("Only CIFAR-10 is implemented.")

    tf_train = build_transforms(dataset_cfg["preprocessing"], train=True)
    tf_eval = build_transforms(dataset_cfg["preprocessing"], train=False)

    full_train = datasets.CIFAR10(root=str(CACHE_DIR), train=True, download=True, transform=tf_train)

    val_ratio = dataset_cfg["splits"]["val"]
    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val])
    val_ds.dataset.transform = tf_eval

    if mode == "trial":
        subset_len = training_cfg["batch_size"] * 2
        train_ds = Subset(train_ds, list(range(subset_len)))
        val_ds = Subset(val_ds, list(range(subset_len)))

    batch_size = training_cfg["batch_size"]
    num_workers = 4 if torch.cuda.is_available() else 2

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }