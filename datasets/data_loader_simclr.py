# datasets/dataset_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import os

from datasets.augmentations import build_simclr_transform_from_yaml


class SimCLRDataset(Dataset):
    """Dataset wrapper to apply two random augmentations for each image."""
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, _ = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")

        # apply two different random augmentations
        xi, xj = self.transform(image)
        return (xi, xj), 0  # label 0 is dummy, SimCLR is self-supervised

    def __len__(self):
        return len(self.dataset)


def get_dataloader(cfg):
    """Build DataLoader for SimCLR training."""
    data_cfg = cfg["dataset"]
    root = data_cfg["root"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)

    # Load SimCLR augmentation pipeline
    transform = build_simclr_transform_from_yaml("./config/augmentation_config.yaml")

    dataset = SimCLRDataset(root=root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    print(f"[INFO] Loaded {len(dataset)} training samples from {root}")
    return datalo
