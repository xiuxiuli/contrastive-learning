import io
import torchvision.transforms as T
import torch

from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image

import pytorch_lightning as pl
from pathlib import Path

# datamodule_dino.py

"""
datamodule_dino.py
------------------
DataModule for DINOv2 / SimCLR-style self-supervised pretraining
on ImageNet-100. Provides multi-crop augmentation for training and
center-crop normalization for evaluation.

Usage:
    from data.datamodule_dino import ImageNet100DinoDataModule
    dm = ImageNet100DinoDataModule(cfg.data, cfg.train)
"""

# ---------------------------
# Utils: multi-crop transforms - augmentations
# ---------------------------
class MultiCropTransform:
    """
    DINO-style multi-crop:
      - 2 global crops @224 - 整体裁剪
      - N local crops @96 - 局部裁剪
    """
    def __init__(
        self,
        global_size=224,
        local_size=96,
        n_global=2,
        n_local=6,
        color_jitter=0.8,
        gray_p=0.2,
        blur_p_global=1.0,
        blur_p_local=0.5 
    ):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # global views
        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=(int(0.1 * global_size) // 2 * 2 + 1)),  #ernel size value should be an odd and positive number.
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # local views
        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=(int(0.1 * global_size) // 2 * 2 + 1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.n_global, self.n_local = n_global, n_local
    
    def __call__(self, img):
        crops = [self.global_transform(img) for _ in range(self.n_global)]
        crops += [self.local_transform(img) for  _ in range(self.n_local)]
        return crops

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # HuggingFace dataset 存的是字节流
        img_data = item["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        else:
            image = img_data.convert("RGB")

        if self.transform:
            image = self.transform(image)
        label=item.get("label", -1)
        return image, label
    

# ---------------------------
# DataModule
# ---------------------------
class ImageNet100DinoDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, train_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        self.mc = MultiCropTransform(
            global_size=data_cfg.get("image_size", 224),
            local_size=96,
            n_global=2,
            n_local=6
        )

        # eval transform (no heavy augmentation)
        self.eval_tr = T.Compose([
            T.Resize(int(1.14 * data_cfg.get("image_size", 224))),
            T.CenterCrop(data_cfg.get("image_size", 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def setup(self, stage=None):
        root=self.data_cfg.get("cache_dir")
        dataset = load_from_disk(root)
        # base = Path(self.data_cfg.get("cache_dir"))  # e.g. data/processed
        # 情况 A：一次性保存的 DatasetDict（根目录下有 dataset_dict.json）

        self.trainset = HFDatasetWrapper(dataset["train"], transform=self.mc)
        self.valset = HFDatasetWrapper(dataset["validation"], transform=self.eval_tr)

    # custom collate for multi-crop
    def collate_mc(self, batch): # group each crop type
        views = list(zip(*[b[0] for b in batch]))
        views = [torch.stack(v, dim=0) for v in views]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return views, labels
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.train_cfg.get("batch_size", 256),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_mc
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.train_cfg.get("batch_size", 256),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

