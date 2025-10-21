# datasets/augmentations.py
import random
from torchvision import transforms
from PIL import Image, ImageFilter
import torch
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import tool as cfg_util


class GaussianBlur(object):
    """Implements Gaussian blur as in SimCLR (Appendix A)."""
    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            radius = random.uniform(0.1, 2.0)
            return x.filter(ImageFilter.GaussianBlur(radius=radius))
        return x


class SimCLRAugment:
    """Configurable SimCLR augmentation pipeline."""

    def __init__(self, config):
        aug_cfg = config["augmentations"]
        size = aug_cfg["image_size"]

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=size,
                scale=tuple(aug_cfg["random_resized_crop"]["scale"]),
                ratio=tuple(aug_cfg["random_resized_crop"]["ratio"])
            ),
            transforms.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]["p"]),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=aug_cfg["color_jitter"]["brightness"],
                    contrast=aug_cfg["color_jitter"]["contrast"],
                    saturation=aug_cfg["color_jitter"]["saturation"],
                    hue=aug_cfg["color_jitter"]["hue"],
                )],
                p=aug_cfg["color_jitter"]["p"]
            ),
            transforms.RandomGrayscale(p=aug_cfg["random_grayscale"]["p"]),
            GaussianBlur(
                kernel_size=int(aug_cfg["gaussian_blur"]["kernel_scale"] * size),
                p=aug_cfg["gaussian_blur"]["p"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_cfg["normalize_mean"],
                std=aug_cfg["normalize_std"]
            ),
        ])

        # Optional random erase
        if aug_cfg.get("random_erase", {}).get("use", False):
            self.train_transform.transforms.append(
                transforms.RandomErasing(
                    p=aug_cfg["random_erase"]["p"],
                    scale=tuple(aug_cfg["random_erase"]["scale"]),
                    ratio=tuple(aug_cfg["random_erase"]["ratio"]),
                )
            )

    def __call__(self, x):
        """Return two different augmented views."""
        q = self.train_transform(x)
        k = self.train_transform(x)
        return q, k


def build_simclr_transform_from_yaml(yaml_path="./config/augmentation_config.yaml"):
    """Load YAML config and return a SimCLR transform."""
    cfg = cfg_util.load_yaml(yaml_path)
    return SimCLRAugment(cfg)
