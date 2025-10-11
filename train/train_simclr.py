import torch
from datasets.augumentations import build_simclr_transform

def run(cfg):
    print("[INFO] Starting SimCLR training...")

    # 1. env setting
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    temperature = cfg["train"]["temperature"]
    num_workers = cfg["train"]["num_workers"]
    dataset_root = cfg["dataset"]["root"]
    encoder_type = cfg["model"]["base_encoder"]
    projection_dim = cfg["model"]["projection_dim"]

    # 2. data augumentation and dataloader
    transform = build_simclr_transform("./config/augmentation_config.yaml")

    # 3. model, loss , optimizer

    # 4. train loop

    # save encoder .pth

