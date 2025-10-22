"""
main.py — Unified Hydra + Lightning + MLflow Pipeline
Author: Xiuxiu Li
------------------------------------------------------
Controls multi-stage workflow:
DINOv2 → Linear Probe → CLIP → Demo
"""

import torch
import os
from utils import tool as util
from train import train_simclr, train_clip 

def setup_environment(cfg):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    os.makedirs(f"./checkpoints", exist_ok=True)

    method = cfg.get("method", "simclr")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Method: {method}")
    print("=" * 40)


def main():
    # read yaml
    cfg = util.load_yaml("./config/train_config.yaml")
    method = cfg.get("method", "simclr")

    # set env
    setup_environment(cfg)

    # training
    if method == "simclr":
        train_simclr.run(cfg)
    elif method == "clip":
        train_clip.run(cfg)
    else:
        raise NotImplementedError(f"Training method '{method}' not implemented.")
    
    print("\n✅ Training completed successfully.")

if __name__ == "__main__":
    main()