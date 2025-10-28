# data/preprocess_texts.py
from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT) 

def preprocess_text(item):
    txt = item["text"].lower().strip()
    txt = txt.split(",")[0]                 # 取主类别名
    item["caption"] = f"a photo of a {txt}" # CLIP 风格
    return item


if __name__ == "__main__":
    cfg= OmegaConf.load("configs/data_config.yaml")
    cache_dir = Path(cfg.clean.cache_dir)
    save_dir = Path(cfg.clean.save_dir)

    print(f"📂 Loading dataset from: {cache_dir}")
    # 1️⃣ 加载数据（本地已有缓存则复用）
    dataset = load_dataset(cfg.clean.name, cache_dir="data/raw")

    # 对每个 split 执行清洗
    for split in dataset.keys():
        print(f"🧹 Processing split: {split}")
        dataset[split] = dataset[split].map(preprocess_text)

    os.makedirs(cfg.clean.save_dir, exist_ok=True)

    dataset.save_to_disk(save_dir)

    print(f"✅ Saved processed dataset to: {save_dir}")
