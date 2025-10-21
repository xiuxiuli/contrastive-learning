import os
import torchvision
from utils import tool
from datasets import load_dataset

def get_datasets(cfg):
    
    subCfg = cfg["dataset"]
    setname = subCfg["name"]

    # destination
    output_dir, output_path = tool.get_dir_path(cfg, subCfg)

    dataset = load_dataset(setname , cache_dir=output_dir)

    trainset = sample_dataset(dataset[subCfg["train_split"]],
                                subCfg.get("train_size"),
                                cfg["sampling"]["seed"])
    
    valset = sample_dataset(dataset[subCfg["val_split"]],
                            subCfg.get("val_size"),
                            cfg["sampling"]["seed"])
        
    print(f"[INFO] {setname} dataset ready at {output_dir}")
    print(f"[INFO] Train size: {len(trainset)}, Val size: {len(valset)}")

    return trainset, valset

# Hugging Face, select a subset
def sample_dataset(dataset, size, seed=42):
    if size is None or size > len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(size))

if __name__ == "__main__":
    # load config
    cfg = tool.load_yaml('./config/data_config.yaml')

    # download data
    get_datasets(cfg)