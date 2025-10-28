import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT) 

def download(cfg):
    
    data_cfg = cfg.dataset
    data_name = data_cfg.name

    # destination
    # root_dir = tool.get_root_dir(cfg)
    # output_dir = os.path.join(root_dir, data_cfg["output_subdir"])
    output_dir = Path(data_cfg.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset(data_name, cache_dir=output_dir)

    splits = data_cfg.splits
    sizes = data_cfg.sizes
    seed =  cfg.sampling.seed

    trainset = sample_dataset(data[splits.train], sizes.train, seed)
    valset = sample_dataset(data[splits.val], sizes.val, seed)
    testset = sample_dataset(data[splits.test], sizes.test, seed)
   
    print(f"[INFO] {data_name} downloaded and cached to {output_dir}")
    print(f"[INFO] Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    return trainset, valset, testset

# Hugging Face, select a subset
def sample_dataset(dataset, size, seed=42):
    if size is None or size > len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(size))

if __name__ == "__main__":
    # load config
    cfg = OmegaConf.load('./configs/data_config.yaml')
    # cfg = tool.load_yaml('./config/data_config.yaml', True)

    # download data
    download(cfg)