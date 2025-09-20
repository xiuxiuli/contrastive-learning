import os
import torchvision
from utils import config_util as cfg_util
from datasets import load_dataset

def get_datasets(config):
    setname= config["active_dataset"]
    ds_cfg = config["datasets"][setname]

    # destination
    des_path = config["path"]["root"]
    os.makedirs(des_path , exist_ok=True)

    # plan change: SKIP STL-10
    if setname == "stl10":
        trainset = torchvision.datasets.STL10(
            root=des_path,
            split= ds_cfg["train_split"], 
            download=True
        )
        valset = torchvision.datasets.STL10(
            root=des_path,
            split=ds_cfg["val_split"],
            download=True
        ) 

    elif setname == "cline9":
        # hugging face Clane9/Imagenet100
        dataset = load_dataset("clane9/imagenet-100", cache_dir=des_path)

        trainset = sample_dataset(dataset[ds_cfg["train_split"]],
                                  ds_cfg.get("train_size"),
                                  cfg["sampling"]["seed"])
        valset = sample_dataset(dataset[ds_cfg["val_split"]],
                                ds_cfg.get("val_size"),
                                cfg["sampling"]["seed"])
        
    print(f"[INFO] {setname} dataset ready at {des_path}")
    print(f"[INFO] Train size: {len(trainset)}, Val size: {len(valset)}")

    return trainset, valset

# Hugging Face, select a subset
def sample_dataset(dataset, size, seed=42):
    if size is None or size > len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(size))

if __name__ == "__main__":
    # load config
    cfg = cfg_util.load_yaml('./config/dataset_config.yaml')

    # download data
    get_datasets(cfg)