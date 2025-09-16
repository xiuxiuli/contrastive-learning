import os
import torchvision
from utils import config_util as cfg_util

def download(config):
    timestamp = cfg_util.get_timestamp()
    setname= cfg["active_dataset"]
    ds_cfg = cfg["datasets"][setname]

    # destination
    des_path = cfg["path"]["root"]
    os.makedirs(des_path , exist_ok=True)

    # 
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
    elif setname == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=des_path,
            train=True,
            download=True
        )
        valset = torchvision.datasets.CIFAR10(
            root=des_path,
            train=False,
            download=True
        )
    elif setname == "imagenet100":
        # ⚠️ 
        print("[WARN] ImageNet-100 is not included in torchvision, it needs to be downloaded manually.")
        return
    
    print(f"[INFO] {setname} dataset downloaded to {des_path}")
    print(f"[INFO] Train size: {len(trainset)}, Val size: {len(valset)}")

if __name__ == "__main__":
    # load config
    cfg = cfg_util.load_yaml('./config/dataset_config.yaml')

    # download data
    download(cfg)