import yaml
import datetime, os
from pathlib import Path

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H")

def load_yaml(yaml_path:str, with_global=True):
    """Load YAML config file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if with_global:
        global_path = Path("config/global.yaml")
        if global_path.exists():
            with open(global_path, "r", encoding="utf-8") as f:
                global_cfg = yaml.safe_load(f) or {}
                cfg["global"] = global_cfg
        else:
            print("⚠️ global.yaml not found, skip global merge")

    return cfg

def get_root_dir(cfg): 
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        root_dir = cfg.globals.root["colab"]
    else: root_dir =  cfg.globals.root["local"]
    print(f"📁 Root dir set to: {root_dir}")
    return root_dir
    
# def get_root_dir(cfg): 
#     if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
#         root_dir = cfg["global"]["root"]["colab"]
#     else: root_dir = cfg["global"]["root"]["local"]
#     print(f"📁 Root dir set to: {root_dir}")
#     return root_dir

def get_dir_path(cfg, subCfg):
    root_dir = get_root_dir(cfg)

    output_dir = os.path.join(root_dir, subCfg['output_subdir'])
    output_path= os.path.join(root_dir, subCfg['output_subdir'], subCfg["output_file"])
    
    os.makedirs(output_dir , exist_ok=True)

    return output_dir, output_path
    