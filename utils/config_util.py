import yaml
import datetime

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H")

def load_yaml(yaml_path):
    """Load YAML config file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    

