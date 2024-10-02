import yaml

def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config