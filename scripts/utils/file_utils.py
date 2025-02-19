import os
import json

def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Parameters:
    -----------
    config_path : str
        Path to the JSON configuration file.

    Returns:
    --------
    dict
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

