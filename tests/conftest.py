from pathlib import Path
import pytest
# Assuming this function loads the JSON file into a Python dictionary
from fepa.utils.file_utils import load_config 

def make_paths_absolute(config: dict, root: Path) -> dict:
    """
    Converts relative path templates in the config to absolute paths 
    based on the project root.
    """
    # Keys in your config that contain relative paths
    path_keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template"
    ]
    
    # Iterate and update the paths
    for key in path_keys:
        if key in config:
            # Join the project root with the relative path from the config.
            # We use Path() for safe, cross-platform path joining.
            relative_path = Path(config[key])
            config[key] = str(root / relative_path)
            
    return config


@pytest.fixture
def test_env():
    """Provide the test environment with project root and loaded test configuration."""
    
    # 1. Determine the project root (assuming tests/ is one level down)
    # Path(__file__).resolve().parents[1] correctly points to the directory 
    # above 'tests', which is your project root.
    root = Path(__file__).resolve().parents[1]
    
    # 2. Load the configuration file
    config_path = root / "tests" / "test_config" / "config.json"
    config = load_config(config_path)

    # 3. Convert relative paths inside the loaded config to absolute paths
    config = make_paths_absolute(config, root)
    
    return {"root": root, "config": config}