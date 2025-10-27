from pathlib import Path
import pytest
from fepa.utils.file_utils import load_config


@pytest.fixture
def test_env():
    """Provide the test environment with project root and loaded test configuration."""
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "tests" / "test_config" / "config.json")
    return {"root": root, "config": config}
