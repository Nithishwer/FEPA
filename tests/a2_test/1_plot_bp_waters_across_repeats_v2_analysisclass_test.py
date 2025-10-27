"""
Test: BP water featurization (minimal slice).
1) Runs featurization for van=1 and windows ['coul.00', 'coul.01'].
2) Asserts artifacts exist (CSV).
3) Compares WaterOccupancy_features.csv to a golden reference.
"""

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

from pathlib import Path
import pandas as pd
import pandas.testing as pdt
import pytest

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.featurizers import BPWaterFeaturizer
from fepa.tests.utils import round_numeric, align_on_common_keys, check_csv_equality
from fepa.tests.conftest import test_env

DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8

BP_SELECTION_STRING = (
    "name CA and resid "
    "54 55 56 57 58 59 60 61 62 64 65 68 83 84 85 87 88 91 92 "
    "173 176 177 180 217 218 221 225 235 238 239 240 241 242 243 244 245 246 247"
)

@pytest.mark.slow
@pytest.mark.integration
def test_bp_waters_minimal_against_golden(tmp_path, test_env):
    # --- Setup ---
    config = test_env["config"]
    repo_root = test_env["root"]
    cmp_name = config["compounds"][0]
    out_dir = tmp_path / "a3_compare_bp_waters" / "wdir" / cmp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Featurization ---
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp_name,
        van_list=[1],
        leg_window_list=[f"coul.{i:02d}" for i in range(2)],
        bp_selection_string=BP_SELECTION_STRING,
        apo=False,
    )
    eh = EnsembleHandler(path_dict)
    eh.make_universes()
    featurizer = BPWaterFeaturizer(eh)
    featurizer.featurize(radius=10)
    featurizer.save_features(str(out_dir), overwrite=True)

    # --- Artifact check ---
    act_csv = out_dir / "WaterOccupancy_features.csv"
    assert act_csv.exists(), f"Expected feature file not found: {act_csv}"

    # --- Comparison ---
    golden_csv = (
        repo_root / "tests" / "test_data" / "2_expected" / cmp_name / "WaterOccupancy_features.csv"
    )
    assert golden_csv.exists(), f"Golden CSV not found: {golden_csv}"

    check_csv_equality(act_csv, golden_csv, label="Water occupancy CSV", rtol=RTOL, atol=ATOL)
