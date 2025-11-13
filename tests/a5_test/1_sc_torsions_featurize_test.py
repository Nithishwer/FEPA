"""
Test: Side-chain torsions featurization (minimal slice).
1) Runs featurization for van=1 and windows ['coul.00', 'coul.01'].
2) Asserts artifacts exist (CSV).
3) Compares SideChainTorsions_features.csv to a est reference.
"""

import warnings
from Bio import BiopythonDeprecationWarning
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version.splitlines()[0])
print("sys.path:")
for p in sys.path:
    print("  ", p)

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import pytest
import shutil

from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from ..tests.utils import check_csv_equality
import sys

RTOL = 1e-6
ATOL = 1e-8

BP_SELECTION_STRING = "name CA and resid 54 55 56 57"  # Doesnt matter what residues we pick here, as sidechain torsions dont use this selection


@pytest.mark.slow
@pytest.mark.integration
def test_sidechain_torsions_minimal_against_est(tmp_path, test_env):
    # --- Setup ---
    config = test_env["config"]
    repo_root = test_env["root"]
    cmp_name = config["compounds"][0]
    out_dir = tmp_path / "a4_compare_sidechain_torsions" / "wdir" / cmp_name
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
    featurizer = SideChainTorsionsFeaturizer(eh)
    featurizer.featurize(selection=BP_SELECTION_STRING)
    featurizer.save_features(str(out_dir), overwrite=True)

    # --- Artifact check ---
    act_csv = out_dir / "SideChainTorsions_features.csv"
    assert act_csv.exists(), f"Expected feature file not found: {act_csv}"

    # Copy actual CSV to a known location for easier debugging if needed
    shutil.copy(act_csv, repo_root / "tests" / "test_data")

    # --- Comparison ---
    expected_csv = (
        repo_root
        / "tests"
        / "test_data"
        / "5_expected"
        / cmp_name
        / "SideChainTorsions_features.csv"
    )
    assert expected_csv.exists(), f"Golden CSV not found: {expected_csv}"

    check_csv_equality(
        act_csv, expected_csv, label="Side-chain torsions CSV", rtol=RTOL, atol=ATOL
    )
