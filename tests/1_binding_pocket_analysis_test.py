"""
1_binding_pocket_analysis_test using a minimal dataset.

This test runs the full FEPA binding pocket analysis pipeline on a very small
subset of data: one ABFE replica (van1), two leg windows, and one apo replicate.

It mirrors ``1_bp_analysis_all_abfe_windows.py`` but overrides the lists passed
to ``load_abfe_paths_for_compound`` so that only test directories are accessed.
The configuration file points to small test datasets via the
``*_path_template`` fields.

When run in an environment with FEPA and dependencies (MDAnalysis, scikit-learn,
etc.), this test performs the complete workflow — featurization, PCA,
clustering, and plotting — and verifies that expected outputs are produced
and match reference CSVs within numeric tolerances.
"""

import warnings
from Bio import BiopythonDeprecationWarning

# Suppress irrelevant library warnings for clean test output
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import os
from pathlib import Path
import builtins
from typing import Literal

import pytest
import shutil
import numpy as np
import pandas as pd
import pandas.testing as pdt

# Default numeric tolerances for floating-point comparisons
DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8

# Ensure Literal works under older Python versions
builtins.Literal = Literal

# --- FEPA imports ---
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow

# --- Test utilities and fixtures ---
from fepa.tests.utils import (
    round_numeric,
    sort_by,
    align_pc_signs,
    first_existing,
    check_csv_equality,
)
from fepa.tests.conftest import test_env as test_env_fixture


@pytest.mark.integration
def test_minimal_bp_analysis(tmp_path, test_env):
    """Run the binding pocket analysis pipeline on a minimal dataset and validate outputs."""
    # --- Setup phase ---
    config = test_env["config"]
    root = test_env["root"]

    # Use the first compound from the test configuration
    cmp = config["compounds"][0]

    # Prepare a temporary working directory for analysis output
    analysis_output_dir = tmp_path / "a1_binding_pocket_analysis_apo_holo_reus" / "wdir"
    cmp_output_dir = analysis_output_dir / cmp
    os.makedirs(cmp_output_dir, exist_ok=True)

    # Only one replica and two Coulomb windows for minimal test runtime
    van_list = [1]
    leg_window_list = [f"coul.{i:02}" for i in range(2)]

    # Build the path dictionary for ABFE and apo trajectories
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp,
        van_list=van_list,
        leg_window_list=leg_window_list,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        apo=True,
    )

    # Keep only apo_1 for this test (drop extra apo replicates)
    for key in list(path_dict.keys()):
        if key.startswith("apo_") and key != "apo_1":
            del path_dict[key]

    # --- Execution phase ---
    # Instantiate workflow and run all major analysis steps
    ensemble_handler = EnsembleHandler(path_dict)
    workflow = binding_pocket_analysis_workflow(cmp, ensemble_handler)

    workflow.check_consistency()
    workflow.featurize(str(cmp_output_dir))
    workflow.reduce_dimensions(str(cmp_output_dir))
    workflow.cluster_pca(n_clusters=2, n_components=2)
    workflow.plot_pca_cluster(str(cmp_output_dir))
    workflow.load_results(str(cmp_output_dir))
    workflow.plot_holo_pca(str(cmp_output_dir))
    workflow.plot_apo_holo_pca(str(cmp_output_dir), apo_list=["apo_1"])
    workflow.plot_apo_abfe_pca(str(cmp_output_dir))

    # --- Validation phase ---
    # Directory containing reference expected results
    expected_dir = root / "tests" / "test_data" / "1_expected"
    assert expected_dir.exists(), f"Missing expected directory: {expected_dir}"

    # === 1. Validate feature CSV ===
    check_csv_equality(
        actual_path=cmp_output_dir / "SelfDistance_features.csv",
        expected_path=expected_dir / "SelfDistance_features.csv",
        label="Feature CSV",
    )

    # === 2. Validate PCA projection CSV ===
    check_csv_equality(
        actual_path=cmp_output_dir / "pca_projection_df.csv",
        expected_path=expected_dir / "pca_projection_df.csv",
        pc_prefix="PC",
        label="PCA projection CSV",
    )
