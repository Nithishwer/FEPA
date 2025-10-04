"""
1_binding_pocket_analysis_test using a minimal dataset.

This test exercises the full FEPA pipeline on a very small subset of data: a
single ABFE replica (van1), a pair of leg windows and a single apo replicate.
It mirrors the logic of ``1_bp_analysis_all_abfe_windows.py`` but overrides the
``van_list`` and ``leg_window_list`` arguments passed to
``load_abfe_paths_for_compound`` so that only the test directories are
accessed.  The configuration used by the test points to the test data
via the ``*_path_template`` fields.


When run in an environment where FEPA and its dependencies (MDAnalysis,
scikit-learn, etc.) are installed, this test will perform the full
featurization, PCA, clustering and plotting steps and assert that the key
output artifacts were written to disk.
"""

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")



import os
from pathlib import Path
import builtins
from typing import Literal

import pytest

import numpy as np                      
import pandas as pd                     
import pandas.testing as pdt      

DECIMALS = 6                            
RTOL = 1e-6                             
ATOL = 1e-8                             


builtins.Literal = Literal

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow


# === small helpers for deterministic comparisons
def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].round(DECIMALS)
    return df

def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {paths!r}")

def _sort_by(df: pd.DataFrame, keys=("ensemble", "timestep", "frame")) -> pd.DataFrame:
    use = [k for k in keys if k in df.columns]
    return df.sort_values(use).reset_index(drop=True) if use else df

def _align_pc_signs(actual: pd.DataFrame, expected: pd.DataFrame, pc_prefix: str = "PC") -> pd.DataFrame:
    """Return a copy of expected with sign flipped per PC if needed to match actual."""
    exp = expected.copy()
    pc_cols = [c for c in exp.columns if c.startswith(pc_prefix) and c in actual.columns]
    for c in pc_cols:
        a = actual[c].to_numpy()
        b = exp[c].to_numpy()
        # if dot < 0, flip sign of expected to align direction with actual
        if np.dot(a, b) < 0:
            exp[c] = -exp[c]
    return exp

@pytest.mark.integration
def test_minimal_bp_analysis(tmp_path):
    """Run the binding pocket analysis on a minimal test dataset."""
    # Locate the test configuration file.
    test_root = Path(__file__).resolve().parents[1]
    config_path = test_root / "tests" / "test_config" / "config.json"
    assert config_path.exists(), f"Missing configuration file: {config_path}"
    config = load_config(str(config_path))

    # Use the first compound listed in the config.
    cmp = config["compounds"][0]

    # Create an output directory in tmp_path.
    analysis_output_dir = tmp_path / "a1_binding_pocket_analysis_apo_holo_reus" / "wdir"
    cmp_output_dir = analysis_output_dir / cmp
    os.makedirs(cmp_output_dir, exist_ok=True)

    # Only one replica and two Coulomb windows.
    van_list = [1]
    leg_window_list = [f"coul.{i:02}" for i in range(2)]

    # Build the path dictionary, including apo=True.
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp,
        van_list=van_list,
        leg_window_list=leg_window_list,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        apo=True,
    )

    # Remove apo_2 and apo_3 if present.
    for key in list(path_dict.keys()):
        if key.startswith("apo_") and key != "apo_1":
            del path_dict[key]

    # Run the analysis pipeline.
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

    # Verify expected outputs exist.
    assert (cmp_output_dir / "pca_projection_df.csv").exists(), "Missing PCA projection file"
    png_files = list(cmp_output_dir.glob("*.png"))
    assert png_files, "No plots were generated by the analysis"

    # compare against CSVs in tests/test_data/1_expected/
    expected_dir = test_root / "tests" / "test_data" / "1_expected"
    assert expected_dir.exists(), f"Missing expected directory: {expected_dir}"

    # 1) SelfDistance_features.csv (or fallback name)
    exp_feat_path = _first_existing(
        expected_dir / "SelfDistance_features.csv"
    )
    act_feat_path = _first_existing(
        cmp_output_dir / "SelfDistance_features.csv"
    )

    act = _round_numeric(pd.read_csv(act_feat_path))
    exp = _round_numeric(pd.read_csv(exp_feat_path))

    if "ensemble" in act.columns and "ensemble" in exp.columns:
        keep = set(act["ensemble"].unique())
        exp = exp[exp["ensemble"].isin(keep)].copy()

    act = _sort_by(act)
    exp = _sort_by(exp)

    # strict schema expectation (your point about not over-engineering)
    assert list(act.columns) == list(exp.columns), "Feature CSV columns differ"
    assert len(act) == len(exp), "Feature CSV row counts differ"

    pdt.assert_frame_equal(
        act.reset_index(drop=True),
        exp.reset_index(drop=True),
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )

    # 2) pca_projection_df.csv (compare whole frame; numeric columns tol)
    act = _round_numeric(pd.read_csv(cmp_output_dir / "pca_projection_df.csv"))
    exp = _round_numeric(pd.read_csv(expected_dir / "pca_projection_df.csv"))

    if "ensemble" in act.columns and "ensemble" in exp.columns:
        keep = set(act["ensemble"].unique())
        exp = exp[exp["ensemble"].isin(keep)].copy()

    act = _sort_by(act)
    exp = _sort_by(exp)

    assert list(act.columns) == list(exp.columns), "PCA projection CSV columns differ"
    assert len(act) == len(exp), f"PCA projection CSV row counts differ (act={len(act)} exp={len(exp)})"

    exp = _align_pc_signs(act, exp, pc_prefix="PC")

    pdt.assert_frame_equal(
        act.reset_index(drop=True),
        exp.reset_index(drop=True),
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )
