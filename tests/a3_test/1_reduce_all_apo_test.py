"""
Test for apo binding pocket analysis with PCA dimensionality reduction.

- Loads trajectories, featurizes, performs PCA, and creates visualizations in tmp_path
- Validates core FEPA functionality: trajectory loading, featurization, PCA, visualization
- Compares produced outputs against reference data in 3_expected
"""

from __future__ import annotations

import sys as _sys
import os as _os
import warnings as _warnings

_sys.dont_write_bytecode = True
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_os.environ["MPLBACKEND"] = "Agg"

_warnings.filterwarnings("ignore")
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", module="MDAnalysis")
_warnings.filterwarnings("ignore", module="Bio")
# ---------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pandas.testing as pdt
import pytest

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import check_bp_residue_consistency
from fepa.utils.path_utils import load_paths_for_apo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ================================ Path Helpers ================================
def _detect_repo_root() -> Path:
    """Compute the repo root relative to this test file."""
    env_root = _os.getenv("FEPA_REPO_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.is_dir():
            return p
    here = Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "tests").is_dir():
            return anc
    return here.parents[2]


def _abspath_templates(cfg: dict, repo_root: Path) -> dict:
    """Convert relative path templates in config to absolute paths."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(cfg)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out


# ================================ Fixtures ================================
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Session-scoped fixture providing repository root path."""
    root = _detect_repo_root()
    logger.info("Using FEPA repo root: %s", root)
    return root


@pytest.fixture(scope="session")
def config_and_inputs(repo_root: Path) -> Tuple[dict, Path, Path]:
    """Load FEPA configuration and locate input/expected data directories."""
    cfg_path = (repo_root / "tests" / "test_config" / "config.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    cfg = _abspath_templates(cfg, repo_root)

    expected_root = (repo_root / "tests" / "test_data" / "3_expected").resolve()
    if not expected_root.exists():
        raise FileNotFoundError(f"Expected data root not found: {expected_root}")

    input_features_dir = (repo_root / "tests" / "test_data" / "1_expected").resolve()
    if not input_features_dir.exists():
        raise FileNotFoundError(
            f"Input features directory not found: {input_features_dir}"
        )

    return cfg, expected_root, input_features_dir


@pytest.fixture(scope="function")
def run_apo_analysis(
    tmp_path: Path, config_and_inputs: Tuple[dict, Path, Path]
) -> Dict[str, Path]:
    """
    Run the full apo binding pocket analysis pipeline.

    Returns dict with produced file paths and reference paths.
    """
    config, expected_root, input_features_dir = config_and_inputs
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)

    cmp = "1"  # Compound to filter out

    # Prepare paths and load trajectories
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )
    ensemble_handler = EnsembleHandler(path_dict)
    ensemble_handler.make_universes()
    check_bp_residue_consistency(ensemble_handler.get_universe_dict())

    # Load and filter features
    featurizer = SelfDistanceFeaturizer(ensemble_handler)
    featurizer.load_features(input_dir=str(input_features_dir))
    featurizer.feature_df = featurizer.feature_df[
        featurizer.feature_df["ensemble"].str.contains(f"{cmp}") == False
    ]

    # PCA
    dimreducer = PCADimReducer(
        featurizer.get_feature_df(), n_components=8, random_state=42
    )
    dimreducer.reduce_dimensions()
    dimreducer.calculate_projections()

    projection_csv = out_dir / "pca_projection_df.csv"
    dimreducer.save_projection_df(save_path=str(projection_csv))
    projection_df = dimreducer.get_pca_projection_df()

    # Entropy heatmaps
    try:
        plot_entropy_heatmaps(
            cmp="apo",
            entropy_metric="jsd",
            columns_to_consider=["PC1", "PC2"],
            projection_df=projection_df,
            output_dir=str(out_dir),
        )
    except Exception as e:
        logger.warning("Entropy heatmap failed (may be expected): %s", e)

    # Visualizations
    dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")

    pca_ensemble_plot = out_dir / "pca_components_ensemble.png"
    dimred_visualizer.plot_dimred_sims(save_path=str(pca_ensemble_plot))

    pca_time_plot = out_dir / "pca_components_time.png"
    dimred_visualizer.plot_dimred_time(save_path=str(pca_time_plot))

    eigenvalues_plot = out_dir / "eigenvalues.png"
    plot_eigenvalues(
        pca_object=dimreducer.get_pca(),
        n_components=8,
        save_path=str(eigenvalues_plot),
    )

    pca_components_plot = out_dir / "pca_components.png"
    plot_pca_components(
        pca_object=dimreducer.get_pca(),
        feature_df=featurizer.get_feature_df(),
        num=8,
        save_path=str(pca_components_plot),
    )

    logger.info("Apo analysis workflow complete")

    return {
        "projection_csv": projection_csv,
        "ref_projection_csv": expected_root / "pca_projection_df.csv",
        "plots": {
            "ensemble": pca_ensemble_plot,
            "time": pca_time_plot,
            "eigenvalues": eigenvalues_plot,
            "components": pca_components_plot,
        },
    }


# ================================ Test Functions ================================


def test_apo_workflow_completes_and_creates_files(
    run_apo_analysis: Dict[str, Path],
) -> None:
    """
    Test that the apo analysis workflow completes and creates expected files.

    Validates:
      - PCA projection CSV exists and is non-empty
      - All 4 visualization plots are created
    """
    projection_csv = run_apo_analysis["projection_csv"]
    plots = run_apo_analysis["plots"]

    # CSV checks
    assert projection_csv.exists(), f"Projection CSV not found: {projection_csv}"
    assert projection_csv.stat().st_size > 0, "Projection CSV is empty"

    # Plot checks
    for plot_name, plot_path in plots.items():
        assert plot_path.exists(), f"Plot '{plot_name}' not found: {plot_path}"
        assert plot_path.stat().st_size > 0, f"Plot '{plot_name}' is empty"

    logger.info("All expected output files created (%d plots + CSV)", len(plots))


def test_apo_analysis_scientific_validity(run_apo_analysis: Dict[str, Path]) -> None:
    """
    Test that PCA analysis produces scientifically valid results.

    Validates:
      - CSV has correct structure (PC1-PC8, ensemble, timestep)
      - No NaN values in PCs
      - PCs have variance (not constant)
      - PC1 has highest variance
      - Variance decreases across components (general trend)
    """
    projection_csv = run_apo_analysis["projection_csv"]
    df = pd.read_csv(projection_csv)

    # Structure checks
    assert len(df) > 0, "Projection should have rows"
    assert "ensemble" in df.columns, "Missing 'ensemble' column"
    assert "timestep" in df.columns, "Missing 'timestep' column"

    # Check all PC columns exist
    expected_pc_cols = [f"PC{i + 1}" for i in range(8)]
    for col in expected_pc_cols:
        assert col in df.columns, f"Missing column: {col}"
        assert not df[col].isnull().any(), f"{col} contains NaN values"
        assert df[col].std() > 0, f"{col} has no variance"

    # Validate PCA properties
    variances = [df[col].var() for col in expected_pc_cols]

    assert variances[0] == max(variances), "PC1 should have highest variance"

    # Early PCs should have more variance than late PCs
    early_var = sum(variances[:3])
    late_var = sum(variances[5:])
    assert early_var > late_var, "Early PCs should dominate variance"

    logger.info("PCA results are scientifically valid")


def test_apo_analysis_matches_reference_data(run_apo_analysis: Dict[str, Path]) -> None:
    """
    Test that produced outputs match expected reference data.

    Uses correlation-based comparison to handle PCA sign ambiguity.
    Validates:
      - Shapes match
      - Metadata (ensemble, timestep) matches exactly
      - PC values have high correlation (>0.99) with reference
    """
    projection_csv = run_apo_analysis["projection_csv"]
    ref_projection_csv = run_apo_analysis["ref_projection_csv"]

    if not ref_projection_csv.exists():
        pytest.skip(f"Reference CSV not found: {ref_projection_csv}")

    # Load and align
    df_prod = pd.read_csv(projection_csv)
    df_ref = pd.read_csv(ref_projection_csv)

    # Keep common columns and sort
    common_cols = [c for c in df_ref.columns if c in df_prod.columns]
    df_prod = df_prod[common_cols].copy()
    df_ref = df_ref[common_cols].copy()

    sort_keys = [c for c in ["ensemble", "timestep"] if c in common_cols]
    if sort_keys:
        df_prod = df_prod.sort_values(by=sort_keys).reset_index(drop=True)
        df_ref = df_ref.sort_values(by=sort_keys).reset_index(drop=True)

    # Check shapes
    assert df_prod.shape == df_ref.shape, (
        f"Shape mismatch: produced {df_prod.shape} vs expected {df_ref.shape}"
    )

    # Metadata should match exactly
    pdt.assert_series_equal(df_prod["ensemble"], df_ref["ensemble"], check_names=False)
    pdt.assert_series_equal(df_prod["timestep"], df_ref["timestep"], check_names=False)

    # PC values: check correlation (handles sign ambiguity)
    pc_cols = [c for c in df_prod.columns if c.startswith("PC")]
    for col in pc_cols:
        corr = df_prod[col].corr(df_ref[col])
        assert abs(corr) > 0.99, (
            f"{col} correlation too low: {corr:.4f} (expected >0.99)"
        )

    logger.info("Outputs match reference data (correlation-based comparison)")
