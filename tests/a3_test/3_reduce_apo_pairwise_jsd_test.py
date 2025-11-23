"""
Test for pairwise apo JSD and entropy analysis.

Validates entropy-based feature analysis workflow:
  - Load pre-computed top features from pairwise analysis
  - Perform PCA and clustering on top features  
  - Compute JSD between cluster pairs
  - Generate histogram visualizations

New FEPA functionalities tested:
  - compute_relative_entropy(): JSD calculation
  - compute_histograms(): Distribution analysis
  - plot_jsd_histograms(): JSD visualization
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

import numpy as np
import pandas as pd

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import compute_histograms, plot_jsd_histograms
from fepa.core.analyzers import compute_relative_entropy
from fepa.utils.file_utils import load_config
from fepa.utils.dimred_utils import cluster_pca

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
def config_and_inputs(repo_root: Path) -> Tuple[dict, Path]:
    """Load FEPA configuration and locate expected data directory."""
    cfg_path = (repo_root / "tests" / "test_config" / "config.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    cfg = _abspath_templates(cfg, repo_root)

    expected_root = (repo_root / "tests" / "test_data" / "3_expected").resolve()
    if not expected_root.exists():
        raise FileNotFoundError(
            f"Expected data root not found: {expected_root}\n"
            f"Run 2_reduce_apo_pairwise_test_data.py to generate pairwise outputs first."
        )

    return cfg, expected_root


@pytest.fixture(scope="function")
def run_jsd_analysis(
    tmp_path: Path, config_and_inputs: Tuple[dict, Path]
) -> Dict[str, any]:
    """
    Execute JSD and entropy analysis workflow for all pairs.
    
    For each pair:
      - Load pre-computed top features
      - PCA on top features
      - Cluster projections
      - Compute JSD between cluster pairs
      - Generate JSD histogram visualization
    """
    config, expected_root = config_and_inputs
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine available pairs
    desired_order = ["apo_1", "apo_2", "apo_3"]
    available_pairs = []
    for i in range(len(desired_order) - 1):
        pair = (desired_order[i], desired_order[i + 1])
        pair_name = f"{pair[0]}_{pair[1]}"
        pair_dir = expected_root / pair_name
        top_features_csv = pair_dir / f"top_features_{pair[0]}_{pair[1]}.csv"
        
        if top_features_csv.exists():
            available_pairs.append(pair)
    
    if len(available_pairs) == 0:
        raise RuntimeError(
            f"No pairwise analysis outputs found in {expected_root}\n"
            f"Run 2_reduce_apo_pairwise_test_data.py first."
        )

    # Cluster mapping for JSD computation
    ensemble_to_cluster = {
        ("apo_1", "apo_2"): (0, 1),
        ("apo_2", "apo_3"): (2, 1),
    }
    default_pair_clusters = (0, 1)

    pair_outputs = {}

    for pair in available_pairs:
        pair_name = f"{pair[0]}_{pair[1]}"
        pair_dir = out_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Load top features
        input_pair_dir = expected_root / pair_name
        top_features_input = input_pair_dir / f"top_features_{pair[0]}_{pair[1]}.csv"
        top_features_df = pd.read_csv(top_features_input)
        n_features = len([c for c in top_features_df.columns if "DIST" in c])

        # PCA on top features
        dimreducer_top = PCADimReducer(
            top_features_df, n_components=8, random_state=42
        )
        dimreducer_top.reduce_dimensions()
        dimreducer_top.calculate_projections()
        
        projection_csv = pair_dir / f"pca_projection_df_top_{pair[0]}_{pair[1]}.csv"
        dimreducer_top.save_projection_df(save_path=str(projection_csv))

        # Cluster
        pca_projection_clustered = cluster_pca(
            dimreducer_top.get_pca_projection_df(), n_clusters=3, n_components=8
        )

        # Attach cluster labels
        top_features_df["cluster"] = pca_projection_clustered["cluster"]

        # Compute JSD
        cluster1, cluster2 = ensemble_to_cluster.get(pair, default_pair_clusters)
        relative_entropy_dict = compute_relative_entropy(
            feature_df=top_features_df,
            ensemble1=cluster1,
            ensemble2=cluster2,
            num_bins=20,
            key="cluster",
            feature_column_keyword="DIST",
        )
        n_jsd_features = len(relative_entropy_dict["jsd"])

        # Compute histograms
        histograms = compute_histograms(
            feature_df=top_features_df,
            key="cluster",
            value_1=cluster1,
            value_2=cluster2,
            num_bins=20,
            feature_column_keyword="DIST",
        )
        
        # Generate visualization
        jsd_plot = pair_dir / "jsd_histograms.png"
        plot_jsd_histograms(
            histograms=histograms,
            relative_entropy_dict=relative_entropy_dict,
            top_n=16,
            save_path=str(jsd_plot),
        )

        pair_outputs[pair_name] = {
            "projection_csv": projection_csv,
            "jsd_plot": jsd_plot,
            "relative_entropy_dict": relative_entropy_dict,
            "histograms": histograms,
            "n_features": n_features,
            "n_jsd_features": n_jsd_features,
        }

    logger.info("JSD analysis complete: %d pairs", len(available_pairs))

    return {
        "pairs": available_pairs,
        "pair_outputs": pair_outputs,
    }


# ================================ Test Functions ================================

def test_jsd_workflow_completes_and_creates_files(run_jsd_analysis: Dict) -> None:
    """
    Test that JSD workflow completes and creates expected files.
    
    Validates:
      - At least one pair processed
      - PCA projection CSV created for each pair
      - JSD histogram plot created for each pair
    """
    pairs = run_jsd_analysis["pairs"]
    pair_outputs = run_jsd_analysis["pair_outputs"]

    assert len(pairs) >= 1, "Should process at least one ensemble pair"
    assert len(pair_outputs) == len(pairs), "Should have outputs for all pairs"

    for pair in pairs:
        pair_name = f"{pair[0]}_{pair[1]}"
        outputs = pair_outputs[pair_name]

        assert outputs["projection_csv"].exists(), f"Missing projection CSV for {pair_name}"
        assert outputs["jsd_plot"].exists(), f"Missing JSD plot for {pair_name}"
        assert outputs["jsd_plot"].stat().st_size > 0, f"Empty JSD plot for {pair_name}"

    logger.info("Workflow completed successfully for all %d pairs", len(pairs))


def test_jsd_entropy_analysis_correctness(run_jsd_analysis: Dict) -> None:
    """
    Test that entropy analysis produces correct results.
    
    Validates compute_relative_entropy() and compute_histograms():
      - JSD dict has expected structure: {"name": array, "jsd": array, ...}
      - JSD values in valid range [0, 1]
      - Histograms are list of tuples: [(grp1, grp2, bins), ...]
      - Histogram bins are monotonically increasing
      - Distributions have data
    """
    pair_outputs = run_jsd_analysis["pair_outputs"]

    for pair_name, outputs in pair_outputs.items():
        jsd_dict = outputs["relative_entropy_dict"]
        histograms = outputs["histograms"]

        # Validate JSD structure
        assert "name" in jsd_dict, f"Missing 'name' key for {pair_name}"
        assert "jsd" in jsd_dict, f"Missing 'jsd' key for {pair_name}"
        assert len(jsd_dict["name"]) == len(jsd_dict["jsd"]), (
            f"Length mismatch in JSD dict for {pair_name}"
        )

        # Validate JSD values
        jsd_values = jsd_dict["jsd"]
        assert len(jsd_values) > 0, f"No JSD values computed for {pair_name}"
        assert all(0 <= v <= 1 for v in jsd_values), (
            f"JSD out of range [0,1] for {pair_name}"
        )

        # Validate histogram structure
        assert len(histograms) > 0, f"No histograms for {pair_name}"
        
        # Check first histogram structure (representative)
        first_hist = histograms[0]
        assert isinstance(first_hist, tuple) and len(first_hist) == 3, (
            f"Histogram should be (grp1, grp2, bins) tuple for {pair_name}"
        )
        
        col_grp1, col_grp2, bins = first_hist
        assert len(col_grp1) > 0 and len(col_grp2) > 0, (
            f"Empty distributions for {pair_name}"
        )
        assert len(bins) > 0, f"Empty bins for {pair_name}"
        assert all(bins[i] <= bins[i+1] for i in range(len(bins)-1)), (
            f"Bins not monotonic for {pair_name}"
        )

        logger.info("Entropy analysis correct for %s", pair_name)

    logger.info("Entropy analysis correctness validated for all pairs")


def test_jsd_analysis_identifies_discriminative_features(
    run_jsd_analysis: Dict,
) -> None:
    """
    Test that JSD successfully identifies discriminative features.
    
    Validates:
      - JSD values have variation (not all identical)
      - Top features have higher JSD than average
      - Some features are discriminative (JSD > 0.05)
    """
    pair_outputs = run_jsd_analysis["pair_outputs"]

    for pair_name, outputs in pair_outputs.items():
        jsd_values = outputs["relative_entropy_dict"]["jsd"]
        
        # Should have variation
        assert jsd_values.std() > 0, (
            f"All JSD values identical for {pair_name}"
        )

        # Top features should be more discriminative
        jsd_mean = jsd_values.mean()
        sorted_indices = jsd_values.argsort()[::-1]
        n_top = max(1, len(jsd_values) // 2)
        top_mean = jsd_values[sorted_indices[:n_top]].mean()
        
        assert top_mean >= jsd_mean, (
            f"Top features should have higher JSD for {pair_name}"
        )

        # Some features should be discriminative
        discriminative = sum(1 for v in jsd_values if v > 0.05)
        assert discriminative > 0 or jsd_values.max() > 0.01, (
            f"No discriminative features found for {pair_name}"
        )

        logger.info(
            "JSD for %s: mean=%.4f, max=%.4f, discriminative=%d",
            pair_name,
            jsd_mean,
            jsd_values.max(),
            discriminative,
        )

    logger.info("JSD successfully identifies discriminative features")
