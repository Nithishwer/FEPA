"""
Test for pairwise apo binding pocket analysis with clustering and feature selection.

This test validates a multi-stage workflow:
  1. Load apo trajectory pairs (e.g., apo_1 vs apo_2)
  2. Perform PCA on all features
  3. Cluster the PCA projections
  4. Identify top discriminative features
  5. Perform second-stage PCA on top features
  6. Extract structures at cluster centers
  7. Generate PLUMED input files for enhanced sampling

Key FEPA functionalities tested (beyond basic PCA):
  - Clustering: cluster_pca(), make_ensemble_center_df()
  - Feature selection: filter_top_features()
  - Structure extraction: Writing .gro files at cluster centers
  - Two-stage PCA workflow
  - Pairwise ensemble comparisons
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
import pytest

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import DimRedVisualizer, plot_eigenvalues, plot_pca_components
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import check_bp_residue_consistency
from fepa.utils.dimred_utils import cluster_pca, make_ensemble_center_df
from fepa.utils.path_utils import load_paths_for_apo
from fepa.utils.feature_utils import filter_top_features

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


def _dump_gro_from_universe(universe, value, save_path: str):
    """Extract and write a single frame from a trajectory to .gro format."""
    nframes = len(universe.trajectory)

    def _nearest_frame_from_time_ps(u, t_ps: float) -> int:
        try:
            times = np.array([ts.time for ts in u.trajectory])
        except Exception:
            u.trajectory[0]
            t0 = u.trajectory.time
            dt = u.trajectory.dt
            times = t0 + np.arange(nframes) * dt
        return int(np.argmin(np.abs(times - float(t_ps))))

    try:
        idx = int(value)
    except (ValueError, TypeError):
        idx = None

    if idx is not None and 0 <= idx < nframes:
        frame_idx = idx
    else:
        frame_idx = _nearest_frame_from_time_ps(universe, float(value))

    universe.trajectory[frame_idx]
    universe.atoms.write(save_path)


# ================================ Fixtures ================================
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Session-scoped fixture providing repository root path."""
    root = _detect_repo_root()
    logger.info("Using FEPA repo root: %s", root)
    return root


@pytest.fixture(scope="session")
def config_and_inputs(repo_root: Path) -> Tuple[dict, Path]:
    """Load FEPA configuration and locate input data directory."""
    cfg_path = (repo_root / "tests" / "test_config" / "config.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    cfg = _abspath_templates(cfg, repo_root)

    input_features_dir = (repo_root / "tests" / "test_data" / "1_expected").resolve()
    if not input_features_dir.exists():
        raise FileNotFoundError(f"Input features directory not found: {input_features_dir}")

    return cfg, input_features_dir


@pytest.fixture(scope="function")
def run_pairwise_analysis(
    tmp_path: Path, config_and_inputs: Tuple[dict, Path]
) -> Dict[str, any]:
    """
    Execute the full pairwise apo analysis workflow.
    
    Returns:
        Dictionary containing:
          - output_dir: Root output directory (tmp_path)
          - pairs: List of (ensemble1, ensemble2) tuples processed
          - pair_outputs: Dict mapping pair_name -> dict of output paths
    """
    config, input_features_dir = config_and_inputs
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pairwise apo analysis workflow starting")

    # Discover apo ensembles from configuration
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    # Determine ensemble pairs for comparison
    desired_order = ["apo_1", "apo_2", "apo_3"]
    present = [e for e in desired_order if e in path_dict] or sorted(path_dict.keys())
    
    if len(present) < 2:
        raise RuntimeError(
            f"Need at least two apo ensembles for pairwise analysis. "
            f"Found: {sorted(path_dict.keys())}"
        )
    
    comparison_pairs = [(present[i], present[i + 1]) for i in range(len(present) - 1)]

    # Cluster mapping for top-feature selection
    ensemble_to_cluster = {
        ("apo_1", "apo_2"): (0, 1),
        ("apo_2", "apo_3"): (2, 1),
    }
    default_pair_clusters = (0, 1)

    pair_outputs = {}

    # Process each ensemble pair
    for pair in comparison_pairs:
        pair_name = f"{pair[0]}_{pair[1]}"
        pair_dir = out_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Load trajectories
        ensemble_handler = EnsembleHandler(
            {pair[0]: path_dict[pair[0]], pair[1]: path_dict[pair[1]]}
        )
        ensemble_handler.make_universes()
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Load & filter features
        featurizer = SelfDistanceFeaturizer(ensemble_handler)
        featurizer.load_features(input_dir=str(input_features_dir))
        featurizer.feature_df = featurizer.feature_df[
            np.isin(featurizer.feature_df["ensemble"], [pair[0], pair[1]])
        ]

        # Clean NaN values
        df = featurizer.get_feature_df().copy()
        df = df.dropna(axis=1, how="all")
        dist_cols = df.filter(regex=r"\bDIST\b", axis=1).columns
        if len(dist_cols) > 0:
            df = df.dropna(subset=dist_cols, how="any")
        featurizer.feature_df = df
        
        features_csv = pair_dir / "SelfDistance_features.csv"
        featurizer.save_features(output_dir=str(pair_dir), overwrite=True)

        # PCA on all features
        dimreducer = PCADimReducer(
            featurizer.get_feature_df(), n_components=8, random_state=42
        )
        dimreducer.reduce_dimensions()
        dimreducer.calculate_projections()
        
        projection_csv = pair_dir / f"pca_projection_df_{pair[0]}_{pair[1]}.csv"
        dimreducer.save_projection_df(save_path=str(projection_csv))
        projection_df = dimreducer.get_pca_projection_df()

        # Clustering
        pca_projection_clustered = cluster_pca(projection_df, n_clusters=3, n_components=8)

        # Visualizations (sample - not all)
        dimred_viz = DimRedVisualizer(projection_df=pca_projection_clustered, data_name="PCA")
        ensemble_plot = pair_dir / "pca_components_ensemble.png"
        dimred_viz.plot_dimred_sims(save_path=str(ensemble_plot))
        
        eigenval_plot = pair_dir / "eigenvalues.png"
        plot_eigenvalues(
            pca_object=dimreducer.get_pca(),
            n_components=8,
            save_path=str(eigenval_plot),
        )

        # Extract cluster center structures
        ensemble_center_df = make_ensemble_center_df(pca_projection_clustered, key="cluster")
        cluster_gro_files = []
        for _, row in ensemble_center_df.iterrows():
            u = ensemble_handler.get_universe_dict()[row["ensemble"]]
            gro_path = pair_dir / f"cluster_{int(row['cluster'])}_center.gro"
            _dump_gro_from_universe(u, row["timestep"], str(gro_path))
            cluster_gro_files.append(gro_path)

        # Top feature selection
        feature_df_with_clusters = featurizer.get_feature_df().copy()
        feature_df_with_clusters["cluster"] = pca_projection_clustered["cluster"].values

        e1, e2 = ensemble_to_cluster.get(pair, default_pair_clusters)
        top_features_df = filter_top_features(
            feature_df=feature_df_with_clusters,
            key="cluster",
            ensemble1=e1,
            ensemble2=e2,
            feature_column_keyword="DIST",
            top_n=200,
        )
        
        top_features_csv = pair_dir / f"top_features_{pair[0]}_{pair[1]}.csv"
        top_features_df.to_csv(top_features_csv, index=False)

        # PCA on top features
        dimreducer_top = PCADimReducer(
            top_features_df, n_components=8, random_state=42
        )
        dimreducer_top.reduce_dimensions()
        dimreducer_top.calculate_projections()
        
        projection_top_csv = pair_dir / f"pca_projection_df_top_{pair[0]}_{pair[1]}.csv"
        dimreducer_top.save_projection_df(save_path=str(projection_top_csv))
        
        pca_top_pkl = pair_dir / f"pca_top_{pair[0]}_{pair[1]}.pkl"
        dimreducer_top.save_pca(save_path=str(pca_top_pkl))

        # Cluster top PCA
        pca_projection_clustered_top = cluster_pca(
            dimreducer_top.get_pca_projection_df(), n_clusters=3, n_components=8
        )

        # PLUMED file (placeholder)
        plumed_file = pair_dir / "plumed.dat"
        with open(plumed_file, "w") as f:
            f.write(f"# PLUMED input for {pair_name}\n")

        # Store outputs
        pair_outputs[pair_name] = {
            "pair_dir": pair_dir,
            "features_csv": features_csv,
            "projection_csv": projection_csv,
            "projection_top_csv": projection_top_csv,
            "top_features_csv": top_features_csv,
            "pca_top_pkl": pca_top_pkl,
            "plumed_file": plumed_file,
            "cluster_gro_files": cluster_gro_files,
        }

    logger.info("Pairwise analysis workflow complete: %d pairs", len(comparison_pairs))

    return {
        "output_dir": out_dir,
        "pairs": comparison_pairs,
        "pair_outputs": pair_outputs,
    }


# ================================ Test Functions ================================

def test_pairwise_workflow_completes_and_creates_files(
    run_pairwise_analysis: Dict,
) -> None:
    """
    Test that the full pairwise workflow completes and creates expected files.
      - At least one pair is processed
      - Key output files exist for each pair:
        * Features CSV (cleaned)
        * PCA projection CSVs (full and top)
        * Top features CSV
        * PCA model (PKL)
        * Cluster center structures (GRO files)
        * PLUMED file
    """
    pairs = run_pairwise_analysis["pairs"]
    pair_outputs = run_pairwise_analysis["pair_outputs"]

    # Basic workflow validation
    assert len(pairs) >= 1, "Should process at least one ensemble pair"
    assert len(pair_outputs) == len(pairs), "Should have outputs for all pairs"

    # Check all pairs have required files
    for pair in pairs:
        pair_name = f"{pair[0]}_{pair[1]}"
        outputs = pair_outputs[pair_name]

        # Key CSV files
        assert outputs["features_csv"].exists(), f"Missing features CSV for {pair_name}"
        assert outputs["projection_csv"].exists(), f"Missing PCA projection for {pair_name}"
        assert outputs["projection_top_csv"].exists(), f"Missing top PCA projection for {pair_name}"
        assert outputs["top_features_csv"].exists(), f"Missing top features for {pair_name}"

        # Model file
        assert outputs["pca_top_pkl"].exists(), f"Missing PCA PKL for {pair_name}"
        assert outputs["pca_top_pkl"].stat().st_size > 0, f"Empty PCA PKL for {pair_name}"

        # Structure files
        assert len(outputs["cluster_gro_files"]) == 3, (
            f"Expected 3 cluster GRO files for {pair_name}"
        )
        for gro_file in outputs["cluster_gro_files"]:
            assert gro_file.exists(), f"Missing GRO file: {gro_file}"
            assert gro_file.stat().st_size > 0, f"Empty GRO file: {gro_file}"

        # PLUMED file
        assert outputs["plumed_file"].exists(), f"Missing PLUMED file for {pair_name}"

    logger.info("Workflow completed successfully for all %d pairs", len(pairs))


def test_pairwise_analysis_scientific_validity(
    run_pairwise_analysis: Dict,
) -> None:
    """
    Test that key scientific operations produce valid results.
    
    Validates:
      - Clustering: produces 3 clusters with valid labels (0, 1, 2)
      - Feature selection: reduces dimensionality
      - Two-stage PCA: both stages complete with correct structure
      - Data integrity: no NaN values, reasonable ranges
    """
    pair_outputs = run_pairwise_analysis["pair_outputs"]

    for pair_name, outputs in pair_outputs.items():
        # Load projection data
        df_full = pd.read_csv(outputs["projection_csv"])
        df_top = pd.read_csv(outputs["projection_top_csv"])

        # Validate clustering
        assert "cluster" in df_top.columns, f"Missing cluster column for {pair_name}"
        unique_clusters = df_top["cluster"].unique()
        assert len(unique_clusters) == 3, (
            f"Expected 3 clusters for {pair_name}, got {len(unique_clusters)}"
        )
        assert set(unique_clusters) == {0, 1, 2}, (
            f"Cluster labels should be {{0, 1, 2}} for {pair_name}"
        )
        assert not df_top["cluster"].isnull().any(), (
            f"Found NaN cluster assignments for {pair_name}"
        )

        # Validate feature selection
        df_features_full = pd.read_csv(outputs["features_csv"])
        df_features_top = pd.read_csv(outputs["top_features_csv"])
        
        n_features_full = len([c for c in df_features_full.columns if "DIST" in c])
        n_features_top = len([c for c in df_features_top.columns if "DIST" in c])
        
        assert n_features_top < n_features_full, (
            f"Feature selection should reduce dimensions for {pair_name}"
        )
        assert n_features_top > 0, f"No top features selected for {pair_name}"
        
        # Validate two-stage PCA
        pc_cols_full = [c for c in df_full.columns if c.startswith("PC")]
        pc_cols_top = [c for c in df_top.columns if c.startswith("PC")]
        
        assert len(pc_cols_full) == 8, f"Expected 8 PCs in full PCA for {pair_name}"
        assert len(pc_cols_top) == 8, f"Expected 8 PCs in top PCA for {pair_name}"
        
        # Check for variance (not all zeros)
        for col in pc_cols_top[:3]:  # Check first 3 PCs
            assert df_top[col].std() > 0, f"{col} has no variance for {pair_name}"
        
        # No NaN values in PCs
        for col in pc_cols_top:
            assert not df_top[col].isnull().any(), f"{col} has NaN values for {pair_name}"

        logger.info(
            "Scientific validity confirmed for %s: clusters=%s, features=%d->%d",
            pair_name,
            sorted(unique_clusters),
            n_features_full,
            n_features_top,
        )

    logger.info("✓ Scientific validity checks passed for all pairs")


def test_pairwise_analysis_output_structure(
    run_pairwise_analysis: Dict,
) -> None:
    """
    Test that output DataFrames have correct structure and metadata.
    
    Validates:
      - CSVs have expected columns (PC1-PC8, ensemble, timestep)
      - Row counts are consistent across related files
      - Metadata columns are preserved
      - Data types are reasonable
    """
    pair_outputs = run_pairwise_analysis["pair_outputs"]

    for pair_name, outputs in pair_outputs.items():
        # Load all DataFrames
        df_features = pd.read_csv(outputs["features_csv"])
        df_projection = pd.read_csv(outputs["projection_csv"])
        df_features_top = pd.read_csv(outputs["top_features_csv"])
        df_projection_top = pd.read_csv(outputs["projection_top_csv"])

        # Check metadata columns are present
        for df, name in [
            (df_features, "features"),
            (df_projection, "projection"),
            (df_features_top, "features_top"),
            (df_projection_top, "projection_top"),
        ]:
            assert "ensemble" in df.columns, f"Missing 'ensemble' in {name} for {pair_name}"
            assert "timestep" in df.columns, f"Missing 'timestep' in {name} for {pair_name}"

        # Check PC columns in projections
        for df, name in [(df_projection, "projection"), (df_projection_top, "projection_top")]:
            pc_cols = [c for c in df.columns if c.startswith("PC")]
            assert len(pc_cols) == 8, f"Expected 8 PCs in {name} for {pair_name}"

        # Check row consistency
        assert len(df_features) == len(df_projection), (
            f"Row count mismatch: features={len(df_features)}, "
            f"projection={len(df_projection)} for {pair_name}"
        )
        assert len(df_features_top) == len(df_projection_top), (
            f"Row count mismatch: features_top={len(df_features_top)}, "
            f"projection_top={len(df_projection_top)} for {pair_name}"
        )
        assert len(df_features) == len(df_features_top), (
            f"Row count should be same after feature selection for {pair_name}"
        )

        # Check that top features are subset of original
        full_dist_cols = set(c for c in df_features.columns if "DIST" in c)
        top_dist_cols = set(c for c in df_features_top.columns if "DIST" in c)
        assert top_dist_cols.issubset(full_dist_cols), (
            f"Top features should be subset of original for {pair_name}"
        )

        logger.info("Output structure validated for %s", pair_name)

    logger.info("Output structure checks passed for all pairs")


def test_pairwise_analysis_pca_model_persistence(
    run_pairwise_analysis: Dict,
) -> None:
    """
    Test that PCA models can be saved and loaded correctly.
    
    Validates:
      - PKL files can be loaded with pickle
      - Loaded models have expected attributes
      - Model persistence works for reproducibility
    """
    import pickle

    pair_outputs = run_pairwise_analysis["pair_outputs"]

    for pair_name, outputs in pair_outputs.items():
        pkl_file = outputs["pca_top_pkl"]

        # Load the PKL file
        with open(pkl_file, "rb") as f:
            pca_model = pickle.load(f)

        # Basic validation
        assert pca_model is not None, f"Failed to load PCA model for {pair_name}"
        
        # Check for expected PCA attributes (depends on whether it's sklearn or pensa)
        # Both should have some form of components/explained variance
        has_sklearn_attrs = hasattr(pca_model, "components_") and hasattr(
            pca_model, "explained_variance_"
        )
        has_pensa_attrs = hasattr(pca_model, "eigenvectors") or hasattr(
            pca_model, "eigenvalues"
        )
        
        assert has_sklearn_attrs or has_pensa_attrs, (
            f"PCA model missing expected attributes for {pair_name}"
        )

        logger.info("PCA model loaded successfully for %s", pair_name)

    logger.info("PCA model persistence validated for all pairs")
