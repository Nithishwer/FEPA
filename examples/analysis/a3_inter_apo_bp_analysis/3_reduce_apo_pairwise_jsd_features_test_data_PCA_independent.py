"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
    compute_histograms,
    plot_jsd_histograms,
)
from fepa.core.analyzers import compute_relative_entropy

from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import check_bp_residue_consistency
from fepa.utils.dimred_utils import cluster_pca, get_ensemble_center, make_ensemble_center_df
from fepa.utils.plumed_utils import (
    write_plumed_file,
    write_plumed_restraints,
    make_restraint_array_from_ensemble_centers,
)
from fepa.utils.path_utils import load_paths_for_apo
from fepa.utils.feature_utils import filter_top_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative path templates in the config (in-memory only)."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(config)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out


def main():
    """Main function to run the analysis (paths & discovery match the first script)"""

    # --- Match the first script's path setup ---
    repo_root = Path("/biggin/b230/magd5710/Documents/FEPA").resolve()
    config_path = repo_root / "tests" / "test_config" / "config_allapo.json"

    # Load configuration and absolutize templates like in the first script
    config = load_config(config_path)
    config = _abspath_templates(config, repo_root)

    # Output locations (match first script)
    parent_output_dir = repo_root / "tests" / "test_data" / "3_expected"
    analysis_output_dir = parent_output_dir

    # (Kept for reference; not used below because we read per-pair top_features from disk)
    cmp = "1"
    existing_analysis_dir = repo_root / "tests" / "test_data" / "1_expected"

    # Discover apo ensembles exactly like the first script
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    desired_order = ["apo_1", "apo_2", "apo_3"]
    present = [e for e in desired_order if e in path_dict] or sorted(path_dict.keys())
    if len(present) < 2:
        raise RuntimeError(f"Need at least two apo ensembles. Found {sorted(path_dict.keys())}")
    comparison_pairs = [(present[i], present[i + 1]) for i in range(len(present) - 1)]

    logging.info("Available apo ensembles: %s", sorted(path_dict.keys()))
    logging.info("Comparison pairs: %s", comparison_pairs)

    # Pair → cluster mapping (kept for compatibility with other steps; not used for JSD now)
    ensemble_to_cluster = {
        ("apo_1", "apo_2"): (0, 1),
        ("apo_2", "apo_3"): (2, 1),
    }
    default_pair_clusters = (0, 1)

    for pair in comparison_pairs[:]:
        pair_name = f"{pair[0]}_{pair[1]}"
        apo_output_dir = analysis_output_dir / pair_name
        apo_output_dir.mkdir(parents=True, exist_ok=True)

        # Load the per-pair top_features created by the first script
        top_csv = apo_output_dir / f"top_features_{pair[0]}_{pair[1]}.csv"
        if not top_csv.exists():
            raise FileNotFoundError(
                f"Expected top features CSV not found: {top_csv}\n"
                "Make sure you've run the first script to generate per-pair top features."
            )
        top_features_df = pd.read_csv(top_csv)

        # --- OPTIONAL: You may still run PCA/cluster on top_features for other plots/uses ---
        # (left here to preserve functionality/structure; not used for entropy anymore)
        dimreducer_top = PCADimReducer(top_features_df, n_components=8)
        dimreducer_top.reduce_dimensions()
        dimreducer_top.calculate_projections()
        dimreducer_top.save_projection_df(
            save_path=str(apo_output_dir / f"pca_projection_df_top_{pair[0]}_{pair[1]}.csv")
        )
        dimreducer_top.save_pca(save_path=str(apo_output_dir / f"pca_top_{pair[0]}_{pair[1]}.pkl"))

        pca_projection_df_clustered_top = cluster_pca(
            dimreducer_top.get_pca_projection_df(), n_clusters=3, n_components=8
        )

        # ======== PCA-INDEPENDENT ENTROPY/JSD (requested change) ========
        # Clean NaNs in DIST columns for stable entropy estimates
        dist_cols = top_features_df.filter(regex=r"\bDIST\b", axis=1).columns
        if len(dist_cols) == 0:
            raise ValueError("No DIST columns found in top_features_df.")
        top_features_df = top_features_df.dropna(subset=dist_cols, how="any")

        # Compute JSD between ensembles directly (no clusters, no PCA)
        relative_entropy_dict = compute_relative_entropy(
            feature_df=top_features_df,
            ensemble1=pair[0],
            ensemble2=pair[1],
            num_bins=20,
            key="ensemble",                  # <-- compare by ensemble
            feature_column_keyword="DIST",
        )
        histograms = compute_histograms(
            feature_df=top_features_df,
            key="ensemble",                  # <-- compare by ensemble
            value_1=pair[0],
            value_2=pair[1],
            num_bins=20,
            feature_column_keyword="DIST",
        )

        # Save plots with requested suffix
        plot_jsd_histograms(
            histograms=histograms,
            relative_entropy_dict=relative_entropy_dict,
            top_n=16,
            save_path=str(apo_output_dir / "jsd_histograms_PCA_independent_entropy.png"),
        )
        # =================================================================


if __name__ == "__main__":
    main()
