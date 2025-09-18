import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_apo, load_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    compute_histograms,
    plot_jsd_histograms,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.analyzers import compute_relative_entropy


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def add_offset_to_residues(input_str, offset):
    parts = input_str.strip().split(' ')
    if len(parts) == 4:
        # parts[3] is the residue ID, convert and add offset
        new_resid = str(int(parts[3]) + offset)
        new_string = f"{parts[0]} {parts[1]} {parts[2]} {new_resid}"
        return new_string
    else:
        logging.warning("Unexpected format in chunk: %s", input_str)
        return input_str  # Keep unchanged if format is unexpected

def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a10_sidechain_torsions_analysis", "wdir")
    bp_waters_output_dir = os.path.join("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a3_compare_bp_waters/a3_compare_bp_waters/wdir")
    resid_offset = 567

    for cmp in config["compounds"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)


        # Using the angles df instead of the components df so the jsd histograms can be interpreted easily
        features_df = pd.read_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features.csv"))
        # subset only van ensembles
        features_df = features_df[features_df["ensemble"].str.contains("van")]
        # Add offsets to feature names
        features_df.columns = [add_offset_to_residues(col, resid_offset) if col.startswith("CHI") else col for col in features_df.columns]

        # Convert all angles to radians
        for col in features_df.columns:
            if col.startswith("CHI"):
                features_df[col] = np.mod(np.deg2rad(features_df[col]), 2 * np.pi)

        # Projection df without nvt:
        projection_df_clustered = pd.read_csv(os.path.join(cmp_output_dir, "pca_projection_df_clustered.csv"))
        # Subset only van ensembles
        projection_df_clustered = projection_df_clustered[projection_df_clustered["ensemble"].str.contains("van")]

        # Transfer clustering to features_df
        assert features_df["ensemble"].equals(
            projection_df_clustered["ensemble"]
        ) and features_df["timestep"].equals(
            projection_df_clustered["timestep"]), "Mismatch in ensemble series between feature_df and projection_df"
        features_df["cluster"] = projection_df_clustered["cluster"]

        # Plot jsd among clusters
        clusters = projection_df_clustered["cluster"].unique()
        # sort the clusters alphabetically
        clusters = sorted(clusters)

        # Loop over all pairs of elements in clusters
        for i, cluster_1 in enumerate(clusters):
            for j, cluster_2 in enumerate(clusters[i + 1 :], start=i + 1):
                # Load relative entropy df
                relative_entropy_df = pd.read_csv(
                    os.path.join(
                        cmp_output_dir,
                        f"relative_entropy_{cluster_1}_{cluster_2}.csv",
                    ),
                )
                # Get the histograms after sorting by jsd
                relative_entropy_df = relative_entropy_df.sort_values(
                    by="jsd", ascending=False
                )
                # Get the top 8 features
                top_features = relative_entropy_df.head(10)['name'].values
                # Add offset to top residues
                top_features = [add_offset_to_residues(feature, resid_offset) for feature in top_features]
                # Plot the top 10 features from features_df vs the water occupancy in different y axes
                fig, axes = plt.subplots(
                    nrows=5,
                    ncols=2,
                    figsize=(10, 20),
                )
                fig.suptitle(
                    f"Top 10 features from {cluster_1} vs {cluster_2} relative entropy",
                    fontsize=16,
                )
                # Loop over the top features
                for i, feature in enumerate(top_features):
                    # Plot the feature
                    sns.lineplot(
                        data=features_df,
                        x="timestep",
                        y=feature,
                        hue="ensemble",
                        ax=axes[i // 2, i % 2],
                        linewidth=0.8,  # Set line width to be very thin
                        alpha=0.7,  # Set transparency
                    )
                    axes[i // 2, i % 2].set_title(feature)
                    axes[i // 2, i % 2].set_xlabel("Timestep")
                    axes[i // 2, i % 2].set_ylabel("Feature Value (radians)")
                    axes[i // 2, i % 2].legend(title="Ensemble")
                    
                    # Add y-axis markers for relevant points (e.g., 0, π/2, π, 3π/2, 2π)
                    y_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
                    y_tick_labels = ["0", "π/2", "π", "3π/2", "2π"]
                    axes[i // 2, i % 2].set_yticks(y_ticks)
                    axes[i // 2, i % 2].set_yticklabels(y_tick_labels)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.savefig(
                    os.path.join(
                        cmp_output_dir,
                        f"top_10_features_{cluster_1}_{cluster_2}_van.png",
                    ),
                    dpi=300,
                )
                plt.close(fig)
        





  
if __name__ == "__main__":
    main()
