"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound, load_paths_for_compound
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
import pandas as pd
from fepa.flows.torsions_flow import torsions_analysis_workflow
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
    plot_pca_components,
)
import mdaencore as encore
from fepa.core.dim_reducers import PCADimReducer, UMAPDimReducer
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.analyzers import compute_relative_entropy
from fepa.utils.feature_utils import (
    convert_features_df_w_components_to_angles,
    convert_features_df_w_angles_to_w_components,
)
import seaborn as sns
import re
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = "wdir"
    analysis_output_dir = parent_output_dir
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    cmps_of_interest = ["9", "12"]
    # cmps_of_interest = ["46905"]

    for cmp in cmps_of_interest:
        cluster_df = pd.read_csv(
            os.path.join(analysis_output_dir, f"{cmp}_conformation_cluster_df.csv")
        )

        # Plot a barplot of clusters in each ensemble
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=cluster_df,
            x="ensemble",
            hue="cluster",
            palette="Set2",
            order=sorted(cluster_df["ensemble"].unique()),
        )
        plt.title(f"Cluster distribution in each ensemble for {cmp}")
        plt.xlabel("Ensemble")
        plt.ylabel("Count")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"{cmp}_conformation_cluster_distribution.png",
            )
        )

        # Plot timeseries
        sns.set(style="whitegrid")
        cluster_df_van = cluster_df.copy()
        # Keep only rows where ensemble column doesnt contain nvt or coul
        cluster_df_van = cluster_df_van[
            ~cluster_df_van["ensemble"].str.contains("nvt|coul", regex=True)
        ]

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=cluster_df_van, x="timestep", y="cluster", hue="ensemble")

        # Add plot labels and title
        plt.xlabel("Timestep")
        plt.ylabel("Cluster")
        plt.title("Cluster vs Timestep Colored by Ensemble")
        plt.legend(title="Ensemble", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Show the plot
        plt.savefig(
            os.path.join(
                analysis_output_dir, f"{cmp}_conformation_cluster_timeseries.png"
            )
        )

        # Plot the fraction of each  cluster in each ensemble in a stacked bar plot
        cluster_df_stacked = cluster_df.copy()
        # Count cluster occurrences per ensemble
        cluster_counts = (
            cluster_df_stacked.groupby(["ensemble", "cluster"])
            .size()
            .unstack(fill_value=0)
        )

        # Convert counts to fractions
        cluster_fractions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

        # Plot stacked bar chart
        cluster_fractions.plot(
            kind="bar", stacked=True, figsize=(10, 6), colormap="tab20"
        )

        # Add labels and legend
        plt.xlabel("Ensemble")
        plt.ylabel("Fraction")
        plt.title("Fraction of Each Cluster in Each Ensemble")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"{cmp}_conformation_cluster_fraction_stacked_bar.png",
            )
        )
        plt.close()


if __name__ == "__main__":
    main()
