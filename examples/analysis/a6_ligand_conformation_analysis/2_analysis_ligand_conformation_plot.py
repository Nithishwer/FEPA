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
    parent_output_dir = "./wdir"
    analysis_output_dir = parent_output_dir
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Creating van_list and leg_window_list
    # van_list = [i for i in range(1, 4)]
    leg_window_list = [f"coul.{i:02}" for i in range(0, 11)]
    # [f"coul.{i:02}" for i in range(0, 11)]
    # + [f"vdw.{i:02}" for i in range(0, 12)]
    # + [f"rest.{i:02}" for i in range(0, 11)]
    # )

    # cmps_of_interest = ["48951", "47594", "49599", "52542", "47821", "46905", "48845"]

    # Prepare paths
    # path_dict = load_abfe_paths_for_compound(
    #     config,
    #     cmp="42922",
    #     bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    #     van_list=van_list,
    #     leg_window_list=leg_window_list,
    #     apo=False,
    # )

    for cmp in config["compounds"][:]:
        # Create cmp output path
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)

        # Load paths
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp=cmp,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            van_list=[1, 2, 3],
            leg_window_list=leg_window_list,
            apo=False,
        )

        logging.info("Path dict: %s", path_dict)

        # Load trajectories
        ensemble_handler = EnsembleHandler(path_dict)
        cluster_df = pd.read_csv(
            os.path.join(cmp_output_dir, f"{cmp}_conformation_cluster_df.csv")
        )

        # Plot a normalized stacked barplot of clusters in each ensemble (all stacks sum to 100)
        cluster_counts = (
            cluster_df.groupby(["ensemble", "cluster"]).size().unstack(fill_value=0)
        )
        cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

        cluster_props.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 6),
            # colormap="Set2",
        )
        plt.title(f"Cluster distribution in each ensemble for {cmp} (normalized to 100%)")
        plt.xlabel("Ensemble")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=90)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
            cmp_output_dir,
            f"{cmp}_conformation_cluster_distribution.png",
            )
        )
        plt.close()

        # # Plot each ensemble
        # clusters = cluster_df["cluster"].unique()
        # colors = plt.cm.tab10.colors  # or use any colormap you like
        # color_map = {
        #     cluster: colors[i % len(colors)] for i, cluster in enumerate(clusters)
        # }

        # # Plot
        # fig, ax = plt.subplots(figsize=(12, 3))

        # for i in range(len(cluster_df) - 1):
        #     start = cluster_df.iloc[i]["timestep"]
        #     end = cluster_df.iloc[i + 1]["timestep"]
        #     cluster = cluster_df.iloc[i]["cluster"]
        #     ax.plot(
        #         [start, end], [cluster, cluster], color=color_map[cluster], linewidth=5
        #     )

        # ax.set_xlabel("Timestep")
        # ax.set_ylabel("Cluster")
        # ax.set_title("Cluster over Time")
        # ax.grid(True)

        # # Optional: Create a legend
        # handles = [
        #     plt.Line2D(
        #         [0], [0], color=color_map[cluster], lw=4, label=f"Cluster {cluster}"
        #     )
        #     for cluster in clusters
        # ]
        # ax.legend(
        #     handles=handles,
        #     title="Clusters",
        #     bbox_to_anchor=(1.05, 1),
        #     loc="upper left",
        # )

        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(
        #         cmp_output_dir,
        #         f"{cmp}_conformation_cluster_over_time.png",
        #     )
        # )
        # plt.close()


if __name__ == "__main__":
    main()
