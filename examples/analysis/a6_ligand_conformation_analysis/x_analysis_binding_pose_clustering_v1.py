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
    analysis_output_dir = os.path.join("wdir")
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Creating van_list and leg_window_list
    # van_list = [i for i in range(1, 4)]
    leg_window_list = [f"coul.{i:02}" for i in range(0, 11)]
    # [f"coul.{i:02}" for i in range(0, 11)]
    # + [f"vdw.{i:02}" for i in range(0, 12)]
    # + [f"rest.{i:02}" for i in range(0, 11)]
    # )

    # cmps_of_interest = ["9", "12"]

    for cmp in config["compounds"]:
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
        ensemble_handler.make_universes()
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        sel = "resname unk"
        universe_dict = ensemble_handler.get_universe_dict()

        # Define clustering method
        dbscan_method = encore.DBSCAN(
            eps=0.5, min_samples=5, algorithm="auto", leaf_size=30
        )

        # Define the ensemble list
        ensemble_list = list(universe_dict.values())
        # Cluster the binding poses
        cluster_collection = encore.cluster(
            ensembles=ensemble_list,
            select=sel,
            superimposition_subset="name CA",
            method=dbscan_method,
        )

        # Getting a list of framewise cluster values for the universes in the list
        # Create a list with nones the size of the universe_frames
        cluster_series = []
        timstep_series = []
        ensemble_series = []
        for ensemble, universe in universe_dict.items():
            timstep_series = (
                timstep_series
                + ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            cluster_series = cluster_series + [None] * len(universe.trajectory)
            ensemble_series = ensemble_series + [ensemble] * len(universe.trajectory)

        # Making a cluster list for every frame in the universe list
        for i, cluster in enumerate(cluster_collection):
            logging.info("Cluster %d: %s", i, cluster)
            cluster_id = cluster.id
            indices = cluster.elements
            for index in indices:
                cluster_series[index] = cluster_id
            # Dump cluster centroid frame to output directory
            centroid_id = cluster.centroid
            centroid_ensemble = ensemble_series[centroid_id]
            centroid_timestep = timstep_series[centroid_id]
            ensemble_handler.dump_gro(
                ensemble=centroid_ensemble,
                timestep=centroid_timestep,
                save_path=os.path.join(
                    analysis_output_dir,
                    f"{cmp}_conformation_cluster_{cluster_id}.gro",
                ),
            )
            logging.info(f"No of elements in cluster :{len(indices)}")

        # Create a DataFrame with the cluster series
        cluster_df = pd.DataFrame(
            {
                "timestep": timstep_series,
                "ensemble": ensemble_series,
                "cluster": cluster_series,
            }
        )
        cluster_df.to_csv(
            os.path.join(analysis_output_dir, f"{cmp}_conformation_cluster_df.csv"),
            index=False,
        )

        # # Plot a barplot of clusters in each ensemble
        # plt.figure(figsize=(10, 6))
        # sns.countplot(
        #     data=cluster_df,
        #     x="ensemble",
        #     hue="cluster",
        #     palette="Set2",
        #     order=sorted(cluster_df["ensemble"].unique()),
        # )
        # plt.title(f"Cluster distribution in each ensemble for {cmp}")
        # plt.xlabel("Ensemble")
        # plt.ylabel("Count")
        # plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(
        #         analysis_output_dir,
        #         f"{cmp}_conformation_cluster_distribution.png",
        #     )
        # )

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
        #         analysis_output_dir,
        #         f"{cmp}_conformation_cluster_over_time.png",
        #     )
        # )
        # plt.close()


if __name__ == "__main__":
    main()
