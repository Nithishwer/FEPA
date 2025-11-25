"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
import pandas as pd
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
    leg_window_list = [f"coul.{i:02}" for i in range(0, 11)]
    +[f"coul.{i:02}" for i in range(0, 11)]
    +[f"vdw.{i:02}" for i in range(0, 12)]
    +[f"rest.{i:02}" for i in range(0, 11)]

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
        plt.title(
            f"Cluster distribution in each ensemble for {cmp} (normalized to 100%)"
        )
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


if __name__ == "__main__":
    main()
