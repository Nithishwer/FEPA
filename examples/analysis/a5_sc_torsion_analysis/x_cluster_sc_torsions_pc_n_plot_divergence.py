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


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a10_sidechain_torsions_analysis", "wdir")
    bp_waters_output_dir = os.path.join("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a3_compare_bp_waters/a3_compare_bp_waters/wdir")

    for cmp in config["compounds"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)

        # path_dict = load_paths_for_apo(
        #     config,
        #     bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket 
        #     )

        path_dict = load_paths_for_compound(
            config,
            cmp,
            bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket
            apo = True
        )

        # Using the angles df instead of the components df so the jsd histograms can be interpreted easily
        features_df = pd.read_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features.csv"))

        # Projection df without nvt:
        projection_df = pd.read_csv(os.path.join(cmp_output_dir, "pca_projection_df.csv"))

        # clustering
        projection_df_clustered = cluster_pca(
            projection_df,
            n_clusters=3,
            n_components=3,
        )

        # Save clustered projection df
        projection_df_clustered.to_csv(
            os.path.join(cmp_output_dir, "pca_projection_df_clustered.csv"),
            index=False,
        )

        # Visualize the clustering
        dim_red_visualizer = DimRedVisualizer(
            projection_df=projection_df_clustered,
        )
        dim_red_visualizer.plot_dimred_cluster(
            cluster_column="cluster",
            save_path=os.path.join(cmp_output_dir, "pc1_pc2_scatter_clusters.png"),
        )
        # Transfer clustering to features_df
        assert features_df["ensemble"].equals(
            projection_df_clustered["ensemble"]
        ) and features_df["timestep"].equals(
            projection_df_clustered["timestep"]), "Mismatch in ensemble series between feature_df and projection_df"
        features_df["cluster"] = projection_df_clustered["cluster"]

        # Plot jsd among clusters
        clusters = projection_df_clustered["cluster"].unique()
        # sort clusters alphabetically
        clusters = sorted(clusters)

        # Loop over all pairs of elements in clusters
        for i, cluster_1 in enumerate(clusters):
            for j, cluster_2 in enumerate(clusters[i + 1 :], start=i + 1):
                logging.info(
                    "Computing JSD for clusters %s and %s", cluster_1, cluster_2
                )
                relative_entropy_dict = compute_relative_entropy(
                    feature_df=features_df,
                    key="cluster",
                    ensemble1=cluster_1,
                    ensemble2=cluster_2,
                    num_bins=50,
                    feature_column_keyword="CHI",
                )
                histograms = compute_histograms(
                    feature_df=features_df,
                    key="cluster",
                    value_1=cluster_1,
                    value_2=cluster_2,
                    num_bins=50,
                    feature_column_keyword="CHI",
                )
                plot_jsd_histograms(
                    relative_entropy_dict=relative_entropy_dict,
                    histograms=histograms,
                    save_path=os.path.join(
                        cmp_output_dir,
                        f"jsd_histograms_{cluster_1}_{cluster_2}.png",
                    ),
                )
                # Save relative entropy dict
                relative_entropy_df = pd.DataFrame(relative_entropy_dict)
                # Sort according to jsd
                relative_entropy_df = relative_entropy_df.sort_values(
                    by="jsd", ascending=False
                )
                relative_entropy_df.to_csv(
                    os.path.join(
                        cmp_output_dir,
                        f"relative_entropy_{cluster_1}_{cluster_2}.csv",
                    ),
                    index=False,
                )

        





  
if __name__ == "__main__":
    main()
