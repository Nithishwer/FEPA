"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
from fepa.flows.torsions_flow import torsions_analysis_workflow
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
    plot_pca_components,
)
from fepa.core.featurizers import TorsionsFeaturizer
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
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(
        parent_output_dir, "analysis_p6_comparing_ligand_binding_modes_w_components"
    )
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Creating van_list and leg_window_list
    van_list = [i for i in range(1, 4)]
    leg_window_list = (
        [f"coul.{i:02}" for i in range(0, 4)]
        # [f"coul.{i:02}" for i in range(0, 11)]
        # + [f"vdw.{i:02}" for i in range(0, 12)]
        # + [f"rest.{i:02}" for i in range(0, 11)]
    )

    cmps_of_interest = ["48951", "47594", "49599", "52542", "47821"]

    # Prepare paths
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp="42922",
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        van_list=van_list,
        leg_window_list=leg_window_list,
        apo=False,
    )
    # path_dict = load_paths_for_compound(
    #     config,
    #     cmp="42922",
    #     bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    #     apo=False,
    # )

    logging.info("Path dict: %s", path_dict)

    # Load trajectories
    ensemble_handler = EnsembleHandler(path_dict)
    ensemble_handler.make_universes()
    check_bp_residue_consistency(ensemble_handler.get_universe_dict())

    # Using the flow
    sel = "resname unk"
    featurizer = TorsionsFeaturizer(ensemble_handler, sel=sel)
    featurizer.featurize()
    featurizer.save_features(output_dir=analysis_output_dir, overwrite=True)
    feature_df = featurizer.get_feature_df()

    # Decompose the torsions into components
    feature_df_components = convert_features_df_w_angles_to_w_components(
        features_df_w_angles=feature_df,
        feature_column_keyword="TORS",
        save_path=os.path.join(analysis_output_dir, "features_df_w_components.csv"),
    )

    # Reduce dimensions
    n_components = 2

    for n_neighbors in [5, 10, 15]:
        for min_dist in [0.1, 0.2, 0.3]:
            dimreducer = UMAPDimReducer(
                feature_df=feature_df_components, n_components=n_components
            )
            dimreducer.reduce_dimensions(
                feature_column_keyword="TORS",
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
            projection_df = dimreducer.get_umap_projection_df()
            # projection_df.to_csv(
            #     os.path.join(analysis_output_dir, "umap_projection_df.csv"), index=False
            # )

            # Visualization
            dimred_visualizer = DimRedVisualizer(
                projection_df=projection_df, data_name="UMAP"
            )
            save_path = os.path.join(
                analysis_output_dir, f"dimred_sims_{n_neighbors}_{min_dist}.png"
            )
            dimred_visualizer.plot_dimred_sims(save_path=save_path)
    # save_path = os.path.join(analysis_output_dir, "dimred_time.png")
    # dimred_visualizer.plot_dimred_time(save_path=save_path)
    # save_path = os.path.join(analysis_output_dir, "dimred_eig.png")
    # plot_eigenvalues(
    #     pca_object=dimreducer.get_pca(),
    #     n_components=n_components,
    #     save_path=save_path,
    # )
    # # Subset only the rows with vanilla ensembles ending with van_{digit}
    # van_feature_df = feature_df_components[
    #     feature_df_components["ensemble"].str.match(r".*van_\d+$", flags=re.IGNORECASE)
    # ]
    # logging.info("Subsetted vanilla ensembles: %s", van_feature_df)
    # plot_pca_components(
    #     pca_object=dimreducer.get_pca(),
    #     feature_df=van_feature_df,
    #     num=8,
    #     save_path=os.path.join(analysis_output_dir, "pca_components_van.png"),
    #     feature_column_keyword="TORS",
    # )

    # # Subset only the rows with vanilla ensembles ending with van_{digit}
    # abfe_feature_df = feature_df_components[
    #     feature_df_components["ensemble"].str.match(
    #         r".*van_\d+_.*$", flags=re.IGNORECASE
    #     )
    # ]
    # logging.info("Subsetted abfe ensembles: %s", abfe_feature_df)
    # plot_pca_components(
    #     pca_object=dimreducer.get_pca(),
    #     feature_df=abfe_feature_df,
    #     num=8,
    #     save_path=os.path.join(analysis_output_dir, "pca_components_abfe.png"),
    #     feature_column_keyword="TORS",
    # )

    # # Cluster PCA
    # n_clusters = 4
    # projection_df = cluster_pca(
    #     pca_projection_df=projection_df,
    #     n_clusters=n_clusters,
    #     n_components=n_components,
    # )
    # dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")
    # assert feature_df["ensemble"].equals(projection_df["ensemble"]), (
    #     "Mismatch in ensemble series between feature_df and projection_df"
    # )
    # feature_df["cluster"] = projection_df["cluster"]

    # # Plot clusters
    # cluster_column = "cluster"
    # save_path = os.path.join(analysis_output_dir, "dimred_cluster.png")
    # dimred_visualizer.plot_dimred_cluster(
    #     save_path=save_path, cluster_column=cluster_column
    # )

    # # Define function to plot JSD histograms
    # def plot_torsions_jsd_histograms(cluster_1, cluster_2, save_path):
    #     relative_entropy_dict = compute_relative_entropy(
    #         feature_df=feature_df,
    #         key="cluster",
    #         ensemble1=cluster_1,
    #         ensemble2=cluster_2,
    #         num_bins=50,
    #         feature_column_keyword="BAT",
    #     )
    #     histograms = compute_histograms(
    #         feature_df=feature_df,
    #         key="cluster",
    #         value_1=cluster_1,
    #         value_2=cluster_2,
    #         num_bins=50,
    #         feature_column_keyword="BAT",
    #     )
    #     plot_jsd_histograms(
    #         relative_entropy_dict=relative_entropy_dict,
    #         histograms=histograms,
    #         save_path=save_path,
    #     )

    # # Function Loop over all pairs of elements and plot JSD histograms
    # output_dir = os.path.join(analysis_output_dir)
    # clusters = projection_df["cluster"].unique()
    # # Loop over all pairs of elements in clusters
    # for i, cluster_1 in enumerate(clusters):
    #     for j, cluster_2 in enumerate(clusters[i + 1 :], start=i + 1):
    #         logging.info("Computing JSD for clusters %s and %s", cluster_1, cluster_2)
    #         plot_torsions_jsd_histograms(
    #             cluster_1,
    #             cluster_2,
    #             save_path=os.path.join(
    #                 output_dir, f"jsd_histograms_{cluster_1}_{cluster_2}.png"
    #             ),
    #         )

    # # Annotate ensembles and abfe, van , apo
    # projection_df["sim_type"] = [
    #     "abfe"
    #     if any(x in element for x in ["vdw", "coul", "rest"])
    #     else "nvt"
    #     if "nvt" in element
    #     else "van"
    #     for element in projection_df["ensemble"]
    # ]
    # projection_df.to_csv(
    #     save_path,
    #     index=False,
    # )

    # # Plot cluster distribution
    # groupby = ["sim_type", "cluster"]
    # save_path = os.path.join(analysis_output_dir, "cluster_distribution_sim_type.png")

    # sim_type_cluster_counts = (
    #     projection_df.groupby(groupby).size().unstack(fill_value=0)
    # )
    # sim_type_totals = sim_type_cluster_counts.sum(axis=1)
    # sim_type_cluster_percentages = (
    #     sim_type_cluster_counts.div(sim_type_totals, axis=0) * 100
    # )
    # # Create the stacked bar chart
    # plt.figure(figsize=(14, 7), dpi=300)  # Increased DPI for higher resolution
    # sim_type_cluster_percentages.plot(kind="bar", stacked=True)
    # # Set the title and labels
    # plt.title("Percentage of torsion clusters per Sim Type", fontsize=16)
    # plt.xlabel("Sim Type", fontsize=12)
    # plt.ylabel("Percentage (%)", fontsize=12)
    # plt.xticks(rotation=45, ha="right", fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.legend(title="Torsion Cluster", fontsize=10)
    # sns.despine()  # Remove top and right spines for a cleaner look
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    main()
