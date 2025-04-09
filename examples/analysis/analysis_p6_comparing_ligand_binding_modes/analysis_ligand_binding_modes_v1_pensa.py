"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import BATFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
)
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fepa.utils.stat_utils import (
    find_optimal_clusters_elbow,
    find_optimal_clusters_silhouette,
    find_optimal_clusters_bic,
)
from fepa.utils.path_utils import load_abfe_paths_for_compound, load_paths_for_compound
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
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(
        parent_output_dir, "analysis_p6_comparing_ligand_binding_modes"
    )
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Creating van_list and leg_window_list
    van_list = [i for i in range(1, 4)]
    leg_window_list = (
        [f"coul.{i:02}" for i in range(0, 11)]
        + [f"vdw.{i:02}" for i in range(0, 12)]
        + [f"rest.{i:02}" for i in range(0, 11)]
    )

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

    # Featurize
    featurizer = BATFeaturizer(ensemble_handler, sel="resname unk")
    featurizer.featurize()
    featurizer.save_features(output_dir=analysis_output_dir, overwrite=True)

    # Dimensionality Reduction
    feature_df = featurizer.get_feature_df()
    dimreducer = PCADimReducer(feature_df, n_components=8)
    dimreducer.reduce_dimensions(feature_column_keyword="BAT")
    dimreducer.calculate_projections()
    dimreducer.save_projection_df(
        save_path=os.path.join(analysis_output_dir, "pca_projection_df.csv")
    )

    # Projection df
    projection_df = dimreducer.get_pca_projection_df()

    # Visualization
    projection_df = dimreducer.get_pca_projection_df()
    dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")
    dimred_visualizer.plot_dimred_sims(
        save_path=os.path.join(analysis_output_dir, "pca_components_ensemble.png")
    )
    dimred_visualizer.plot_dimred_time(
        save_path=os.path.join(analysis_output_dir, "pca_components_time.png")
    )
    plot_eigenvalues(
        pca_object=dimreducer.get_pca(),
        n_components=8,
        save_path=os.path.join(analysis_output_dir, "eigenvalues.png"),
    )
    # plot_pca_components(
    #     pca_object=dimreducer.get_pca(),
    #     feature_df=feature_df,
    #     num=8,
    #     save_path=os.path.join(analysis_output_dir, "pca_components.png"),
    # )

    # Clustering
    no_of_binding_modes = find_optimal_clusters_bic(
        dimreducer.get_pca_projection_df()["PC1"].tolist(),
        save_path=os.path.join(analysis_output_dir, "bic_method.png"),
    )
    logging.info(
        "Number of binding modes as guessed from PC1 bic: %s", no_of_binding_modes
    )

    projection_df_w_clusters = cluster_pca(
        pca_projection_df=projection_df, n_clusters=no_of_binding_modes, n_components=1
    )
    dimred_visualizer = DimRedVisualizer(
        projection_df=projection_df_w_clusters, data_name="PCA"
    )
    dimred_visualizer.plot_dimred_cluster(
        save_path=os.path.join(analysis_output_dir, "pca_components_cluster.png"),
        cluster_column="cluster",
    )
    # projection_df_w_clusters.to_csv(
    #     os.path.join(analysis_output_dir, "pca_projection_df_with_clusters.csv"),
    #     index=False,
    # )

    # JSD calculation
    assert feature_df["ensemble"].equals(projection_df_w_clusters["ensemble"]), (
        "Mismatch in ensemble series between feature_df and projection_df_w_clusters"
    )
    feature_df["cluster"] = projection_df_w_clusters["cluster"]
    clusters = projection_df_w_clusters["cluster"].unique()
    # Loop over all pairs of elements in clusters
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_1 = clusters[i]
            cluster_2 = clusters[j]
            logging.info("Computing JSD for clusters %s and %s", cluster_1, cluster_2)
            relative_entropy_dict = compute_relative_entropy(
                feature_df=feature_df,
                key="cluster",
                ensemble1=cluster_1,
                ensemble2=cluster_2,
                num_bins=50,
                feature_column_keyword="BAT",
            )
            histograms = compute_histograms(
                feature_df=feature_df,
                key="cluster",
                value_1=cluster_1,
                value_2=cluster_2,
                num_bins=50,
                feature_column_keyword="BAT",
            )
            plot_jsd_histograms(
                relative_entropy_dict=relative_entropy_dict,
                histograms=histograms,
                save_path=os.path.join(
                    analysis_output_dir, f"jsd_histograms_{cluster_1}_{cluster_2}.png"
                ),
            )

    # Plotting distribution of simulations across binding modes

    # Annotate the ensembles in dimred_df
    projection_df_w_clusters["sim_type"] = [
        element
        if "van" in element
        else "abfe"
        if any(x in element for x in ["vdw", "coul", "rest"])
        else "unknown"
        for element in projection_df_w_clusters["ensemble"]
    ]
    projection_df_w_clusters.to_csv(
        os.path.join(analysis_output_dir, "pca_projection_df_with_clusters.csv"),
        index=False,
    )

    # Plot the distribution of simulations across binding modesm as bar plot
    print(projection_df_w_clusters)
    # Calculate the percentage of each sim_type within each cluster

    # Calculate the percentage of each cluster within each sim_type
    sim_type_cluster_counts = (
        projection_df_w_clusters.groupby(["sim_type", "cluster"])
        .size()
        .unstack(fill_value=0)
    )
    sim_type_totals = sim_type_cluster_counts.sum(axis=1)
    sim_type_cluster_percentages = (
        sim_type_cluster_counts.div(sim_type_totals, axis=0) * 100
    )

    # Create the stacked bar chart
    plt.figure(figsize=(14, 7), dpi=400)  # Increased DPI for higher resolution
    sim_type_cluster_percentages.plot(kind="bar", stacked=True)

    # Set the title and labels
    plt.title("Percentage of Clusters per Sim Type", fontsize=16)
    plt.xlabel("Sim Type", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Cluster", fontsize=10)
    sns.despine()  # Remove top and right spines for a cleaner look
    plt.tight_layout()
    plt.savefig(
        os.path.join(analysis_output_dir, "percentage_of_clusters_per_sim_type.png")
    )


if __name__ == "__main__":
    main()
