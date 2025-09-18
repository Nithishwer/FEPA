"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
import numpy as np
import pandas as pd

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.core.analyzers import (
    compute_relative_entropy,
)
from fepa.core.visualizers import (
    compute_histograms,
    plot_jsd_histograms,
)
from fepa.core.analyzers import (
    compute_relative_entropy,
)
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
from fepa.utils.dimred_utils import (
    cluster_pca,
    get_ensemble_center,
    make_ensemble_center_df,
)
from fepa.utils.plumed_utils import (
    write_plumed_file,
    write_plumed_restraints,
    make_restraint_array_from_ensemble_centers,
)

from fepa.utils.path_utils import load_paths_for_apo
from fepa.utils.feature_utils import filter_top_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = "wdir"
    analysis_output_dir = os.path.join(parent_output_dir)
    cmp = "6"  # Using a random compounds features df to speed up the process
    existing_analysis_dir = f"/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a1_binding_pocket_analysis/a1_binding_pocket_analysis/wdir/{cmp}"

    # Prepare paths
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    # Pick all combinations of the keys of path_dict:
    comparison_pairs = [("apo_1", "apo_2"), ("apo_2", "apo_3")]

    # Ensemble to cluster
    ensemble_to_cluster = {
        ("apo_1", "apo_2"): (
            0,
            1,
        ),  # The cluster in the basic PCA to pick the top n features for
        ("apo_2", "apo_3"): (2, 1),
    }

    for pair in comparison_pairs[:]:
        # Defina names for the pair:
        pair_name = f"{pair[0]}_{pair[1]}"
        apo_output_dir = os.path.join(analysis_output_dir, pair_name)

        if not os.path.exists(apo_output_dir):
            os.makedirs(apo_output_dir)

        # # Load trajectories
        # ensemble_handler = EnsembleHandler(
        #     {pair[0]: path_dict[pair[0]], pair[1]: path_dict[pair[1]]}
        # )
        # ensemble_handler.make_universes()
        # check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # # Featurize all
        # featurizer = SelfDistanceFeaturizer(ensemble_handler)
        # # featurizer.featurize()
        # # featurizer.save_features(output_dir=apo_output_dir, overwrite=True)
        # featurizer.load_features(input_dir=existing_analysis_dir)
        # # Remove all rows with "cmp_name" in the ensemble column
        # featurizer.feature_df = featurizer.feature_df[
        #     np.isin(featurizer.feature_df["ensemble"], [pair[0], pair[1]])
        # ]
        # featurizer.save_features(
        #     output_dir=apo_output_dir, overwrite=True
        # )  # Save the features df for the pair

        # # Dimensionality Reduction
        # dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
        # dimreducer.reduce_dimensions()
        # dimreducer.calculate_projections()
        # dimreducer.save_projection_df(
        #     save_path=os.path.join(
        #         apo_output_dir, f"pca_projection_df_{pair[0]}_{pair[1]}.csv"
        #     )
        # )

        # # Projection df without nvt:
        # projection_df = dimreducer.get_pca_projection_df()
        # plot_entropy_heatmaps(
        #     cmp="apo",  # Abusing function should make it more general
        #     entropy_metric="jsd",
        #     columns_to_consider=["PC1", "PC2"],
        #     ensemble_handler=ensemble_handler,
        #     projection_df=projection_df,
        #     output_dir=apo_output_dir,
        # )

        # # Projection df clustered
        # pca_projection_df_clustered = cluster_pca(
        #     projection_df, n_clusters=3, n_components=8
        # )

        # # Visualization
        # dimred_visualizer = DimRedVisualizer(
        #     projection_df=pca_projection_df_clustered, data_name="PCA"
        # )
        # dimred_visualizer.plot_dimred_sims(
        #     save_path=os.path.join(apo_output_dir, "pca_components_ensemble.png")
        # )
        # dimred_visualizer.plot_dimred_cluster(
        #     save_path=os.path.join(apo_output_dir, "pca_components_clusters.png"),
        #     cluster_column="cluster",
        # )
        # dimred_visualizer.plot_dimred_time(
        #     save_path=os.path.join(apo_output_dir, "pca_components_time.png")
        # )
        # plot_eigenvalues(
        #     pca_object=dimreducer.get_pca(),
        #     n_components=8,
        #     save_path=os.path.join(apo_output_dir, "eigenvalues.png"),
        # )
        # plot_pca_components(
        #     pca_object=dimreducer.get_pca(),
        #     feature_df=featurizer.get_feature_df(),
        #     num=8,
        #     save_path=os.path.join(apo_output_dir, "pca_components.png"),
        # )

        # # Ensemble center df
        # ensemble_center_df = make_ensemble_center_df(
        #     pca_projection_df_clustered, key="cluster"
        # )

        # # Centroid df
        # dimred_visualizer.plot_dimred_cluster(
        #     save_path=os.path.join(
        #         apo_output_dir, "pca_components_clusters_w_center.png"
        #     ),
        #     centroid_df=ensemble_center_df,
        #     cluster_column="cluster",
        # )

        # # Save frames as gro from the ensemble center
        # for row in ensemble_center_df.iterrows():
        #     row = row[1]
        #     ensemble_handler.dump_gro(
        #         ensemble=row["ensemble"],
        #         timestep=row["timestep"],
        #         save_path=os.path.join(
        #             apo_output_dir, f"cluster_{int(row['cluster'])}_center.gro"
        #         ),
        #     )

        # # Transfer cluster labels to feature df
        # feature_df = featurizer.get_feature_df()
        # feature_df["cluster"] = pca_projection_df_clustered["cluster"]

        # # Filter top features
        # top_features_df = filter_top_features(
        #     feature_df=feature_df,
        #     key="cluster",
        #     ensemble1=ensemble_to_cluster[pair][0],
        #     ensemble2=ensemble_to_cluster[pair][1],
        #     feature_column_keyword="DIST",
        #     top_n=200,
        # )
        # top_features_df[:100].to_csv(
        #     os.path.join(
        #         apo_output_dir, f"top_features_{pair[0]}_{pair[1]}_sample.csv"
        #     ),
        #     index=False,
        # )
        top_features_df = pd.read_csv(
            os.path.join(apo_output_dir, f"top_features_{pair[0]}_{pair[1]}.csv")
        )

        # Do PCA of top df
        dimreducer_top = PCADimReducer(top_features_df, n_components=8)
        dimreducer_top.reduce_dimensions()
        dimreducer_top.calculate_projections()
        dimreducer_top.save_projection_df(
            save_path=os.path.join(
                apo_output_dir, f"pca_projection_df_top_{pair[0]}_{pair[1]}.csv"
            )
        )
        dimreducer_top.save_pca(
            save_path=os.path.join(apo_output_dir, f"pca_top_{pair[0]}_{pair[1]}.pkl")
        )

        # Cluster the top features
        pca_projection_df_clustered_top = cluster_pca(
            dimreducer_top.get_pca_projection_df(), n_clusters=3, n_components=8
        )

        # Assert ensemble column and timestep column same in top_features_df and pca_projection_df_clustered_top
        assert top_features_df["ensemble"].equals(
            pca_projection_df_clustered_top["ensemble"]
        ), "Mismatch in 'ensemble' column"
        assert top_features_df["timestep"].equals(
            pca_projection_df_clustered_top["timestep"]
        ), "Mismatch in 'timestep' column"

        # Transfer cluster labels to top features df
        top_features_df["cluster"] = pca_projection_df_clustered_top["cluster"]

        # Plot jsd
        relative_entropy_dict = compute_relative_entropy(
            feature_df=top_features_df,
            ensemble1=ensemble_to_cluster[pair][0],
            ensemble2=ensemble_to_cluster[pair][1],
            num_bins=20,
            key="cluster",
            feature_column_keyword="DIST",
        )
        histograms = compute_histograms(
            feature_df=top_features_df,
            key="cluster",
            value_1=ensemble_to_cluster[pair][0],
            value_2=ensemble_to_cluster[pair][1],
            num_bins=20,
            feature_column_keyword="DIST",
        )
        plot_jsd_histograms(
            histograms=histograms,
            relative_entropy_dict=relative_entropy_dict,
            top_n=16,
            save_path=os.path.join(apo_output_dir, "jsd_histograms.png"),
        )


# PErform PCA of those top features and see if you casn differentiate ensemles with the PCA

# Write the PCA components as projection vector and define the feature vector with top features in plumed data file


if __name__ == "__main__":
    main()
