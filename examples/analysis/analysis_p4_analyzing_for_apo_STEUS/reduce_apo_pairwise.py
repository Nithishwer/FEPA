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
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(parent_output_dir, "analysis_p4_reduce_all_apo")

    # Prepare paths
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    # Pick all combinations of the keys of path_dict:
    comparison_pairs = [("apo_1", "apo_2"), ("apo_1", "apo_3"), ("apo_2", "apo_3")]

    # Ensemble to cluster
    ensemble_to_cluster = {
        ("apo_1", "apo_2"): (0, 1),
        ("apo_1", "apo_3"): (0, 1),
        ("apo_2", "apo_3"): (0, 1),
    }

    for pair in comparison_pairs[:]:
        # Defina names for the pair:
        pair_name = f"{pair[0]}_{pair[1]}"
        apo_output_dir = os.path.join(analysis_output_dir, pair_name)

        if not os.path.exists(apo_output_dir):
            os.makedirs(apo_output_dir)

        # Load trajectories
        ensemble_handler = EnsembleHandler(
            {pair[0]: path_dict[pair[0]], pair[1]: path_dict[pair[1]]}
        )
        ensemble_handler.make_universes()
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Featurize all
        featurizer = SelfDistanceFeaturizer(ensemble_handler)
        featurizer.featurize()
        # featurizer.save_features(output_dir=apo_output_dir, overwrite=True)

        # Dimensionality Reduction
        dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
        dimreducer.reduce_dimensions()
        dimreducer.calculate_projections()
        dimreducer.save_projection_df(
            save_path=os.path.join(
                apo_output_dir, f"pca_projection_df_{pair[0]}_{pair[1]}.csv"
            )
        )

        # Projection df without nvt:
        projection_df = dimreducer.get_pca_projection_df()
        plot_entropy_heatmaps(
            cmp="apo",  # Abusing function should make it more general
            entropy_metric="jsd",
            columns_to_consider=["PC1", "PC2"],
            ensemble_handler=ensemble_handler,
            projection_df=projection_df,
            output_dir=apo_output_dir,
        )

        # Projection df clustered
        pca_projection_df_clustered = cluster_pca(
            projection_df, n_clusters=3, n_components=8
        )

        # Visualization

        dimred_visualizer = DimRedVisualizer(
            projection_df=pca_projection_df_clustered, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(apo_output_dir, "pca_components_ensemble.png")
        )
        dimred_visualizer.plot_dimred_cluster(
            save_path=os.path.join(apo_output_dir, "pca_components_clusters.png"),
            cluster_column="cluster",
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(apo_output_dir, "pca_components_time.png")
        )
        plot_eigenvalues(
            pca_object=dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(apo_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=dimreducer.get_pca(),
            feature_df=featurizer.get_feature_df(),
            num=8,
            save_path=os.path.join(apo_output_dir, "pca_components.png"),
        )

        # Ensemble center df
        ensemble_center_df = make_ensemble_center_df(
            pca_projection_df_clustered, key="cluster"
        )

        # Centroid df
        dimred_visualizer.plot_dimred_cluster(
            save_path=os.path.join(
                apo_output_dir, "pca_components_clusters_w_center.png"
            ),
            centroid_df=ensemble_center_df,
            cluster_column="cluster",
        )

        # Save frames as gro from the ensemble center
        for row in ensemble_center_df.iterrows():
            row = row[1]
            ensemble_handler.dump_gro(
                ensemble=row["ensemble"],
                timestep=row["timestep"],
                save_path=os.path.join(
                    apo_output_dir, f"cluster_{int(row['cluster'])}_center.gro"
                ),
            )

        # Transfer cluster labels to feature df
        feature_df = featurizer.get_feature_df()
        feature_df["cluster"] = pca_projection_df_clustered["cluster"]

        # Filter top features
        top_features_df = filter_top_features(
            feature_df=feature_df,
            key="cluster",
            ensemble1=ensemble_to_cluster[pair][0],
            ensemble2=ensemble_to_cluster[pair][1],
            feature_column_keyword="DIST",
            top_n=200,
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

        # Visualize the top features
        dimred_visualizer_top = DimRedVisualizer(
            projection_df=pca_projection_df_clustered_top, data_name="PCA"
        )

        dimred_visualizer_top.plot_dimred_sims(
            save_path=os.path.join(apo_output_dir, "pca_components_top_ensemble.png")
        )
        dimred_visualizer_top.plot_dimred_cluster(
            save_path=os.path.join(apo_output_dir, "pca_components_top_clusters.png"),
            cluster_column="cluster",
        )
        dimred_visualizer_top.plot_dimred_time(
            save_path=os.path.join(apo_output_dir, "pca_components_top_time.png")
        )
        plot_pca_components(
            pca_object=dimreducer_top.get_pca(),
            feature_df=top_features_df,
            num=8,
            save_path=os.path.join(apo_output_dir, "pca_components_top.png"),
        )

        # Get only features df
        top_features = top_features_df.filter(regex="DIST", axis=1).columns

        # Get penters of pca_projection_df_clustered_top
        ensemble_centers_top = make_ensemble_center_df(
            pca_projection_df_clustered_top,
            key="cluster",
        )

        # Save frames as gro from the ensemble center
        for row in ensemble_centers_top.iterrows():
            row = row[1]
            ensemble_handler.dump_gro(
                ensemble=row["ensemble"],
                timestep=row["timestep"],
                save_path=os.path.join(
                    apo_output_dir, f"top_cluster_{int(row['cluster'])}_center.gro"
                ),
            )

        # Plot centers
        dimred_visualizer_top.plot_dimred_cluster(
            save_path=os.path.join(
                apo_output_dir, "pca_components_top_clusters_w_center.png"
            ),
            centroid_df=ensemble_centers_top,
            cluster_column="cluster",
        )

        # Get only features df
        top_features = top_features_df.filter(regex="DIST", axis=1).columns

        write_plumed_file(
            sdf_names=top_features,
            top_features_pca=dimreducer_top.get_pca(),
            save_path=os.path.join(apo_output_dir, "plumed.dat"),
            molinfo_structure="../reference.pdb",  # fix molinfo here
        )

        # Getting the restraint array
        max_pc1_row = pca_projection_df_clustered_top.loc[
            pca_projection_df_clustered_top["PC1"].idxmax()
        ]
        min_pc1_row = pca_projection_df_clustered_top.loc[
            pca_projection_df_clustered_top["PC1"].idxmin()
        ]
        PC1_min = pca_projection_df_clustered_top["PC1"].min()
        PC1_max = pca_projection_df_clustered_top["PC1"].max()
        # If min PC1 is from ensemble 1, then the restraint array should be from PC1_min to PC1_max
        if min_pc1_row["ensemble"] == pair[0] and max_pc1_row["ensemble"] == pair[1]:
            restraint_array = np.linspace(PC1_min, PC1_max, 24)
        elif min_pc1_row["ensemble"] == pair[1] and max_pc1_row["ensemble"] == pair[0]:
            restraint_array = np.linspace(PC1_max, PC1_min, 24)
        else:
            raise ValueError("The min and max PC1 rows are not from the two ensembles")

        # Write the restraint array to the plumed file
        write_plumed_restraints(
            plumed_file=os.path.join(apo_output_dir, "plumed.dat"),
            restraint_centers=restraint_array,
            kappa=5,
        )


# PErform PCA of those top features and see if you casn differentiate ensemles with the PCA

# Write the PCA components as projection vector and define the feature vector with top features in plumed data file


if __name__ == "__main__":
    main()
