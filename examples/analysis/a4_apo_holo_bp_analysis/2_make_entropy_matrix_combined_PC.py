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

from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.utils.feature_utils import filter_top_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    feature_df_path_template = "/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a1_binding_pocket_analysis/a1_binding_pocket_analysis/wdir/{cmp}/SelfDistance_features.csv"
    out_dir = os.path.join("wdir", "combined_PC_analysis")

    feature_dfs = []

    for cmp in config["compounds"][:]:
        logging.info("Combining feature df for %s ...", cmp)
        feature_dfs.append(
            pd.read_csv(
                feature_df_path_template.format(cmp=cmp),
            )
        )
    logging.info("Concatenating feature dfs")
    # Concatenate the feature dataframes
    feature_df = pd.concat(feature_dfs, ignore_index=True)
    # Remove all duplicate rows
    feature_df = feature_df.drop_duplicates()
    # Print unique values in the "ensemble" column
    unique_ensembles = feature_df["ensemble"].unique()
    logging.info("Unique ensembles: %s", unique_ensembles)
    # Saving
    logging.info("Saving combined feature df")
    feature_df.to_csv(out_dir + "/combined_features.csv", index=False)

    # Dimensionality Reduction
    dimreducer = PCADimReducer(feature_df, n_components=8)
    dimreducer.reduce_dimensions()
    dimreducer.calculate_projections()
    dimreducer.save_projection_df(
        save_path=os.path.join(out_dir, "pca_projection_df.csv")
    )
    # Projection df without nvt:
    projection_df = dimreducer.get_pca_projection_df()
    # Prepare entropy heatmap
    projection_df = projection_df[~projection_df["ensemble"].str.contains("nvt")]

    # subset_projection_df to only include the rows with ensemble values apo and vdw.20
    projection_df = projection_df[
        projection_df["ensemble"].str.contains("apo")
        | projection_df["ensemble"].str.contains("vdw.20")
    ]

    # Making entropy heatmaps for each compound
    # Load the paths for the compound
    plot_entropy_heatmaps(
        cmp="combined",
        entropy_metric="jsd",
        columns_to_consider=["PC1", "PC2"],
        projection_df=projection_df,
        output_dir=out_dir,
    )

    # Visualization
    projection_df = dimreducer.get_pca_projection_df()
    # remove rows with ensemble containing nvt
    projection_df = projection_df[~projection_df["ensemble"].str.contains("apo")]
    dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")
    dimred_visualizer.plot_dimred_sims(
        save_path=os.path.join(out_dir, "pca_components_ensemble_noapo.png"),
    )
    dimred_visualizer.plot_dimred_time(
        save_path=os.path.join(out_dir, "pca_components_time_noapo.png")
    )
    # Visualization with apo
    projection_df = dimreducer.get_pca_projection_df()
    dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")
    dimred_visualizer.plot_dimred_sims(
        save_path=os.path.join(out_dir, "pca_components_ensemble.png"),
    )
    dimred_visualizer.plot_dimred_time(
        save_path=os.path.join(out_dir, "pca_components_time.png")
    )
    plot_eigenvalues(
        pca_object=dimreducer.get_pca(),
        n_components=8,
        save_path=os.path.join(out_dir, "eigenvalues.png"),
    )


if __name__ == "__main__":
    main()
