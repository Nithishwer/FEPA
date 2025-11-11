"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)

from fepa.utils.path_utils import load_paths_for_apo


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("wdir")
    cmp = '6' # USing a random compounds features df to speed up the process
    existing_analysis_dir = f'/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a1_binding_pocket_analysis/a1_binding_pocket_analysis/wdir/{cmp}'

    # Create output directory if it doesn't exist
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Prepare paths
    path_dict = load_paths_for_apo(
        config,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    # Load trajectories
    ensemble_handler = EnsembleHandler(path_dict)
    ensemble_handler.make_universes()
    check_bp_residue_consistency(ensemble_handler.get_universe_dict())

    # Featurize
    featurizer = SelfDistanceFeaturizer(ensemble_handler)
    # featurizer.featurize()
    # featurizer.save_features(output_dir=analysis_output_dir, overwrite=True)
    featurizer.load_features(input_dir=existing_analysis_dir)
    # Remove all rows with "cmp_name" in the ensemble column
    featurizer.feature_df = featurizer.feature_df[
        featurizer.feature_df["ensemble"].str.contains(f'{cmp}') == False
    ] 

    # Dimensionality Reduction
    dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
    dimreducer.reduce_dimensions()
    dimreducer.calculate_projections()
    dimreducer.save_projection_df(
        save_path=os.path.join(analysis_output_dir, "pca_projection_df.csv")
    )

    # Projection df without nvt:
    projection_df = dimreducer.get_pca_projection_df()
    plot_entropy_heatmaps(
        cmp="apo",  # Abusing function should make it more general
        entropy_metric="jsd",
        columns_to_consider=["PC1", "PC2"],
        ensemble_handler=ensemble_handler,
        projection_df=projection_df,
        output_dir=analysis_output_dir,
    )
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
    plot_pca_components(
        pca_object=dimreducer.get_pca(),
        feature_df=featurizer.get_feature_df(),
        num=8,
        save_path=os.path.join(analysis_output_dir, "pca_components.png"),
    )


if __name__ == "__main__":
    main()
