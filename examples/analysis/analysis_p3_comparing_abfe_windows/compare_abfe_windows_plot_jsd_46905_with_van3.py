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
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound

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
        parent_output_dir, "analysis_p3_comparing_abfe_windows"
    )

    for cmp in ["46905"]:
        # for cmp in config["compounds"][:]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1, 2, 3],
            leg_window_list=[
                "rest.00",
                "rest.05",
                "rest.10",
                "coul.00",
                "coul.05",
                "coul.10",
                "vdw.00",
                "vdw.05",
                "vdw.10",
                "vdw.15",
                "vdw.20",
            ],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        )

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        logging.info("Making universes for compound %s...", cmp)
        ensemble_handler.make_universes()
        logging.info("Checking residue consistency for compound %s...", cmp)
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Featurize
        logging.info("Featurizing compound %s...", cmp)
        featurizer = SelfDistanceFeaturizer(ensemble_handler)
        featurizer.featurize()
        featurizer.save_features(output_dir=cmp_output_dir, overwrite=True)

        # Dimensionality Reduction
        logging.info("Performing dimensionality reduction for compound %s...", cmp)
        dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
        dimreducer.reduce_dimensions()
        dimreducer.calculate_projections()
        dimreducer.save_projection_df(
            save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        )

        # Projection df without nvt:
        projection_df = dimreducer.get_pca_projection_df()
        # Prepare entropy heatmap
        projection_df = projection_df[~projection_df["ensemble"].str.contains("nvt")]
        plot_entropy_heatmaps(
            cmp=cmp,
            entropy_metric="jsd",
            columns_to_consider=["PC1", "PC2"],
            projection_df=projection_df,
            output_dir=cmp_output_dir,
        )

        # Visualization
        projection_df = dimreducer.get_pca_projection_df()
        # remove rows with ensemble containing nvt
        projection_df = projection_df[~projection_df["ensemble"].str.contains("apo")]

        logging.info("Visualizing compound %s...", cmp)
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "pca_components_ensemble_noapo.png"),
            highlights=[f"{cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "pca_components_time_noapo.png")
        )

        # Visualization with apo
        projection_df = dimreducer.get_pca_projection_df()

        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "pca_components_ensemble.png"),
            highlights=[f"{cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "pca_components_time.png")
        )
        plot_eigenvalues(
            pca_object=dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=dimreducer.get_pca(),
            feature_df=featurizer.get_feature_df(),
            num=8,
            save_path=os.path.join(cmp_output_dir, "pca_components.png"),
        )


if __name__ == "__main__":
    main()
