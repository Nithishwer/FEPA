import logging
import os
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join(
        "a1_binding_pocket_analysis_apo_holo_reus", "wdir"
    )

    # for cmp in ["10", "17"]:
    for cmp in config["compounds"][:]:
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
            leg_window_list=[f"coul.{i:02}" for i in range(0, 11)]
            + [f"vdw.{i:02}" for i in range(0, 21)]
            + [f"rest.{i:02}" for i in range(0, 12)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)

        # Create workflow
        workflow = binding_pocket_analysis_workflow(cmp, ensemble_handler)

        # Run workflow
        workflow.check_consistency()
        workflow.featurize(cmp_output_dir)
        workflow.reduce_dimensions(cmp_output_dir)

        # Perform clustering
        logging.info("Performing clustering for compound %s...", cmp)
        workflow.cluster_pca(n_clusters=3, n_components=8)
        workflow.plot_pca_cluster(cmp_output_dir)

        # Load data
        workflow.load_results(cmp_output_dir)

        # Plot vanilla
        workflow.plot_holo_pca(cmp_output_dir)

        # Plot apo
        workflow.plot_apo_holo_pca(cmp_output_dir)

        # Plot apo abfe pca
        workflow.plot_apo_abfe_pca(cmp_output_dir)


if __name__ == "__main__":
    main()
