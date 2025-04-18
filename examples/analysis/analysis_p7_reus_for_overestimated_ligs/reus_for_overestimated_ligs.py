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
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(
        parent_output_dir, "analysis_p7_reus_for_overestimated_ligs"
    )

    for cmp in ["43929", "46905"]:
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
                "coul.00",
                "coul.05",
                "coul.10",
                "vdw.00",
                "vdw.10",
                "vdw.15",
                "vdw.20",
                "rest.05",
                "rest.10",
            ],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        print(path_dict)

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
        # workflow.load_results(cmp_output_dir)

        # Plot vanilla
        workflow.plot_holo_pca(cmp_output_dir)

        # Plot apo
        workflow.plot_apo_holo_pca(cmp_output_dir)

        # Plot apo abfe pca
        workflow.plot_apo_abfe_pca(cmp_output_dir)

        # Plot PCA
        # workflow.plot_pca(
        #     cmp_output_dir,
        #     plot_type="scatter"

        # Get ensemble center
        # dump gro
        # make memento for gro
        # make equil


if __name__ == "__main__":
    main()
