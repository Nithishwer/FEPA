import logging
import os
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the test config."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(config)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out

def main():
    """Main function to run the analysis"""

    # FEPA repo root: examples/analysis/a1_binding_pocket_analysis -> examples/analysis -> examples -> FEPA
    repo_root = Path(__file__).resolve().parents[3]

    # Load config (kept relative in the repo)
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    config = load_config(str(cfg_path))

    # Make path templates absolute w.r.t. repo root (no change to the JSON file itself)
    config = _abspath_templates(config, repo_root)

    # Where to write expected outputs
    analysis_output_dir = (repo_root / "tests" / "test_data" / "1_expected").resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)


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
            van_list=[1],
            leg_window_list = [f"coul.{i:02}" for i in range(2)],  # coul.00, coul.01
            # [f"vdw.{i:02}" for i in range(0, 21)]
            # [f"rest.{i:02}" for i in range(0, 12)]
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

        # In the next script, we will do the following:

        # Get top n features

        # reduce top n features

        # Cluster PCA of top n features

        # Plot top n cluster PCA

        # Get centroid of the abfe windows

        # Get centroid of the run

        # Dump both centroid gro files

        #

        # # Plot PCA
        # workflow.plot_pca(
        #     cmp_output_dir,
        #     plot_type="scatter"

        # # Get ensemble center
        # # dump gro
        # # make memento for gro
        # # make equil


if __name__ == "__main__":
    main()
