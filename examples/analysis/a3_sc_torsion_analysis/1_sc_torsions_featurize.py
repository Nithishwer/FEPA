import logging
import os
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import (
    load_abfe_paths_for_compound,
)
from fepa.core.featurizers import SideChainTorsionsFeaturizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("wdir")

    for cmp in config["compounds"][1:]:
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
            leg_window_list=[f"vdw.{i:02d}" for i in range(20, 21)]
            + [f"coul.{i:02d}" for i in range(0, 11)]
            + [f"rest.{i:02d}" for i in range(0, 12)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()

        ## Featurize
        logging.info("Featurizing binding pocket waters ...")
        sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)

        sct_featurizer.featurize()

        # Save features
        logging.info("Saving features for compound %s ...", cmp)
        sct_featurizer.save_features(cmp_output_dir, overwrite=True)


if __name__ == "__main__":
    main()
