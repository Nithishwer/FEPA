import logging
import os
import pandas as pd
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import (
    load_paths_for_compound,
    load_abfe_paths_for_compound,
    load_paths_for_apo,
)
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from scipy.spatial.distance import jensenshannon
from fepa.utils.dimred_utils import (
    cluster_pca,
    get_ensemble_center,
    make_ensemble_center_df,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("wdir")

    for cmp in config["compounds"][:1]:
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
        features_df = pd.read_csv(
            os.path.join(cmp_output_dir, f"SideChainTorsions_features.csv")
        )

        print(features_df.head())
        offset = 567  # Offset for residue IDs

        # # Get only the aminoacid residues of interest
        residues_of_interest_without_offset = [
            810,
            655,
            625,
            628,
            654,
            658,
            659,
            809,
            806,
            785,
            788,
            802,
            805,
            744,
            740,
            781,
        ]
        residues_of_interest = [resid for resid in residues_of_interest_without_offset]

        # Function to extract residue number
        def get_resid(colname):
            parts = colname.split()
            if len(parts) >= 4 and parts[3].isdigit():
                return int(parts[3]) + offset
            return None

        # Filter columns whose residue number is in resid_list
        filtered_cols = [
            col for col in features_df.columns if get_resid(col) in residues_of_interest
        ] + ["timestep", "ensemble"]

        # Create a new DataFrame with the filtered columns
        filtered_features_df = features_df[filtered_cols]

        # Function to get the simulation type
        def get_sim_type(ensemble):
            if "apo" in ensemble:
                return "apo"
            if "coul" in ensemble or "vdw" in ensemble or "rest" in ensemble:
                match = re.search(r"van_(\d+)", ensemble)
                if match:
                    return f"abfe"
                else:
                    raise ValueError(f"van not found in ensemble: {ensemble}")
            else:
                return "holo"

        def add_offset(colname, offset=0):
            parts = colname.split()
            if len(parts) == 4:
                try:
                    resid = int(parts[3])
                    return f"{parts[0]} {parts[1]} {parts[2]} {resid + offset}"
                except ValueError:
                    pass  # non-numeric residue ID, leave as is
            return colname

        filtered_features_df.columns = [
            add_offset(col, offset=offset) for col in filtered_features_df.columns
        ]

        # Save the filtered DataFrame
        filtered_features_df.to_csv(
            os.path.join(
                cmp_output_dir, f"{cmp}_filtered_SideChainTorsions_features.csv"
            ),
            index=False,
        )

        # Plot the distributions
        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=[
                "apo_1",
                "apo_2",
                "apo_3",
            ],
            output_file=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_histograms_apo.png"
            ),
            ncols=4,
        )

        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=[
                f"{cmp}_van_1_vdw.20",
                f"{cmp}_van_2_vdw.20",
                f"{cmp}_van_3_vdw.20",
            ],
            output_file=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_histograms_vdw20.png"
            ),
            ncols=4,
        )

        # Plot the time evolution of CHI-related variables
        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=[
                "apo_1",
                "apo_2",
                "apo_3",
            ],
            figsize=(20, 15),
            max_cols=4,
            save_path=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_evolution_apo.png"
            ),
        )

        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=[
                f"{cmp}_van_1_vdw.20",
                f"{cmp}_van_2_vdw.20",
                f"{cmp}_van_3_vdw.20",
            ],
            figsize=(20, 15),
            max_cols=4,
            save_path=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_evolution_vdw20.png"
            ),
        )

        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=[
                f"{cmp}_van_1",
                f"{cmp}_van_2",
                f"{cmp}_van_3",
            ],
            figsize=(20, 15),
            max_cols=4,
            save_path=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_evolution_vanilla.png"
            ),
        )

        # Annotate sim_type for features_df
        filtered_features_df["sim_type"] = filtered_features_df["ensemble"].apply(
            get_sim_type
        )

        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=["abfe", "apo"],
            colorby="sim_type",
            output_file=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_histograms_simtype_abfe_apo.png"
            ),
            ncols=4,
        )

        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=["holo", "apo"],
            colorby="sim_type",
            output_file=os.path.join(
                cmp_output_dir, f"{cmp}_sidechain_histograms_simtype_holo_apo.png"
            ),
            ncols=4,
        )


if __name__ == "__main__":
    main()
