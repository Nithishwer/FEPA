import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_apo, load_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a10_sidechain_torsions_analysis", "wdir")
    bp_waters_output_dir = os.path.join("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a3_compare_bp_waters/a3_compare_bp_waters/wdir")

    for cmp in config["compounds"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)

        # path_dict = load_paths_for_apo(
        #     config,
        #     bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket 
        #     )

        path_dict = load_paths_for_compound(
            config,
            cmp,
            bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket
            apo = True
        )


        # Load trajectories
        # logging.info("Loading trajectories for compound %s ...", cmp)
        # ensemble_handler = EnsembleHandler(path_dict)
        # ensemble_handler.make_universes()

        # # Featurize
        # logging.info("Featurizing binding pocket waters ...")
        # sc_tors_featurizer = SideChainTorsionsFeaturizer(
        #     ensemble_handler=ensemble_handler)

        # sc_tors_featurizer.featurize()

        # # Save features
        # logging.info("Saving features for compound %s ...", cmp)
        # sc_tors_featurizer.save_features(cmp_output_dir, overwrite=True)

        # Make new features df for components
        features_df_w_components = pd.read_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features_with_components.csv"))

        # Projection df without nvt:
        projection_df = pd.read_csv(os.path.join(cmp_output_dir, "pca_projection_df.csv"))

        # Occupancy df
        van_occupancy_df = pd.read_csv(os.path.join(bp_waters_output_dir, cmp, "WaterOccupancy_features.csv"))
        apo_occupancy_df = pd.read_csv(os.path.join(bp_waters_output_dir, "apo", "WaterOccupancy_features.csv"))
        # Combine the two occupancy dfs
        occupancy_df = pd.concat([van_occupancy_df, apo_occupancy_df], ignore_index=True)
        # change column name from Time (ps) to timestep
        occupancy_df.rename(columns={"Time (ps)": "timestep"}, inplace=True)

        # Merge the Occupancy df and the projection df on ensemble and timestep
        merged_df = pd.merge(
            projection_df,
            occupancy_df,
            on=["ensemble", "timestep"],
            how="inner",
        )

        for col in merged_df.columns:
            if "PC" in col:
                # Create scatter plot with best fit line
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=merged_df, x=col, y="occupancy", hue="ensemble")
                sns.regplot(
                    data=merged_df,
                    x=col,
                    y="occupancy",
                    scatter=False,
                    color="red",
                    line_kws={"label": "Best Fit Line"},
                )
                plt.title(f"{cmp} {col} vs Water Occupancy")
                plt.xlabel(col)
                plt.ylabel("Water Occupancy")
                plt.legend()
                plt.savefig(os.path.join(cmp_output_dir, f"{cmp}_{col}_vs_WaterOccupancy.png"))
                plt.close()




  
if __name__ == "__main__":
    main()
