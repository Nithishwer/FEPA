import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_apo
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import BPWaterFeaturizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a3_compare_bp_waters", "wdir")

    for cmp in ['apo']:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        # os.makedirs(cmp_output_dir, exist_ok=True)

        # # Prepare paths
        # logging.info("Loading paths for compound %s...", cmp)
        # path_dict = load_paths_for_apo(
        #     config,
        #     bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket 
        #     )

        # print(path_dict)

        # # Load trajectories
        # logging.info("Loading trajectories for compound %s ...", cmp)
        # ensemble_handler = EnsembleHandler(path_dict)
        # ensemble_handler.make_universes()

        # # Featurize
        # logging.info("Featurizing binding pocket waters ...")
        # bp_water_featurizer = BPWaterFeaturizer(
        #     ensemble_handler=ensemble_handler)

        # bp_water_featurizer.featurize(radius = 10)

        # # Save features
        # logging.info("Saving features for compound %s ...", cmp)
        # bp_water_featurizer.save_features(cmp_output_dir, overwrite=True)

        features_df = pd.read_csv(os.path.join(cmp_output_dir, "WaterOccupancy_features.csv"))

        for apo in [1, 2, 3]:
            apo_features_df = features_df[features_df['ensemble'].str.contains(f'apo_{apo}')]
            # Plot Time (ps) vs occupancy for each ensemble using matplotlib
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=apo_features_df, x='Time (ps)', y='occupancy', hue='ensemble')
            plt.title(f"Water Occupancy for {cmp}")
            plt.xlabel("Time (ps)")
            plt.ylabel("Number of Waters")
            plt.legend(
                title='Ensemble', 
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                ncol=2
            )
            plt.tight_layout()
            plt.savefig(os.path.join(cmp_output_dir, f"{cmp}_water_occupancy_apo{apo}_timeseries_v2.png"))

        # match apo_? to the items in the coulmn ensemble
        features_df['apo'] = features_df['ensemble'].str.extract(r'apo_(\d)')
        # Remove the 'apo_' from the ensemble column
        features_df['id'] = features_df['ensemble'].str.replace(r'_apo_\d+', '', regex=True)
        # Group by ensemble and calculate the average occupancy
        avg_df = features_df.groupby(['id', 'apo'], as_index=False)['occupancy'].mean()
        # Plot Time (ps) vs occupancy for each ensemble using matplotlib
        plt.figure(figsize=(12, 8))

        sns.lineplot(
            data=avg_df, 
            x='id', 
            y='occupancy', 
            hue='apo', 
            palette="tab10", 
        )
        plt.title(f"Average Water Occupancy Across Windows for {cmp}", fontsize=16, fontweight='bold')
        plt.xlabel("Window ID", fontsize=14)
        plt.ylabel("Average Number of Waters", fontsize=14)
        plt.legend(title='Vanilla Repeat', title_fontsize=12, fontsize=10, loc='upper right')
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(cmp_output_dir, f"{cmp}_water_occupancy_across_windows_v2.png"), 
            dpi=300
        )
        plt.close()



if __name__ == "__main__":
    main()
