import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound
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
    cumulative_df = pd.DataFrame()

    for cmp in config["compounds"][:]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        avg_df = pd.read_csv(os.path.join(cmp_output_dir, "avg_water_occupancy.csv"))
        cumulative_df = pd.concat([cumulative_df, avg_df], ignore_index=True)
    
    cumulative_df['window'] = cumulative_df['id'].str.split('_').str[1]
    cumulative_df['cmp'] = cumulative_df['id'].str.split('_').str[0]

    # Making apo solvation area
    apo_water_features_df = pd.read_csv(os.path.join(analysis_output_dir, "apo", "WaterOccupancy_features.csv"))
    apo_range_dict = {}
    for apo_run in apo_water_features_df['ensemble'].unique():
        apo_range_dict[apo_run] = np.mean(apo_water_features_df[apo_water_features_df['ensemble'] == apo_run]['occupancy'])

    logging.info("Apo occupancy ranges: %s", apo_range_dict)

    # Plot Time (ps) vs occupancy for each ensemble using matplotlib
    plt.figure(figsize=(12, 8))

    sns.lineplot(
        data=cumulative_df, 
        x='window', 
        y='occupancy', 
        hue='cmp', 
        style='van',
        palette="tab10", 
    )
    
    # Add lines for min and max apo occupancy
    x = cumulative_df['window'].unique()
    # Remove na from x
    x = x[~pd.isna(x)]
    for apo_run, mean_occ in apo_range_dict.items():
        logging.info("Plotting apo occupancy lines for %s: %s", apo_run, mean_occ)
        mean_array = np.full_like(x, mean_occ, dtype=float)
        plt.plot(
            x,
            mean_array,
            linestyle='-.',
            color='gray',
            alpha=0.7,
            label=f"Apo Mean ({apo_run})"
        )

    
    plt.title(f"Average Water Occupancy Across Windows for All compounds", fontsize=16, fontweight='bold')
    plt.xlabel("Window ID", fontsize=14)
    plt.ylabel("Average Number of Waters", fontsize=14)
    plt.legend(
        title='Vanilla Repeat / Apo Ranges', 
        title_fontsize=12, 
        fontsize=10, 
        loc='upper left', 
        bbox_to_anchor=(1.05, 1)
    )
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(f"Water_occupancy_across_windows_all_cmp.png"), 
        dpi=300
    )
    plt.close()


if __name__ == "__main__":
    main()
