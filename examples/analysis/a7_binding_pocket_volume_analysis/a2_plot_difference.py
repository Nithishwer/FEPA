"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import BindingPocketVolumeFeaturizer

# from fepa.features.featurizes import BindingPocketVolumeFeaturizer
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
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("wdir")

    # Initialize a list to collect results
    results = []

    for cmp in config["compounds"][1:]:
        # Get the volume data
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        cmp_vol_avg_df = pd.read_csv(os.path.join(cmp_output_dir, "BindingPocketVolume_features_avg.csv"))

        # Subset only the rows where window has vdw in it
        cmp_vol_avg_df = cmp_vol_avg_df[cmp_vol_avg_df['window'].str.contains('vdw')]
        # Get all unique simulation ids
        cmp_vol_avg_df['simulation_id'] = cmp_vol_avg_df['ensemble'].apply(lambda x: '_'.join(x.split('_')[:3]))

        for sim_id in cmp_vol_avg_df['simulation_id'].unique():
            sim_cmp_vol_avg_df = cmp_vol_avg_df[cmp_vol_avg_df['simulation_id'] == sim_id]

            # Get vdw.00 and vdw.20 volumes
            try:
                vol_00 = sim_cmp_vol_avg_df[sim_cmp_vol_avg_df['window'] == 'vdw.00']['pocket_volume'].values[0]
                vol_20 = sim_cmp_vol_avg_df[sim_cmp_vol_avg_df['window'] == 'vdw.20']['pocket_volume'].values[0]
                vol_diff = vol_20 - vol_00
            except IndexError:
                # In case vdw.00 or vdw.20 is missing
                vol_diff = None

            results.append({
                'simulation_id': sim_id,
                'vol_diff': vol_diff
            })

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    result_df['van'] = result_df['simulation_id'].apply(lambda x: '_'.join(x.split('_')[1:3]))
    result_df['cmp'] = result_df['simulation_id'].apply(lambda x: x.split('_')[0])

    # Save to CSV if needed
    result_df.to_csv("simulation_volume_differences.csv", index=False)


    # Load the results CSV or use the dataframe directly
    results_df = pd.read_csv("simulation_volume_differences.csv")

    # Set plot style
    sns.set(style="whitegrid")

    # Create bar plot
    plt.figure(figsize=(14,6))
    bar_plot = sns.barplot(
        data=results_df,
        x="cmp",         # comparison number on x-axis
        y="vol_diff",    # volume difference on y-axis
        hue="van",       # color by van
        dodge=True
    )

    # Add labels and title
    bar_plot.set_xlabel("Cmp", fontsize=12)
    bar_plot.set_ylabel("Volume Difference (vdw.20 - vdw.00)", fontsize=12)
    bar_plot.set_title("Volume Differences by Simulation", fontsize=14)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    # Add legend outside the plot
    plt.legend(title="Van", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("volume_differences_barplot.png", dpi=300)



if __name__ == "__main__":
    main()
