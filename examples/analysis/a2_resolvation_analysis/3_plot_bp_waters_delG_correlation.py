import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    exp_vs_abfe_df = pd.read_csv("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/exp_v_abfe_df_van_reps_all_hrex.csv")

    df_dict = {'compound': [], 'occupancy_vdw.20': [], 'van': [], 'exp_delG': [], 'pred_delG': []}

    for cmp in config["compounds"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        features_df = pd.read_csv(os.path.join(cmp_output_dir, "WaterOccupancy_features.csv"))

        # match van_? to the items in the coulmn ensemble
        features_df['van'] = features_df['ensemble'].str.extract(r'van_(\d)')
        # Remove the 'van_' from the ensemble column
        features_df['id'] = features_df['ensemble'].str.replace(r'_van_\d+', '', regex=True)
        # Group by ensemble and calculate the average occupancy
        avg_df = features_df.groupby(['id', 'van'], as_index=False)['occupancy'].mean()
        # Save the average occupancy to a new CSV file
        avg_df.to_csv(os.path.join(cmp_output_dir, "avg_water_occupancy.csv"), index=False)
        # Get onlt the rows with vdw.20 in id column
        vdw20_df = avg_df[avg_df['id'].str.contains('vdw.20')]
        # Append the values in the df to the df_dict
        for idx, row in vdw20_df.iterrows():
            vanilla = 'van' + str(row['van'])
            exp_delG = exp_vs_abfe_df[
                (exp_vs_abfe_df['Lig_Name'] == int(cmp)) & 
                (exp_vs_abfe_df['Vanilla'] == vanilla)
            ]['Experimental_G'].values[0]
            pred_delG = exp_vs_abfe_df[
                (exp_vs_abfe_df['Lig_Name'] == int(cmp)) & 
                (exp_vs_abfe_df['Vanilla'] == vanilla)
            ]['MBAR'].values[0]
            logging.info("Experimental delG: %s", exp_delG)
            logging.info("Predicted delG: %s", pred_delG)
            logging.info("row['van']: %s", row['van'])
            logging.info('cmp: %s', cmp)
            logging.info('occupancy: %s', row['occupancy'])
            df_dict['exp_delG'].append(exp_delG)
            df_dict['pred_delG'].append(pred_delG)            
            df_dict['compound'].append(cmp)
            df_dict['occupancy_vdw.20'].extend([row['occupancy']])
            df_dict['van'].extend(row['van'])
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(df_dict)
    df['Overestimation']= df['pred_delG'] - df['exp_delG']
    df.to_csv(os.path.join(analysis_output_dir, "water_occupancy_vs_delG.csv"), index=False)

    # Plot occupancy vs overestimation as scatter plot
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=df, x='occupancy_vdw.20', y='Overestimation', hue='van', palette="tab10", s=100)
    for i in range(df.shape[0]):
        plt.text(df['occupancy_vdw.20'][i], df['Overestimation'][i], str(df['compound'][i]), fontsize=9, ha='right')
    plt.title("Water Occupancy vs Overestimation of Binding Free Energy")
    plt.xlabel("Water Occupancy")
    plt.ylabel("Overestimation of Binding Free Energy (kcal/mol)")
    plt.legend(title='Vanilla Runs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, "water_occupancy_vs_overestimation.png"))

    # Plot occupancy vs overestimation as scatter plot
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=df, x='occupancy_vdw.20', y='pred_delG', hue='van', palette="tab10", s=100)
    for i in range(df.shape[0]):
        plt.text(df['occupancy_vdw.20'][i], df['pred_delG'][i], str(df['compound'][i]), fontsize=9, ha='right')
    plt.title("Water Occupancy vs pred_delG of Binding Free Energy")
    plt.xlabel("Water Occupancy")
    plt.ylabel("pred_delG (kcal/mol)")
    plt.legend(title='Vanilla Runs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, "water_occupancy_vs_pred_delG.png"))


        



if __name__ == "__main__":
    main()
