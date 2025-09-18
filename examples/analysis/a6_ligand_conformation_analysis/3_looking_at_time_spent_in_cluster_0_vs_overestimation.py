"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound, load_paths_for_compound
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
import pandas as pd
from fepa.flows.torsions_flow import torsions_analysis_workflow
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
    plot_pca_components,
)
import mdaencore as encore
from fepa.core.dim_reducers import PCADimReducer, UMAPDimReducer
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.analyzers import compute_relative_entropy
from fepa.utils.feature_utils import (
    convert_features_df_w_components_to_angles,
    convert_features_df_w_angles_to_w_components,
)
import seaborn as sns
import re
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = "./wdir"
    analysis_output_dir = parent_output_dir
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Creating van_list and leg_window_list
    leg_window_list = [f"coul.{i:02}" for i in range(0, 11)]+ [f"vdw.{i:02}" for i in range(0, 12)]+ [f"rest.{i:02}" for i in range(0, 11)]

    # Load the exp vs abfe df
    exp_v_abfe_df = pd.read_csv(
        '/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v2_solvated/exp_v_abfe_df_van_reps_all_hrex.csv'
    )

    # --- Step 1: Add overestimation column ---
    exp_v_abfe_df["overestimation"] = exp_v_abfe_df["MBAR"] - exp_v_abfe_df["Experimental_G"]

    # Extract numeric part of van (e.g. 'van1' -> 1)
    exp_v_abfe_df["van"] = exp_v_abfe_df["Vanilla"]

    # Rename Lig_Name -> cmp for clarity
    exp_v_abfe_df = exp_v_abfe_df.rename(columns={"Lig_Name": "cmp"})

    # Make combined df_list
    combined_df_list = []


    for cmp in config["compounds"]:
        # Create cmp output path
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)

        # Load paths
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp=cmp,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            van_list=[1, 2, 3],
            leg_window_list=leg_window_list,
            apo=False,
        )

        logging.info("Path dict: %s", path_dict)

        # Load trajectories
        ensemble_handler = EnsembleHandler(path_dict)
        cluster_df = pd.read_csv(
            os.path.join(cmp_output_dir, f"{cmp}_conformation_cluster_df.csv")
        )

        # --- Extract van count as separate column ---
        cluster_df["van"] = 'van'+cluster_df["ensemble"].str.extract(r"van_(\d+)")
        cluster_df["cmp"] = cluster_df["ensemble"].str.extract(r"^(\d+)_")[0].astype(int)

        # --- Step 3: Merge on cmp and van ---
        merged_df = cluster_df.merge(
            exp_v_abfe_df[["cmp", "van", "overestimation"]],
            on=["cmp", "van"],
            how="left"
        )

        # --- Assign sim_type ---
        merged_df["sim_type"] = merged_df["ensemble"].apply(
            lambda x: "abfe" if any(sub in x for sub in ["coul", "vdw", "rest"]) else "holo"
        )

        # --- Keep only abfe rows ---
        abfe_df = merged_df[merged_df["sim_type"] == "abfe"]

        # Add to combined df list
        combined_df_list.append(abfe_df)         

    # Concatenate all dataframes in the list
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    # Save combined df to csv
    combined_df.to_csv(os.path.join(analysis_output_dir, "combined_abfe_df.csv"), index=False)

    # Make a new df where For each cmp, van combo, make a 
    result = (
    combined_df.groupby(["cmp", "van"])
    .agg(
        pct_cluster0=("cluster", lambda x: (x == 0).mean() * 100),
        overestimation=("overestimation", "mean")
    )
    .reset_index()
)
    # --- Scatter plot ---
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=result,
        x="pct_cluster0",
        y="overestimation",
        hue="van",
        s=100
    )

    # Add labels for cmp
    for _, row in result.iterrows():
        plt.text(
            row["pct_cluster0"]+0.5,   # small offset so it doesn’t overlap
            row["overestimation"],
            str(row["cmp"]),
            fontsize=9
        )

    # Fit and plot regression line

    x = result["pct_cluster0"]
    y = result["overestimation"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color="black", linestyle="--", label=f"Fit: $R^2$={r_value**2:.2f}")

    plt.xlabel("% rows with cluster=0")
    plt.ylabel("Mean overestimation")
    plt.title("Overestimation vs Cluster 0 occupancy")
    plt.legend(title="van")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, "overestimation_vs_cluster0.png"), dpi=300)


if __name__ == "__main__":
    main()
