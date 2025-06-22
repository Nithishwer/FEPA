"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
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
from fepa.utils.path_utils import load_abfe_paths_for_compound

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compile_abfeapo_jsd_ddG_df(
    compounds, analysis_output_dir, abfe_summary_df, van, rep, apo, estimator, window
):
    """van for selecting the ABFE value to compare with the experimental value"""
    # Subset compounds to ones in the abfe_summary_df
    compounds_filtered = [
        cmp for cmp in compounds if cmp in abfe_summary_df["Lig_Name"].values
    ]

    # Making a dictionary to store the results
    results = {
        "ddG": [],
        "jsd": [],
        "compound": [],
        "abfe_window": [],
        "van": [],
        "abfe_rep": [],
        "apo": [],
    }

    for cmp in compounds_filtered:
        # Logging
        logging.info(f"Analyzing compound {cmp} ...")
        # Get the ABFE delta ddG
        row = abfe_summary_df[
            (abfe_summary_df["Lig_Name"] == cmp)
            & (abfe_summary_df["Vanilla"] == f"van{van}")
            & (abfe_summary_df["Rep"] == f"rep{rep}")
        ]
        if len(row) != 1:
            raise ValueError(f"Expected 1 row, got {len(row)}")
        # Get the delta delta G
        delta_delta_G = row[estimator].values[0] - row["Experimental_G"].values[0]
        # Get the jsd
        jsd_df = pd.read_csv(
            f"{analysis_output_dir}/{cmp}/{cmp}_entropy_matrix.csv", index_col=0
        )
        jsd = jsd_df.loc[apo, f"{cmp}_van_{van}_{window}"]
        logging.info(f"Compound: {cmp}, ddG: {row['MBAR'].values[0]}, JSD: {jsd}")
        results["ddG"].append(delta_delta_G)
        results["jsd"].append(jsd)
        results["compound"].append(cmp)
        results["abfe_window"].append(window)
        results["van"].append(van)
        results["abfe_rep"].append(rep)
        results["apo"].append(apo)

    # Create a dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"{analysis_output_dir}/jsd_ddG_comparison_van_{van}_rep_{rep}_apo_{apo}.csv",
        index=False,
    )


def plot_abfeapo_jsd_ddG_df(
    analysis_output_dir, van, rep, apo, estimator, show_labels=False
):
    results_df = (
        f"{analysis_output_dir}/jsd_ddG_comparison_van_{van}_rep_{rep}_apo_{apo}.csv"
    )
    results_df = pd.read_csv(results_df)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(data=results_df, x="jsd", y="ddG")

    if show_labels:
        for line in range(0, results_df.shape[0]):
            scatter_plot.text(
                results_df.jsd[line],
                results_df.ddG[line],
                results_df.compound[line],
                horizontalalignment="left",
                size="medium",
                color="black",
                weight="semibold",
            )

    # Fit a straight line to the data
    X = results_df["jsd"].values.reshape(-1, 1)
    y = results_df["ddG"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Calculate fitting parameters
    r2 = r2_score(y, y_pred)
    pearson_corr, _ = pearsonr(results_df["jsd"], results_df["ddG"])

    # Plot the regression line
    plt.plot(results_df["jsd"], y_pred, color="red", linewidth=2)

    # Display fitting parameters
    plt.text(
        0.05,
        0.95,
        f"R²: {r2:.2f}\nPearson: {pearson_corr:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.xlabel("JSD")
    plt.ylabel("ddG (ABFE - Experimental)")
    plt.title(f"JSD vs ddG for van{van} rep{rep} {apo}")
    plt.savefig(
        f"{analysis_output_dir}/jsd_ddG_comparison_van_{van}_rep_{rep}_apo_{apo}.png",
        bbox_inches="tight",
    )


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(
        parent_output_dir, "analysis_p3_comparing_abfe_windows"
    )

    # Load the ABFE summary csv
    abfe_summary_csv = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/exp_v_abfe_df_van_reps_all_hrex_longer_van3.csv"
    abfe_summary_df = pd.read_csv(abfe_summary_csv)
    print(type(abfe_summary_df["Lig_Name"].values[0]))
    abfe_summary_df["Lig_Name"] = abfe_summary_df["Lig_Name"].astype(str)
    print(type(abfe_summary_df["Lig_Name"].values[0]))

    compounds = config["compounds"][:]

    for van in [1, 2, 3]:
        for rep in [1]:
            for apo in ["apo_1", "apo_2", "apo_3"]:
                compile_abfeapo_jsd_ddG_df(
                    compounds=compounds,
                    analysis_output_dir=analysis_output_dir,
                    abfe_summary_df=abfe_summary_df,
                    van=van,
                    rep=rep,
                    apo=apo,
                    estimator="MBAR",
                    window="vdw.20",
                )
                plot_abfeapo_jsd_ddG_df(
                    analysis_output_dir=analysis_output_dir,
                    van=van,
                    rep=rep,
                    apo=apo,
                    estimator="vdw.20",
                    show_labels=True,
                )

    # compile combined
    combined_results = []
    for van in [1, 2, 3]:
        for rep in [1]:
            for apo in ["apo_1", "apo_2", "apo_3"]:
                results_df_path = f"{analysis_output_dir}/jsd_ddG_comparison_van_{van}_rep_{rep}_apo_{apo}.csv"
                if os.path.exists(results_df_path):
                    df = pd.read_csv(results_df_path)
                    combined_results.append(df)
                else:
                    logging.warning(f"File not found: {results_df_path}")

    if combined_results:
        final_results_df = pd.concat(combined_results, ignore_index=True)
        final_results_df.to_csv(
            f"{analysis_output_dir}/combined_jsd_ddG_comparison.csv", index=False
        )
        logging.info("Combined results saved to combined_jsd_ddG_comparison.csv")
    else:
        logging.warning("No results to combine.")

    # Plot combined
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(data=final_results_df, x="jsd", y="ddG", hue="apo")

    # Fit a straight line to the data for each unique value of apo
    unique_apos = final_results_df["apo"].unique()
    for apo in unique_apos:
        apo_data = final_results_df[final_results_df["apo"] == apo]
        X = apo_data["jsd"].values.reshape(-1, 1)
        y = apo_data["ddG"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        # Plot the regression line for this apo
        plt.plot(
            apo_data["jsd"],
            y_pred,
            label=f"{apo} (R²: {r2_score(y, y_pred):.2f})",
            linewidth=2,
        )

    # Display legend for the regression lines
    plt.legend(title="Apo States")

    plt.xlabel("JSD")
    plt.ylabel("ddG (ABFE - Experimental)")
    plt.title(f"JSD vs ddG for van{van} rep{rep} {apo}")
    plt.savefig(
        f"{analysis_output_dir}/jsd_ddG_comparison_combined.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
