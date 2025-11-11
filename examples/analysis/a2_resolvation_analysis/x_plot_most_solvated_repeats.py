import math
import pandas as pd
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr  # Import for Pearson correlation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_metrics(df, x, y):
    """
    Calculate metrics: MUE, RMSE, R2, and Pearson correlation.
    """
    # Extract the columns based on the column names provided
    x_values = df[x]
    y_values = df[y]

    # Calculate Mean Unsigned Error (MUE)
    mue = np.mean(np.abs(x_values - y_values))

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(x_values, y_values))

    # Calculate R squared (R2)
    r2 = r2_score(x_values, y_values)

    # Calculate Pearson Correlation
    pearson_corr, _ = pearsonr(x_values, y_values)

    return mue, rmse, r2, pearson_corr


def average_over_vanilla(
    df, keep=["Lig_Name"], columns_to_average=["MBAR", "BAR", "TI"]
):
    """
    Make a DataFrame by averaging over vanilla and reps,
    and add averaging error columns for specified columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        keep (list): Columns to group by.
        columns_to_average (list): Columns to calculate averaging errors for.

    Returns:
        pd.DataFrame: DataFrame with averaged values and averaging error columns.
    """
    # Compute averaging errors (max-min)/2 for the specified columns
    for col in columns_to_average:
        print("Averaging column:", col)
        error_col = f"{col}_averaging_error"
        df[error_col] = df.groupby(keep)[col].transform(
            lambda x: (x.max() - x.min()) / 2
        )

    # Group by the specified columns and calculate the mean values for each group
    df = df.drop(columns=["Rep", "color", "Vanilla"])
    grouped_df = df.groupby(keep)
    avg_df = grouped_df
    avg_df = grouped_df.agg(
        {
            "Experimental_G": "first",  # Keep 'Experimental_G' as it's constant for each group
            "MBAR": "mean",
            "MBAR_Error": "mean",
            "BAR": "mean",
            "BAR_Error": "mean",
            "TI": "mean",
            "TI_Error": "mean",
            "MBAR_averaging_error": "first",
            "BAR_averaging_error": "first",
            "TI_averaging_error": "first",
            "Convergence": "all",
        }
    ).reset_index()

    return avg_df


def plot_exp_v_abfe_averaged(
    df, x, y, err_col, title, save_name, xlim=None, ylim=None, color_by=None
):
    """
    Plots the points of the DataFrame 'df' using the columns specified by 'x' and 'y', adds error bars using 'err_col',
    adds a reference line y = x, and saves the plot as a PNG file in publication quality.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    x (str): The column name for the x-axis (Experimental values).
    y (str): The column name for the y-axis (Predicted values).
    err_col (str): The column name for the error bars on the y-axis.
    save_name (str): The filename to save the plot as a PNG file.
    """
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    # Calculate metrics
    mue, rmse, r2, pearson_corr = calculate_metrics(df, y, x)
    # print(f"MUE: {mue:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}, Pearson Correlation: {pearson_corr:.2f}")

    # Define color mapping for 'Rep' column
    def get_color(van):
        if van.startswith("van1"):
            return "#1f77b4"  # Blue for 'van1'
        elif van.startswith("van2"):
            return "#ff7f0e"  # Orange for 'van2'
        elif van.startswith("van3"):
            return "#2ca02c"  # Green for 'van3'
        else:
            return "#d62728"  # Default color

    if color_by == "Vanilla":
        df["color"] = df["Vanilla"].apply(
            get_color
        )  # Assign colors based on the Rep column

    # Plot the points with error bars
    for i, row in df.iterrows():
        if color_by == "Vanilla":
            if row["Convergence"]:
                plt.errorbar(
                    row[x],
                    row[y],
                    yerr=row[err_col],
                    fmt="o",
                    alpha=1,
                    ecolor="gray",
                    capsize=3,
                    elinewidth=1,
                    markeredgewidth=0,
                    markeredgecolor="k",
                    markersize=6,
                    markerfacecolor=row["color"],
                )
            else:
                plt.errorbar(
                    row[x],
                    row[y],
                    yerr=row[err_col],
                    fmt="X",
                    alpha=1,
                    ecolor="gray",
                    capsize=3,
                    elinewidth=1,
                    markeredgewidth=0,
                    markeredgecolor="k",
                    markersize=6,
                    markerfacecolor=row["color"],
                )
        else:
            if row["Convergence"]:
                plt.errorbar(
                    row[x],
                    row[y],
                    yerr=row[err_col],
                    fmt="o",
                    alpha=1,
                    ecolor="gray",
                    capsize=3,
                    elinewidth=1,
                    markeredgewidth=0,
                    markeredgecolor="k",
                    markersize=6,
                    markerfacecolor="#1f77b4",
                )
            else:
                plt.errorbar(
                    row[x],
                    row[y],
                    yerr=row[err_col],
                    fmt="X",
                    alpha=1,
                    ecolor="gray",
                    capsize=3,
                    elinewidth=1,
                    markeredgewidth=0,
                    markeredgecolor="k",
                    markersize=6,
                    markerfacecolor="#1f77b4",
                )

    # Add ligand labels at y = 0 on the x-axis
    for i, row in df.iterrows():
        plt.text(
            row[x],
            -4,
            row["Lig_Name"],
            fontsize=9,
            ha="center",
            va="center",
            color="black",
        )

    # Define limits for the reference lines
    if xlim:
        line_limits = [min(xlim[0], ylim[0]) - 2, max(xlim[1], ylim[1]) + 2]
    else:
        line_limits = [
            min(df[x].min(), df[y].min()) - 2,
            max(df[x].max(), df[y].max()) + 2,
        ]

    # Add shaded areas
    for deviation, alpha in zip([5, 2, 1], [0.1, 0.1, 0.1]):
        plt.fill_between(
            line_limits,
            [line_limits[0] - deviation * 1.414, line_limits[1] - deviation * 1.414],
            [line_limits[0] + deviation * 1.414, line_limits[1] + deviation * 1.414],
            color="grey",
            alpha=alpha,
        )

    # Add the y = x reference line
    plt.plot(
        [-100, 100], [-100, 100], linestyle="--", color="#BA0F52"
    )  # Red dashed line for y = x

    # Plot title with metrics
    plt.title(
        f"{title} | MUE: {mue:.2f} | RMSE: {rmse:.2f} | $R^2$: {r2:.2f} | Pearson: {pearson_corr:.2f}"
    )
    plt.xlabel(r"$\Delta G_{exp}$ (kcal/mol)")
    plt.ylabel(r"$\Delta G_{predicted}$ (kcal/mol)")
    plt.grid(True)  # Add a grid
    plt.axis("scaled")  # Equal scaling by x and y axes

    # Set axis limits
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([min(df[x].min(), df[y].min()) - 2, max(df[x].max(), df[y].max()) + 2])
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([min(df[x].min(), df[y].min()) - 2, max(df[x].max(), df[y].max()) + 2])

    # Save and show the plot
    plt.savefig(
        save_name, dpi=300, bbox_inches="tight"
    )  # Save the figure as a PNG file with high resolution
    plt.close()


def get_ABFE_dG(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Filter the DataFrame to get the row where the 'state' column is 'total'
    total_row = df[df["state"] == "total"]

    # Select all columns except 'state' and convert them to a list of floats
    return total_row.drop(columns="state").values.flatten().tolist()


def get_convergence(abfe_results_csv_path):
    kt_in_kcalmol = 0.592
    # Read the CSV file into a DataFrame
    abfe_analysis_folder = os.path.dirname(abfe_results_csv_path)
    # Read analysis/complex/convergence.csv
    convergence_df = pd.read_csv(
        os.path.join(abfe_analysis_folder, "complex", "ABFE_convergence.csv")
    )
    # Check convergence:
    if (
        abs(
            convergence_df[convergence_df["data_fraction"] == 0.5]["Forward"].values
            - convergence_df[convergence_df["data_fraction"] == 0.5]["Backward"].values
        )
        < kt_in_kcalmol
    ):
        return True
    else:
        return False


def main():
    # save df
    exp_v_abfe_df = pd.read_csv("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/exp_v_abfe_df_van_reps_all_hrex.csv")
    water_occupancy_v_delG_df = pd.read_csv('/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/analysis/a3_compare_bp_waters/a3_compare_bp_waters/wdir/water_occupancy_vs_delG.csv')

    for cmp in water_occupancy_v_delG_df["compound"].unique():
        # Filter the DataFrame for the current compound
        cmp_df = water_occupancy_v_delG_df[water_occupancy_v_delG_df["compound"] == cmp]
        most_solvated_van = cmp_df.loc[
            cmp_df["occupancy_vdw.20"].idxmax(), "van"
        ]
        logging.info(
            "Most solvated van for compound %s: %s", cmp, most_solvated_van
        )
        # Remove all rows in exp_v_abfe_df with LigName == cmp and van != most_solvated_van
        exp_v_abfe_df = exp_v_abfe_df[
            ~(
                (exp_v_abfe_df["Lig_Name"] == int(cmp))
                & (exp_v_abfe_df["Vanilla"] != 'van'+str(most_solvated_van))
            )
        ]
    
    print("exp_v_abfe_df", exp_v_abfe_df)

    # Plot unaveraged results
    plot_exp_v_abfe_averaged(
        exp_v_abfe_df,
        "Experimental_G",
        "MBAR",
        "MBAR_Error",
        "Christopher S1",
        "Christopher_s1_hrex_mbar_van_most_solvated.png",
        xlim=[-25, -4],
        color_by="Vanilla",
        ylim=[-25, -4],
    )
    plot_exp_v_abfe_averaged(
        exp_v_abfe_df,
        "Experimental_G",
        "TI",
        "TI_Error",
        "Christopher S1",
        "Christopher_s1_hrex_ti_van_most_solvated.png",
        xlim=[-25, -4],
        color_by="Vanilla",
        ylim=[-25, -4],
    )
    plot_exp_v_abfe_averaged(
        exp_v_abfe_df,
        "Experimental_G",
        "BAR",
        "BAR_Error",
        "Christopher S1",
        "Christopher_s1_hrex_bar_van_most_solvated.png",
        xlim=[-25, -4],
        color_by="Vanilla",
        ylim=[-25, -4],
    )




if __name__ == "__main__":
    main()
