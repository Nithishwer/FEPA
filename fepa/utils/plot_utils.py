import os
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from fepa.utils.stat_utils import calculate_metrics


def plot_exp_v_predicted(
    df,
    x,
    y,
    err_col,
    title,
    save_name,
    metrics_dict,
    xlim=None,
    ylim=None,
    color_by=None,
    label=False,
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
    title (str): The title of the plot.
    xlim (tuple, optional): The limits for the x-axis. Defaults to None.
    ylim (tuple, optional): The limits for the y-axis. Defaults to None.
    color_by (str, optional): The column name to color the points by. Defaults to None.
    """
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    if color_by:
        unique_vals = df[color_by].unique()
        palette = sns.color_palette("husl", len(unique_vals))
        color_map = dict(zip(unique_vals, palette))
        df["color"] = df[color_by].map(color_map)
    else:
        df["color"] = "#1f77b4"  # default color

    # Plot the points with error bars
    for i, row in df.iterrows():
        fmt = "o" if row["Convergence"] else "X"
        color = row["color"] if color_by else "#1f77b4"
        plt.errorbar(
            row[x],
            row[y],
            yerr=row[err_col],
            fmt=fmt,
            alpha=1,
            ecolor="gray",
            capsize=3,
            elinewidth=1,
            markeredgewidth=0,
            markeredgecolor="k",
            markersize=6,
            markerfacecolor=color,
        )
        if label == True:
            plt.text(
                row[x],
                row[y] + 0.5,
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
        f"{title} | MUE: {metrics_dict['mue']:.2f} | RMSE: {metrics_dict['rmse']:.2f} | $R^2$: {metrics_dict['r2']:.2f} | Pearson: {metrics_dict['pearson_corr']:.2f}"
    )
    plt.xlabel(r"$\Delta G_{exp}$ (kcal/mol)")
    plt.ylabel(r"$\Delta G_{predicted}$ (kcal/mol)")
    plt.grid(True)  # Add a grid
    plt.axis("scaled")  # Equal scaling by x and y axes

    # Add legend
    if color_by:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(k),
                markerfacecolor=v,
                markersize=6,
            )
            for k, v in color_map.items()
        ]
        plt.legend(handles=handles, title=color_by)

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


