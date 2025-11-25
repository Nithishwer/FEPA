import os
import math
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import pandas as pd
import numpy as np
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

def plot_sidechain_distribution(
    df, ensembles, output_file="ensemble_histograms.png", ncols=4, colorby="ensemble"
):
    """
    Plots histograms for each column (excluding 'timestep' and the colorby column) in subplots,
    colored by the ensemble, and saves the entire plot to one image file.

    If exactly two ensembles are provided, columns are ordered by decreasing JS divergence.

    Parameters:
    - df: pandas DataFrame
    - ensembles: list/tuple/set of ensemble names to include (should be length 2 for JS sorting)
    - output_file: file path to save the combined plot
    - ncols: number of columns in the subplot grid
    - colorby: column name to color by (default is 'ensemble')
    """
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list, tuple, or set of ensemble names")

    ensembles = list(ensembles)
    if len(ensembles) < 1:
        raise ValueError("Must provide at least one ensemble")

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [col for col in df.columns if "CHI" in col]

    # Compute JS divergence if exactly two ensembles
    if len(ensembles) == 2:
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            # Drop NA
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna()

            # Shared bin edges
            min_val = min(d1.min(), d2.min())
            max_val = max(d1.max(), d2.max())
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)

            # Histogram densities
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)

            # Avoid 0s for JS divergence (add small constant)
            p1 += 1e-12
            p2 += 1e-12

            p1 /= p1.sum()
            p2 /= p2.sum()

            js = jensenshannon(p1, p2, base=2.0)
            js_scores[col] = js

        # Sort columns by JS divergence
        torsion_columns = sorted(
            torsion_columns, key=lambda k: js_scores[k], reverse=True
        )

    n_plots = len(torsion_columns)
    nrows = math.ceil(n_plots / ncols)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(torsion_columns):
        ax = axes[idx]
        sns.histplot(
            data=filtered_df,
            x=col,
            hue=colorby,
            kde=True,
            stat="density",
            common_norm=False,
            bins=30,
            ax=ax,
        )
        title = col
        if len(ensembles) == 2:
            title += f"{col}; \nJS={js_scores[col]:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove unused subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def circular_spread(angles_deg):
    """Compute circular spread = largest arc between points (max gap)."""
    vals = angles_deg.dropna().values
    if len(vals) < 2:
        return 0
    angles = np.deg2rad(vals)
    angles = np.sort(angles)
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    return np.max(gaps)


def plot_torsions_to_png(df, savedir="torsion_plots"):
    os.makedirs(savedir, exist_ok=True)

    chi_cols = [c for c in df.columns if c != "structure"]
    spreads = {col: circular_spread(df[col]) for col in chi_cols}
    chi_cols_sorted = sorted(chi_cols, key=lambda c: spreads[c], reverse=False)

    structures = df["structure"].unique()
    cmap = get_cmap("Set2")
    color_map = {s: cmap(i % 20) for i, s in enumerate(structures)}

    n_per_page = 25
    page = 1

    for i in range(0, len(chi_cols_sorted), n_per_page):
        print(f"Processing page {page}...")
        cols_this_page = chi_cols_sorted[i : i + n_per_page]

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.flatten()

        for ax, col in zip(axes, cols_this_page):
            theta = np.linspace(0, 2 * np.pi, 500)
            ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=0.6)

            for s in structures:
                vals = df[df["structure"] == s][col].dropna()
                ang = np.deg2rad(vals)
                x, y = np.cos(ang), np.sin(ang)
                ax.scatter(x, y, s=40, color=color_map[s], label=s)

            ax.set_title(f"{col}\nspread={spreads[col]:.2f} rad", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axhline(0, color="grey", linewidth=0.4)
            ax.axvline(0, color="grey", linewidth=0.4)

        for ax in axes[len(cols_this_page) :]:
            ax.axis("off")

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color=color_map[s], linestyle="", markersize=7
            )
            for s in structures
        ]
        fig.legend(
            handles, structures, loc="upper center", ncol=min(len(structures), 6)
        )

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        outfile = os.path.join(savedir, f"torsions_page_{page:02d}.png")
        plt.savefig(outfile, dpi=300)
        print(f"✅ Saved {outfile}")
        plt.close(fig)
        page += 1