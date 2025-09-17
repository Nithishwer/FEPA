"""
This module contains classes and functions for
visualizing dimensionality reduced data and comparing ensembles.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pensa.dimensionality.visualization import compare_mult_projections
from pensa.dimensionality import sort_traj_along_pc, pca_eigenvalues_plot

from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.feature_utils import (
    get_resid_pairs_from_sdf_names,
)
from fepa.utils.dimred_utils import get_projection_df_subset_values
from fepa.utils.stat_utils import js_divergence


class DimRedVisualizer:
    """
    A class for generating plots from dimensionality reduction data with metadata.
    Provides functionalities for scatter plots based on simulations, time, and clustering.
    """

    def __init__(self, projection_df, data_name="DimRed"):
        """
        Initializes the PlottingEngine with data and default plotting parameters.
        """
        logging.info("Initializing PlottingEngine for %s", data_name)
        self.projection_df = projection_df
        # Assumes first two columns are dim-red axes
        self.dimred_axes_names = projection_df.columns[:2]
        self.data_name = data_name
        # Define default plotting parameters for consistency across plots
        self.default_figsize = (10, 8)
        self.default_marker = "o"
        self.default_alpha = 0.8
        self.default_size = 20
        self.default_edgecolor = "white"
        self.linewidth = 0.5
        logging.debug("Default plotting parameters initialized.")

    def _setup_plot(self, title, x_label, y_label, figsize=None):
        """
        Sets up the basic plot structure with title, labels, and grid settings.
        This is a helper function to reduce code duplication in plot functions.
        """
        if figsize is None:
            # Use default figsize if not provided
            figsize = self.default_figsize
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(title, fontsize=20)
        # Disable grid for cleaner plots
        ax.grid(False)
        # Set background color to white
        ax.set_facecolor("white")
        logging.debug("Plot setup complete.")
        return fig, ax

    def _customize_scatter(
        self,
        ax,
        df,
        x_col,
        y_col,
        color,
        alpha=None,
        marker=None,
        size=None,
        label=None,
        linewidth=None,
    ):
        """
        Customizes scatter plot elements for a given dataset and parameters.
        This helper function applies consistent styling to scatter plots.
        """
        if alpha is None:
            # Use default alpha if not provided
            alpha = self.default_alpha
            logging.debug("Using default alpha: %s", alpha)
        if marker is None:
            # Use default marker if not provided
            marker = self.default_marker
            logging.debug("Using default marker: %s", marker)
        if size is None:
            # Use default size if not provided
            size = self.default_size
            logging.debug("Using default size: %s", size)
        # Plot scatter points
        ax.scatter(
            df[x_col],
            df[y_col],
            c=[color],
            s=size,
            alpha=alpha,
            marker=marker,
            edgecolor=self.default_edgecolor,
            label=label,
            linewidth=linewidth,
        )
        # Apply customizations including edgecolor
        logging.debug("Scatter plot customized.")

    def _add_highlight_text(self, ax, df, x_col, y_col, target):
        """
        Adds highlighted text labels near specific data points on the plot.
        Used to emphasize certain data points, like highlighted simulations.
        """
        logging.debug("Adding highlight text for target=%s", target)
        # Iterate through the index of the filtered DataFrame
        for i in df.index:
            # Position text slightly to the left of the point
            ax.text(
                df.loc[i, x_col] - 2,
                # Position text slightly above the point
                df.loc[i, y_col] + 0.5,
                # Text content is the target identifier
                target,
                # Style text for emphasis
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
        logging.debug("Highlight text added.")

    def plot_dimred_sims(
        self,
        column="ensemble",
        targets=None,
        highlights=None,
        save_path=None,
        marker=None,
        alpha=None,
        size=None,
        figsize=None,
    ):
        """
        Generates a scatter plot of dimensionality reduction results, colored by simulation ID.
        Allows highlighting specific simulations and saving the plot to a file.
        """
        logging.info(
            "Plotting %s for %s simulations",
            self.data_name,
            {"all" if targets is None else ", ".join(targets)},
        )

        df = self.projection_df
        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]
        logging.info(f"Plotting {x_col} vs {y_col}")
        title = f"2 Component {self.data_name}"

        if targets is None:
            # Get unique targets if none are specified
            targets = np.unique(df[column])
            logging.info(
                "No targets specified, plotting all unique targets: %s", targets
            )
        if highlights is None:
            # No highlights if none are specified
            highlights = []
            logging.info("No highlights specified, plotting without highlights")
        logging.debug("Targets to plot: %s, Highlights: %s", targets, highlights)

        # Setup plot with title and labels
        _, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)
        # Generate color palette
        colors = sns.color_palette("Set2", len(targets))

        # Iterate through targets and colors
        for target, color in zip(sorted(targets, reverse=True), colors):
            # Filter DataFrame for the current target
            target_df = df[df[column] == target]
            logging.debug(
                "Processing target: %s, Number of data points: %d",
                target,
                len(target_df),
            )
            # Determine alpha, highlight > arg > default
            current_alpha = (
                1.0
                if target in highlights
                else self.default_alpha
                if alpha is None
                else alpha
            )
            # Determine marker, highlight > arg > default
            current_marker = (
                "s"
                if target in highlights
                else self.default_marker
                if marker is None
                else marker
            )
            # Determine size, highlight > arg > default
            current_size = (
                100
                if target in highlights
                else self.default_size
                if size is None
                else size
            )
            # Determine size
            current_color = color if target not in highlights else "grey"
            # Label for legend
            label = target

            # Customize and plot scatter
            self._customize_scatter(
                ax,
                target_df,
                x_col,
                y_col,
                current_color,
                alpha=current_alpha,
                marker=current_marker,
                size=current_size,
                label=label,
                linewidth=self.linewidth,
            )

            # if target in highlights:
            #     # Add highlight text for highlighted targets
            #     self._add_highlight_text(ax, target_df, x_col, y_col, target)

        # Add legend outside the plot
        legend = ax.legend(
            title="Sim_ID",
            fontsize=12,
            title_fontsize="13",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        # Style legend border
        legend.get_frame().set_edgecolor("w")

        if save_path:
            # Save plot if save path is provided
            plt.savefig(save_path, bbox_inches="tight")
            logging.info("Plot saved to: %s", save_path)
        # Close plot to free memory
        plt.close()
        # User feedback
        print(f"{title} plot success")
        logging.info("%s plot complete", title)

    def plot_dimred_time(
        self,
        targets=None,
        column=None,
        method="pca",
        save_path=None,
        marker=None,
        alpha=None,
        size=None,
        figsize=None,
    ):
        """
        Generates a scatter plot of dimensionality reduction results, colored by timestep.
        Useful for visualizing data progression over time.
        """
        logging.info("Plotting %s over time for %s", self.data_name, method)
        title = f"{self.data_name} Over Time"
        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]

        if column is None:
            # Default column if none is provided
            column = "ensemble"

        if targets is None:
            # Use all targets if none specified
            targets = np.unique(self.projection_df[column])
            logging.info(
                "No targets specified, plotting for all unique targets: %s", targets
            )

        # Setup plot
        _, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)

        # Get timesteps for selected targets
        target_times = self.projection_df.loc[
            np.isin(self.projection_df[column], targets), "timestep"
        ]
        # Get unique timesteps
        times = np.unique(target_times)
        # Use viridis colormap for time progression
        cmap = plt.cm.viridis
        # Normalize time values for colormap
        norm = plt.Normalize(vmin=times.min(), vmax=times.max())
        # Create scalar mappable for colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Required for colorbar to work without explicit scatter plot data
        sm.set_array([])

        # Iterate through each timestep
        for time in times:
            # Filter data for current time and targets
            indices_2_keep = (np.isin(self.projection_df[column], targets)) & (
                self.projection_df["timestep"] == time
            )
            # Apply filter
            time_df = self.projection_df.loc[indices_2_keep]
            logging.debug(
                "Plotting time: %d, Number of data points: %s", time, len(time_df)
            )
            # Plot x-coordinates
            ax.scatter(
                time_df[x_col],
                # Plot y-coordinates
                time_df[y_col],
                # Color by time, use provided or default alpha
                c=[cmap(norm(time))],
                alpha=alpha if alpha is not None else 0.5,
                # Use provided or default marker
                marker=marker if marker is not None else self.default_marker,
                # Use provided or default size
                s=size if size is not None else self.default_size,
                # Label for legend
                label=f"Time {time}",
                # No edgecolor
                edgecolor=None,
                # Linewidth of marker
                linewidth=3,
            )

        # Add colorbar to show time scale
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
        # Label colorbar
        cbar.set_label("Time", fontsize=12)

        if save_path:
            # Save plot if path is provided
            plt.savefig(save_path, bbox_inches="tight")
            logging.info("Plot saved to: %s", save_path)
        # Close plot
        plt.close()
        logging.info("%s plot complete", title)

    def plot_dimred_cluster(
        self,
        cluster_column,
        clusters=None,
        palette="husl",
        alpha=None,
        s=None,
        save_path=None,
        figsize=None,
        marker=None,
        centroid_df=None,
    ):
        """
        Generates a scatter plot of dimensionality reduction results, colored by cluster assignment.
        Visualizes clustering outcomes based on a specified cluster column.
        """
        logging.info("Plotting clustering results for %s", self.data_name)
        # Check if cluster column exists
        if cluster_column not in self.projection_df.columns:
            print(f"Cluster column '{cluster_column}' not found.")
            print("Available columns:", self.projection_df.columns)
            logging.warning(
                "Cluster column %s not found in DataFrame. Available columns are: %s",
                cluster_column,
                self.projection_df.columns,
            )
            return  # Exit if cluster column is not found

        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]
        title = f"Clustering Results for {self.data_name}"
        # Get unique cluster labels
        clusters = self.projection_df[cluster_column].unique()

        # Setup plot
        _, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)
        # Generate color palette for clusters
        colors = sns.color_palette(palette, len(clusters))

        # Determine alpha, arg > default
        current_alpha = self.default_alpha if alpha is None else alpha
        # Determine size, arg > default
        current_size = self.default_size if s is None else s
        # Determine marker, arg > default
        current_marker = self.default_marker if marker is None else marker

        # Iterate through each cluster
        for i, cluster in enumerate(clusters):
            # Filter data for current cluster
            cluster_data = self.projection_df[
                self.projection_df[cluster_column] == cluster
            ]
            logging.info("Cluster %s size: %d", cluster, len(cluster_data))
            # Plot scatter for each cluster
            ax.scatter(
                cluster_data[x_col],
                cluster_data[y_col],
                label=f"Cluster {cluster}",
                s=current_size,
                edgecolors=self.default_edgecolor,
                alpha=current_alpha,
                c=[colors[i]],
                marker=current_marker,
                linewidth=self.linewidth,
            )

        if centroid_df is not None:
            # Plot centroids if provided
            ax.scatter(
                centroid_df[x_col],
                centroid_df[y_col],
                label="Centroids",
                s=100,
                edgecolors="black",
                alpha=1,
                c="black",
                marker="x",
                linewidth=1,
            )

        # Add legend
        legend = ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        legend.get_frame().set_edgecolor("w")

        # Adjust layout to prevent labels from overlapping
        plt.tight_layout()
        if save_path:
            # Save plot if path is provided
            plt.savefig(save_path, bbox_inches="tight")
            logging.info("Plot saved to: %s", save_path)
        # Close plot
        plt.close()
        logging.info("%s plot complete", title)


def sort_ensemble_by_pc(
    pca,
    projection_num: int,
    feature_df: pd.DataFrame,
    ensemble_handler: EnsembleHandler,
    ensemble: str,
    save_path: str,
    start_frame: int = 0,
    feature_column_keyword="DIST",
):
    """
    Sorts ensembles along a principal component axis and saves the sorted trajectory."
    #  Todo: Fix save name and path here currently saves as something.xtc_somthing.xtc
    """
    logging.info("Sorting ensembles along PCA projection %d...", projection_num)
    tpr_file = ensemble_handler.path_dict[ensemble]["tpr"]
    xtc_file = ensemble_handler.path_dict[ensemble]["xtc"]
    # Subset feature_df to only include the ensemble of interest
    ensemble_feature_df = feature_df[feature_df["ensemble"] == ensemble]
    ensemble_features = ensemble_feature_df.filter(regex=feature_column_keyword).values
    _ = sort_traj_along_pc(
        data=ensemble_features,
        top=tpr_file,
        trj=xtc_file,
        out_name=save_path,
        pca=pca,
        num_pc=projection_num,
        start_frame=start_frame,
    )
    logging.info(
        "Ensembles sorted along PCA projection %d and saved to %s",
        projection_num,
        save_path,
    )
    return


def compute_histograms(
    feature_df, key, value_1, value_2, num_bins=50, feature_column_keyword="DIST"
):
    """
    Compute histograms for Jensen-Shannon Divergence analysis.
    """
    # Subsetting groups
    ensemble_1_feature_df = feature_df[feature_df[key] == value_1]
    ensemble_2_feature_df = feature_df[feature_df[key] == value_2]
    sda_matrix_grp1 = ensemble_1_feature_df.filter(regex=feature_column_keyword).values
    sda_matrix_grp2 = ensemble_2_feature_df.filter(regex=feature_column_keyword).values
    num_columns = sda_matrix_grp1.shape[1]
    histograms = []
    for col in range(num_columns):
        # Extract the columns
        col_grp1 = sda_matrix_grp1[:, col]
        col_grp2 = sda_matrix_grp2[:, col]
        # Determine the bin edges based on the combined range of both columns
        min_val = min(col_grp1.min(), col_grp2.min())
        max_val = max(col_grp1.max(), col_grp2.max())
        bins = np.linspace(min_val, max_val, num_bins + 1)
        histograms.append((col_grp1, col_grp2, bins))
    return histograms


def plot_jsd_histograms(
    histograms,
    relative_entropy_dict,
    top_n=16,
    save_path=None,
    restr_u1_dict=None,
    restr_u2_dict=None,
):
    """
    Plots histograms of Jensen-Shannon Divergence values for two ensembles.
    """
    logging.info("Plotting Jensen-Shannon Divergence histograms...")

    # Get JS Entropy values
    jsd_values = relative_entropy_dict["jsd"]

    # Get top indices
    top_jsd_indices = np.argsort(jsd_values)[-top_n:]

    # Reverse the order of top_jsd_indices to go from max jsd to min
    top_jsd_indices = top_jsd_indices[::-1]

    # Ensure js_entropies is a numpy array
    jsd_values = np.array(jsd_values)

    # Get real resnames
    feature_name_list = np.array(relative_entropy_dict["name"])[top_jsd_indices]

    rows = 5
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    for i, idx in enumerate(top_jsd_indices):
        col_grp1, col_grp2, bins = histograms[idx]
        if restr_u1_dict is not None and restr_u2_dict is not None:
            # Get the corresponding restraint values
            resid_pair_list = get_resid_pairs_from_sdf_names(feature_name_list)
            # Get Restraints
            restr_u1 = restr_u1_dict[resid_pair_list[i]]
            restr_u2 = restr_u2_dict[resid_pair_list[i]]

            # Calculate means:
            grp1_mean = np.mean(col_grp1)
            grp2_mean = np.mean(col_grp2)

            # Define the boundaries for the shaded area
            lower_bound_1 = grp1_mean - restr_u1
            upper_bound_1 = grp1_mean + restr_u1
            lower_bound_2 = grp2_mean - restr_u2
            upper_bound_2 = grp2_mean + restr_u2

            # Plot Restraints
            axes[i].axvspan(
                xmin=lower_bound_1,
                xmax=upper_bound_1,
                color="blue",
                alpha=0.25,
                label="RMSF*a",
            )
            axes[i].axvspan(
                xmin=lower_bound_2,
                xmax=upper_bound_2,
                color="red",
                alpha=0.25,
                label="RMSF*a",
            )

        # Plot histograms
        sns.histplot(
            col_grp1,
            bins=bins,
            kde=False,
            color="blue",
            label="Group 1",
            stat="density",
            alpha=0.5,
            ax=axes[i],
        )
        sns.histplot(
            col_grp2,
            bins=bins,
            kde=False,
            color="red",
            label="Group 2",
            stat="density",
            alpha=0.5,
            ax=axes[i],
        )

        # Plot KDE
        sns.kdeplot(col_grp1, color="blue", label="Group 1 KDE", ax=axes[i])
        sns.kdeplot(col_grp2, color="red", label="Group 2 KDE", ax=axes[i])
        axes[i].set_title(
            f"{feature_name_list[i]} - JS Entropy: {jsd_values[idx]:.4f}\n"
            f"d: {abs(np.mean(col_grp1) - np.mean(col_grp2)):.2f}"
        )
        axes[i].legend()

    # Hide any empty subplots
    for j in range((len(top_jsd_indices)), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")  # Save the figure
    # plt.show()
    plt.close()


def plot_eigenvalues(pca_object, n_components, save_path=None):
    """
    Plot the eigenvalues of a PCA object.
    """
    if save_path is None:
        save_path = "plots/pca_eigenvalues.pdf"
    logging.info("Plotting PCA Eigenvalues...")
    pca_eigenvalues_plot(pca_object, num=n_components, plot_file=save_path)


def plot_pca_components(
    pca_object,
    feature_df,
    num: int,
    colorby="ensemble",
    save_path=None,
    feature_column_keyword="DIST",
):
    """
    This does not work with a lot of features. Check why -> TODO
    """
    """
    Plot the PCA components.
    """
    if save_path is None:
        save_path = "plots/pca_components.pdf"
    ensemble_label_list = np.unique(feature_df[colorby])
    ensemble_features_list = []
    for ensemble in ensemble_label_list:
        ensemble_feature_df = feature_df[feature_df[colorby] == ensemble]
        ensemble_features_values = ensemble_feature_df.filter(
            regex=feature_column_keyword
        ).values
        ensemble_features_list.append(ensemble_features_values)

    logging.info("Plotting PCA Components %s...", num)
    compare_mult_projections(
        data=ensemble_features_list,
        ana=pca_object,
        num=num,
        saveas=save_path,
        labels=ensemble_label_list,
        colors=sns.color_palette("Set2", len(ensemble_label_list)),
    )


def plot_entropy_heatmaps(
    cmp,
    columns_to_consider,
    projection_df,
    output_dir,
    suffix="",
    entropy_metric="jsd",
):
    """
    Plot the entropy heatmaps for a given compound.
    """

    # Compute relative entropy of features
    logging.info("Computing relative entropy for compound %s...", cmp)
    ensembles = list(np.unique(projection_df["ensemble"]))
    feature_rel_ent_matrix = {}

    # Subset only columns to consider plus ensemble and timestep
    projection_df = projection_df[columns_to_consider + ["ensemble", "timestep"]]

    for ensemble1 in ensembles:
        for ensemble2 in ensembles:
            logging.info(
                "Computing relative entropy for compound %s ensemble %s and ensemble %s...",
                cmp,
                ensemble1,
                ensemble2,
            )
            ensemble_1_data = get_projection_df_subset_values(
                projection_df, "ensemble", ensemble1
            )

            ensemble_2_data = get_projection_df_subset_values(
                projection_df, "ensemble", ensemble2
            )
            entropy = js_divergence(
                samples_p=ensemble_1_data,
                samples_q=ensemble_2_data,
            )
            feature_rel_ent_matrix[(ensemble1, ensemble2)] = entropy

    # Convert dictionary to a DataFrame
    matrix = np.zeros((len(ensembles), len(ensembles)))
    for i, ensemble1 in enumerate(ensembles):
        for j, ensemble2 in enumerate(ensembles):
            matrix[i, j] = feature_rel_ent_matrix[(ensemble1, ensemble2)]

    # Convert to Pandas DataFrame for visualization
    df = pd.DataFrame(matrix, index=ensembles, columns=ensembles)

    # Save df to the output directory
    df.to_csv(f"{output_dir}/{cmp}_entropy_matrix_{suffix}.csv")

    # Plot the full heatmap (no masking)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"size": 10},
        linecolor="white",
    )
    plt.title("Pairwise Entropy Heatmap", fontsize=18, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{cmp}_entropy_heatmap_{entropy_metric}.png", dpi=300)
    plt.close()
    logging.info(
        "Entropy heatmap saved to %s",
        f"{output_dir}/{cmp}_entropy_heatmap_{entropy_metric}.png",
    )
