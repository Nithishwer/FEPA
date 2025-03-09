import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging


class Visualizer:
    """
    A class for generating plots from dimensionality reduction data with metadata.
    Provides functionalities for scatter plots based on simulations, time, and clustering.
    """
    def __init__(self, dimred_df_w_metadata, data_name='DimRed', compound_name=None):
        """
        Initializes the PlottingEngine with data and default plotting parameters.
        """
        logging.info(f"Initializing PlottingEngine for {data_name}")
        self.dimred_df_w_metadata = dimred_df_w_metadata
        # Assumes first two columns are dim-red axes
        self.dimred_axes_names = dimred_df_w_metadata.columns[:2]
        self.data_name = data_name
        self.compound_name = compound_name
        # Define default plotting parameters for consistency across plots
        self.default_figsize = (10, 6)
        self.default_marker = 'o'
        self.default_alpha = 0.8
        self.default_size = 20
        self.default_edgecolor = 'white'
        self.linewidth = 0.5
        logging.debug(f"Default plotting parameters initialized.")


    def _setup_plot(self, title, x_label, y_label, figsize=None):
        """
        Sets up the basic plot structure with title, labels, and grid settings.
        This is a helper function to reduce code duplication in plot functions.
        """
        logging.debug(f"Setting up plot: title='{title}', x_label='{x_label}', y_label='{y_label}', figsize={figsize}")
        if figsize is None:
            # Use default figsize if not provided
            figsize = self.default_figsize
            logging.debug(f"Using default figsize: {figsize}")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(title, fontsize=20)
        # Disable grid for cleaner plots
        ax.grid(False)
        # Set background color to white
        ax.set_facecolor('white')
        logging.debug(f"Plot setup complete.")
        return fig, ax

    def _customize_scatter(self, ax, df, x_col, y_col, target, color, alpha=None, marker=None, size=None, label=None, linewidth=None):
        """
        Customizes scatter plot elements for a given dataset and parameters.
        This helper function applies consistent styling to scatter plots.
        """
        logging.debug(f"Customizing scatter for target='{target}', color='{color}', alpha={alpha}, marker='{marker}, size={size}, label='{label}'")
        if alpha is None:
            # Use default alpha if not provided
            alpha = self.default_alpha
            logging.debug(f"Using default alpha: {alpha}")
        if marker is None:
            # Use default marker if not provided
            marker = self.default_marker
            logging.debug(f"Using default marker: {marker}")
        if size is None:
            # Use default size if not provided
            size = self.default_size
            logging.debug(f"Using default size: {size}")
        # Plot scatter points
        ax.scatter(df[x_col], df[y_col],
                   c=[color], s=size, alpha=alpha, marker=marker, edgecolor=self.default_edgecolor, label=label, linewidth=linewidth)
        # Apply customizations including edgecolor
        logging.debug(f"Scatter plot customized.")

    def _add_highlight_text(self, ax, df, x_col, y_col, target):
        """
        Adds highlighted text labels near specific data points on the plot.
        Used to emphasize certain data points, like highlighted simulations.
        """
        logging.debug(f"Adding highlight text for target='{target}'")
        # Iterate through the index of the filtered DataFrame
        for i in df.index:
            # Position text slightly to the left of the point
            ax.text(df.loc[i, x_col] - 2,
                    # Position text slightly above the point
                    df.loc[i, y_col] + 0.5,
                    # Text content is the target identifier
                    target,
                    # Style text for emphasis
                    fontsize=9, fontweight='bold', ha='left', va='bottom')
        logging.debug(f"Highlight text added.")

    def plot_dimred_sims(self, column = 'id',  targets=None, highlights=None, save_path=None,
                          marker=None, alpha=None, size=None, figsize=None):
        """
        Generates a scatter plot of dimensionality reduction results, colored by simulation ID.
        Allows highlighting specific simulations and saving the plot to a file.
        """
        logging.info(f"Plotting {self.data_name} for {'all' if targets is None else ', '.join(targets)} simulations")

        df = self.dimred_df_w_metadata
        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]
        title = f'2 Component {self.data_name}'

        if targets is None:
            # Get unique targets if none are specified
            targets = np.unique(df[column])
            logging.info(f"No targets specified, plotting all unique targets: {targets}")
        if highlights is None:
            # Initialize highlights list if None
            highlights = [f'{self.compound_name}_nvt']
        logging.debug(f"Targets to plot: {targets}, Highlights: {highlights}")

        # Setup plot with title and labels
        fig, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)
        # Generate color palette
        colors = sns.color_palette("Set2", len(targets))

        # Iterate through targets and colors
        for target, color in zip(sorted(targets, reverse=True), colors):
            # Filter DataFrame for the current target
            target_df = df[df[column] == target]
            logging.debug(f"Processing target: {target}, Number of data points: {len(target_df)}")
            # Determine alpha, highlight > arg > default
            current_alpha = 1.0 if target in highlights else self.default_alpha if alpha is None else alpha
            # Determine marker, highlight > arg > default
            current_marker = 's' if target in highlights else self.default_marker if marker is None else marker
            # Determine size, highlight > arg > default
            current_size = 100 if target in highlights else self.default_size if size is None else size
            # Label for legend
            label = target

            # Customize and plot scatter
            self._customize_scatter(ax, target_df, x_col, y_col, target, color, alpha=current_alpha, marker=current_marker, size=current_size, label=label, linewidth=self.linewidth)

            # if target in highlights:
            #     # Add highlight text for highlighted targets
            #     self._add_highlight_text(ax, target_df, x_col, y_col, target)

        # Add legend outside the plot
        legend = ax.legend(title="Sim_ID", fontsize=12, title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Style legend border
        legend.get_frame().set_edgecolor('w')

        if save_path:
            # Save plot if save path is provided
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Plot saved to: {save_path}")
        # Close plot to free memory
        plt.close()
        # User feedback
        print(f"{title} plot success")
        logging.info(f"{title} plot complete")


    def plot_dimred_time(self, targets=None, column = None, method='pca', save_path=None,
                         marker=None, alpha=None, size=None, figsize=None):
        """
        Generates a scatter plot of dimensionality reduction results, colored by timestep.
        Useful for visualizing data progression over time.
        """
        logging.info(f"Plotting {self.data_name} over time for {targets}")
        title = f'{self.data_name} Over Time'
        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]

        if column is None:
            # Default column if none is provided
            column = 'id'

        if targets is None:
            # Use all targets if none specified
            targets = np.unique(self.dimred_df_w_metadata[column])
            logging.info(f"No targets specified, plotting for all unique targets: {targets}")

        # Setup plot
        fig, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)

        # Get timesteps for selected targets
        target_times = self.dimred_df_w_metadata.loc[np.isin(self.dimred_df_w_metadata[column], targets), 'timestep']
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
            indicesToKeep = (np.isin(self.dimred_df_w_metadata[column],targets)) & (self.dimred_df_w_metadata['timestep'] == time)
            # Apply filter
            time_df = self.dimred_df_w_metadata.loc[indicesToKeep]
            logging.debug(f"Plotting time: {time}, Number of data points: {len(time_df)}")
            # Plot x-coordinates
            ax.scatter(time_df[x_col],
                       # Plot y-coordinates
                       time_df[y_col],
                       # Color by time, use provided or default alpha
                       c=[cmap(norm(time))], alpha= alpha if alpha is not None else 0.5,
                       # Use provided or default marker
                       marker = marker if marker is not None else self.default_marker,
                       # Use provided or default size
                       s= size if size is not None else self.default_size,
                       # Label for legend
                       label=f'Time {time}',
                       # No edgecolor
                       edgecolor=None,
                       # Linewidth of marker
                       linewidth=3)

        # Add colorbar to show time scale
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        # Label colorbar
        cbar.set_label('Time', fontsize=12)

        if save_path:
            # Save plot if path is provided
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Plot saved to: {save_path}")
        # Close plot
        plt.close()
        logging.info(f"{title} plot complete")


    def plot_dimred_cluster(self, cluster_column, clusters=None, palette="husl", alpha=None, s=None, save_path=None, figsize=None, marker=None):
        """
        Generates a scatter plot of dimensionality reduction results, colored by cluster assignment.
        Visualizes clustering outcomes based on a specified cluster column.
        """
        logging.info(f"Plotting clustering results for {self.data_name}")
        # Check if cluster column exists
        if cluster_column not in self.dimred_df_w_metadata.columns:
            print(f"Cluster column '{cluster_column}' not found.")
            print("Available columns:", self.dimred_df_w_metadata.columns)
            logging.warning(f"Cluster column '{cluster_column}' not found in DataFrame. Available columns are: {self.dimred_df_w_metadata.columns.tolist()}")
            return # Exit if cluster column is not found

        x_col = self.dimred_axes_names[0]
        y_col = self.dimred_axes_names[1]
        title = f'Clustering Results for {self.data_name}'
        # Get unique cluster labels
        clusters = self.dimred_df_w_metadata[cluster_column].unique()

        # Setup plot
        fig, ax = self._setup_plot(title, x_col, y_col, figsize=figsize)
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
            cluster_data = self.dimred_df_w_metadata[self.dimred_df_w_metadata[cluster_column] == cluster]
            logging.info(f"Cluster {cluster} size: {len(cluster_data)}")
            # Plot scatter for each cluster
            ax.scatter(cluster_data[x_col], cluster_data[y_col],
                       label=f'Cluster {cluster}', s=current_size, edgecolors=self.default_edgecolor, alpha=current_alpha, c=[colors[i]], marker=current_marker, linewidth=self.linewidth)

        # Add legend
        legend = ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_edgecolor('w')

        # Adjust layout to prevent labels from overlapping
        plt.tight_layout()
        if save_path:
            # Save plot if path is provided
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Plot saved to: {save_path}")
        # Close plot
        plt.close()
        logging.info(f"{title} plot complete")