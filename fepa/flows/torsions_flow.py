import logging
import os
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.featurizers import BATFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fepa.utils.stat_utils import (
    find_optimal_clusters_bic,
    find_optimal_clusters_elbow,
    find_optimal_clusters_silhouette,
)
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.analyzers import compute_relative_entropy


class torsions_analysis_workflow:
    """
    This class is used to analyze ligand torsions from ABFE simulations
    """

    def __init__(self, ensemble_handler, sel="resname unk"):
        self.ensemble_handler = ensemble_handler
        self.sel = sel

    def featurize(self):
        """
        Featurizes the ligand torsions using the BATFeaturizer
        """
        self.featurizer = BATFeaturizer(self.ensemble_handler, sel=self.sel)
        self.featurizer.featurize()
        self.feature_df = self.featurizer.get_feature_df()

    def reduce_dimensions(self, n_components=8):
        """
        Reduces the dimensions of the feature data using PCA
        """
        self.dimreducer = PCADimReducer(
            self.featurizer.get_feature_df(), n_components=n_components
        )
        self.dimreducer.reduce_dimensions(feature_column_keyword="BAT")
        self.dimreducer.calculate_projections()
        self.projection_df = self.dimreducer.get_pca_projection_df()
        self.dimred_visualizer = DimRedVisualizer(
            projection_df=self.projection_df, data_name="PCA"
        )

    def plot_dimred_sims(self, save_path):
        """
        Plots the PCA components for the ensemble
        """
        self.dimred_visualizer.plot_dimred_sims(save_path=save_path)

    def plot_dimred_time(self, save_path):
        """
        Plots the PCA components for the time
        """
        self.dimred_visualizer.plot_dimred_time(save_path=save_path)

    def plot_eigenvalues(self, save_path, n_components=8):
        """
        Plots the eigenvalues of the PCA
        """
        plot_eigenvalues(
            pca_object=self.dimreducer.get_pca(),
            n_components=n_components,
            save_path=save_path,
        )

    def find_optimal_clusters(self, method="bic", save_path=None):
        """
        Finds the optimal number of clusters using the specified method
        """
        if method == "silhouette":
            no_of_clusters = find_optimal_clusters_silhouette(
                self.projection_df["PC1"].tolist(), save_path=save_path
            )
        elif method == "bic":
            no_of_clusters = find_optimal_clusters_bic(
                self.projection_df["PC1"].tolist(), save_path=save_path
            )
        elif method == "elbow":
            no_of_clusters = find_optimal_clusters_elbow(
                self.projection_df["PC1"].tolist(), save_path=save_path
            )
        return no_of_clusters

    def cluster_pca(self, n_clusters, n_components=1):
        self.projection_df = cluster_pca(
            pca_projection_df=self.projection_df,
            n_clusters=n_clusters,
            n_components=n_components,
        )
        self.dimred_visualizer = DimRedVisualizer(
            projection_df=self.projection_df, data_name="PCA"
        )
        assert self.feature_df["ensemble"].equals(self.projection_df["ensemble"]), (
            "Mismatch in ensemble series between feature_df and projection_df"
        )
        self.feature_df["cluster"] = self.projection_df["cluster"]

    def plot_dimred_cluster(self, save_path, cluster_column="cluster"):
        """
        Plots the PCA components with clusters
        """
        self.dimred_visualizer.plot_dimred_cluster(
            save_path=save_path, cluster_column=cluster_column
        )

    def plot_torsions_jsd_histograms(self, cluster_1, cluster_2, save_path):
        relative_entropy_dict = compute_relative_entropy(
            feature_df=self.feature_df,
            key="cluster",
            ensemble1=cluster_1,
            ensemble2=cluster_2,
            num_bins=50,
            feature_column_keyword="BAT",
        )
        histograms = compute_histograms(
            feature_df=self.feature_df,
            key="cluster",
            value_1=cluster_1,
            value_2=cluster_2,
            num_bins=50,
            feature_column_keyword="BAT",
        )
        plot_jsd_histograms(
            relative_entropy_dict=relative_entropy_dict,
            histograms=histograms,
            save_path=save_path,
        )

    def plot_all_torsions_jsd_histograms(self, output_dir):
        clusters = self.projection_df["cluster"].unique()

        # Loop over all pairs of elements in clusters
        for i, cluster_1 in enumerate(clusters):
            for j, cluster_2 in enumerate(clusters[i + 1 :], start=i + 1):
                logging.info(
                    "Computing JSD for clusters %s and %s", cluster_1, cluster_2
                )
                self.plot_torsions_jsd_histograms(
                    cluster_1,
                    cluster_2,
                    save_path=os.path.join(
                        output_dir, f"jsd_histograms_{cluster_1}_{cluster_2}.png"
                    ),
                )

    def annotate_ensembles_in_dimred_df(self, save_path):
        self.projection_df["sim_type"] = [
            "abfe"
            if any(x in element for x in ["vdw", "coul", "rest"])
            else "nvt"
            if "nvt" in element
            else "van"
            for element in self.projection_df["ensemble"]
        ]
        self.projection_df.to_csv(
            save_path,
            index=False,
        )

    def plot_cluster_distr(self, save_path, groupby=["sim_type", "cluster"]):
        # Calculate the percentage of each cluster within each sim_type
        sim_type_cluster_counts = (
            self.projection_df.groupby(groupby).size().unstack(fill_value=0)
        )
        sim_type_totals = sim_type_cluster_counts.sum(axis=1)
        sim_type_cluster_percentages = (
            sim_type_cluster_counts.div(sim_type_totals, axis=0) * 100
        )

        # Create the stacked bar chart
        plt.figure(figsize=(14, 7), dpi=300)  # Increased DPI for higher resolution
        sim_type_cluster_percentages.plot(kind="bar", stacked=True)

        # Set the title and labels
        plt.title("Percentage of torsion clusters per Sim Type", fontsize=16)
        plt.xlabel("Sim Type", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Torsion Cluster", fontsize=10)
        sns.despine()  # Remove top and right spines for a cleaner look
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
