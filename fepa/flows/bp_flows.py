import logging
import os
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    compute_histograms,
    plot_jsd_histograms,
    plot_entropy_heatmaps,
    plot_pca_components,
)
from fepa.utils.dimred_utils import cluster_pca
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)


class binding_pocket_analysis_workflow:
    """
    This class is used to analyze the binding pocket from ABFE simulations and compare it to apo and holo
    """

    def __init__(self, cmp, ensemble_handler):
        self.cmp = cmp
        self.ensemble_handler = ensemble_handler
        self.featurizer = SelfDistanceFeaturizer(ensemble_handler)

    def check_consistency(self):
        # Load trajectories
        logging.info("Making universes for compound %s...", self.cmp)
        self.ensemble_handler.make_universes()
        logging.info("Checking residue consistency for compound %s...", self.cmp)
        check_bp_residue_consistency(self.ensemble_handler.get_universe_dict())

    def featurize(self, cmp_output_dir):
        """
        Featurizes the binding pocket using the BATFeaturizer
        """
        # Featurize
        logging.info("Featurizing compound %s...", self.cmp)
        self.featurizer.featurize()
        self.featurizer.save_features(output_dir=cmp_output_dir, overwrite=True)

    def reduce_dimensions(
        self,
        cmp_output_dir,
    ):
        """
        Performs PCA on the features and saves the projection dataframe
        """
        # Dimensionality Reduction
        logging.info("Performing dimensionality reduction for compound %s...", self.cmp)
        self.dimreducer = PCADimReducer(
            self.featurizer.get_feature_df(), n_components=8
        )
        self.dimreducer.reduce_dimensions()
        self.dimreducer.calculate_projections()
        self.dimreducer.save_projection_df(
            save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        )

        # Projection df without nvt:
        self.projection_df = self.dimreducer.get_pca_projection_df()

    def plot_entropy_heatmaps(self, cmp_output_dir):
        """
        Plots the entropy heatmaps for the binding pocket
        """
        plot_entropy_heatmaps(
            cmp=self.cmp,
            entropy_metric="jsd",
            columns_to_consider=["PC1", "PC2"],
            ensemble_handler=self.ensemble_handler,
            projection_df=self.projection_df,
            output_dir=cmp_output_dir,
        )

    def plot_holo_pca(self, cmp_output_dir):
        """
        Plots the PCA components and eigenvalues
        """
        logging.info("Visualizing compound %s...", self.cmp)
        ensembles_to_plot = [
            f"{self.cmp}_van_1",
            f"{self.cmp}_van_2",
            f"{self.cmp}_van_3",
            f"{self.cmp}_nvt",
        ]
        # Get projection df subset
        projection_df_subset = self.projection_df[
            self.projection_df["ensemble"].isin(ensembles_to_plot)
        ].copy()
        feature_df_subset = self.featurizer.get_feature_df()[
            self.featurizer.get_feature_df()["ensemble"].isin(ensembles_to_plot)
        ].copy()
        # Visualization
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df_subset, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "holo_pca_components_ensemble.png"),
            highlights=[f"{self.cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "holo_pca_components_time.png"),
        )
        plot_eigenvalues(
            pca_object=self.dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=self.dimreducer.get_pca(),
            feature_df=feature_df_subset[
                feature_df_subset["ensemble"] != f"{self.cmp}_nvt"
            ],
            num=8,
            save_path=os.path.join(cmp_output_dir, "holo_pca_components.png"),
        )

    def plot_apo_holo_pca(self, cmp_output_dir):
        """
        Plots the PCA components and eigenvalues for ABFE simulations
        """
        logging.info("Visualizing compound %s...", self.cmp)
        ensembles_to_plot = [
            "apo_1",
            "apo_2",
            "apo_3",
            f"{self.cmp}_van_1",
            f"{self.cmp}_van_2",
            f"{self.cmp}_van_3",
            f"{self.cmp}_nvt",
        ]
        # Get projection df subset
        projection_df_subset = self.projection_df[
            self.projection_df["ensemble"].isin(ensembles_to_plot)
        ].copy()
        feature_df_subset = self.featurizer.get_feature_df()[
            self.featurizer.get_feature_df()["ensemble"].isin(ensembles_to_plot)
        ].copy()
        # Visualization
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df_subset, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(
                cmp_output_dir, "apo_holo_pca_components_ensemble.png"
            ),
            highlights=[f"{self.cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "apo_holo_pca_components_time.png"),
        )
        plot_eigenvalues(
            pca_object=self.dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=self.dimreducer.get_pca(),
            feature_df=feature_df_subset[
                feature_df_subset["ensemble"] != f"{self.cmp}_nvt"
            ],
            num=8,
            save_path=os.path.join(cmp_output_dir, "apo_holo_pca_components.png"),
        )

    def plot_apo_abfe_pca(self, cmp_output_dir):
        """
        Plots the PCA components and eigenvalues
        """
        logging.info("Visualizing compound %s...", self.cmp)
        generic_abfe_windows = [
            "{CMP}_van_{VAN_NO}_coul.00",
            "{CMP}_van_{VAN_NO}_coul.05",
            "{CMP}_van_{VAN_NO}_coul.10",
            "{CMP}_van_{VAN_NO}_vdw.00",
            "{CMP}_van_{VAN_NO}_vdw.10",
            "{CMP}_van_{VAN_NO}_vdw.15",
            "{CMP}_van_{VAN_NO}_vdw.20",
            "{CMP}_van_{VAN_NO}_rest.05",
            "{CMP}_van_{VAN_NO}_rest.10",
        ]
        van1_abfe_windows = [
            string.format(CMP=self.cmp, VAN_NO=1) for string in generic_abfe_windows
        ]
        van2_abfe_windows = [
            string.format(CMP=self.cmp, VAN_NO=2) for string in generic_abfe_windows
        ]
        van3_abfe_windows = [
            string.format(CMP=self.cmp, VAN_NO=3) for string in generic_abfe_windows
        ]
        ensembles_to_plot = (
            [
                f"apo_1",
                f"apo_2",
                f"apo_3",
                f"{self.cmp}_nvt",
            ]
            + van1_abfe_windows
            + van2_abfe_windows
            + van3_abfe_windows
        )
        # Get projection df subset
        projection_df_subset = self.projection_df[
            self.projection_df["ensemble"].isin(ensembles_to_plot)
        ].copy()
        feature_df_subset = self.featurizer.get_feature_df()[
            self.featurizer.get_feature_df()["ensemble"].isin(ensembles_to_plot)
        ].copy()
        # Replace individual abfe windows with vanilla names
        projection_df_subset["ensemble"] = projection_df_subset["ensemble"].replace(
            van1_abfe_windows,
            [f"{self.cmp}_van_1_abfe" for _ in range(len(van1_abfe_windows))],
        )
        projection_df_subset["ensemble"] = projection_df_subset["ensemble"].replace(
            van2_abfe_windows,
            [f"{self.cmp}_van_2_abfe" for _ in range(len(van2_abfe_windows))],
        )
        projection_df_subset["ensemble"] = projection_df_subset["ensemble"].replace(
            van3_abfe_windows,
            [f"{self.cmp}_van_3_abfe" for _ in range(len(van3_abfe_windows))],
        )
        # Replace individual abfe windows with vanilla names
        feature_df_subset["ensemble"] = feature_df_subset["ensemble"].replace(
            van1_abfe_windows,
            [f"{self.cmp}_van_1_abfe" for _ in range(len(van1_abfe_windows))],
        )
        feature_df_subset["ensemble"] = feature_df_subset["ensemble"].replace(
            van2_abfe_windows,
            [f"{self.cmp}_van_2_abfe" for _ in range(len(van2_abfe_windows))],
        )
        feature_df_subset["ensemble"] = feature_df_subset["ensemble"].replace(
            van3_abfe_windows,
            [f"{self.cmp}_van_3_abfe" for _ in range(len(van3_abfe_windows))],
        )
        # Visualization
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df_subset, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(
                cmp_output_dir, "apo_abfe_pca_components_ensemble.png"
            ),
            highlights=[f"{self.cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "apo_abfe_pca_components_time.png"),
        )
        plot_eigenvalues(
            pca_object=self.dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=self.dimreducer.get_pca(),
            feature_df=feature_df_subset[
                feature_df_subset["ensemble"] != f"{self.cmp}_nvt"
            ],
            num=8,
            save_path=os.path.join(cmp_output_dir, "apo_abfe_pca_components.png"),
        )

    def cluster_pca(self, n_clusters, n_components):
        """
        Performs clustering on the PCA components and saves the clustering results
        """
        # Perform clustering
        logging.info("Performing clustering for compound %s...", self.cmp)
        self.projection_df = cluster_pca(
            pca_projection_df=self.projection_df,
            n_clusters=n_clusters,
            n_components=n_components,
        )

    def plot_pca_cluster(self, cmp_output_dir):
        """
        Plots the PCA components with clustering
        """
        # Plotting PCA with clusters
        logging.info("Plotting PCA with clusters for compound %s...", self.cmp)
        dimred_visualizer = DimRedVisualizer(
            projection_df=self.projection_df, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_cluster(
            cluster_column="cluster",
            save_path=os.path.join(
                cmp_output_dir, f"{self.cmp}_pca_components_cluster.png"
            ),
        )

    def load_results(self, cmp_output_dir):
        """
        Loads the results from the clustering
        """
        # Load projection df
        projection_csv = os.path.join(cmp_output_dir, "pca_projection_df.csv")
        features_csv = os.path.join(cmp_output_dir, "SelfDistance_features.csv")
        # if exists, load it
        if os.path.exists(projection_csv):
            logging.info("Loading projection df from %s", projection_csv)
            self.projection_df = pd.read_csv(projection_csv)
            print(self.projection_df)
        else:
            logging.info(f"Projection df not found at {projection_csv}")

        # Load feature df
        if os.path.exists(features_csv):
            logging.info("Loading feature df from %s", features_csv)
            self.featurizer.load_features(input_dir=cmp_output_dir)
            self.dimreducer = PCADimReducer(
                self.featurizer.get_feature_df(), n_components=8
            )
            self.dimreducer.reduce_dimensions()
            self.dimreducer.calculate_projections()
        else:
            logging.info(f"Feature df not found at {features_csv}")
