"""
This module contains classes for dimensionality reduction of features
"""

import logging
from abc import ABC, abstractmethod
import umap
import pandas as pd
from pensa.dimensionality import project_on_eigenvector_pca
from fepa.utils.dimred_utils import calculate_pca
import pickle


class BaseDimReducer(ABC):
    """
    Abstract base class for dimensionality reduction techniques.

    Subclasses must implement the reduce_dimensions and cluster method.
    """

    def __init__(self, feature_df: pd.DataFrame, n_components: int):
        """
        Initializes DimReducer with a MDAnalysis Universe object.
        """
        self.feature_df = feature_df
        self.n_components = n_components

    @abstractmethod  # Define abstract method
    def reduce_dimensions(self, **kwargs) -> pd.DataFrame:
        """
        Abstract method for dimensionality reduction. To be implemented by subclasses.

        Args:
            sda_matrix (numpy.ndarray): Self-Distance Array matrix.
            **kwargs: Keyword arguments for specific dimensionality reduction methods.

        Returns:
            pandas.DataFrame: DataFrame containing the reduced components.
        """


class PCADimReducer(BaseDimReducer):
    """
    Class to perform PCA on features
    """

    def __init__(self, feature_df: pd.DataFrame, n_components: int = 8):
        self.pca = None
        self.feature_df = feature_df
        self.n_components = n_components
        super().__init__(feature_df, n_components)
        self.pca_projection_df = None
        self.method = "PCA"

    def reduce_dimensions(
        self, feature_column_keyword="DIST", **kwargs
    ) -> pd.DataFrame:
        """Reduce dimensions of feature data using PCA"""
        # Extract all data arrays
        self.feature_only_df = self.feature_df.filter(
            regex=feature_column_keyword, axis=1
        )
        # Calculate PCA
        logging.info("Calculating PCA...")
        self.pca = calculate_pca(self.feature_only_df.values, self.n_components)
        self.feature_column_keyword = feature_column_keyword
        return self.pca

    def get_pca(self):
        """Return the PCA object"""
        return self.pca

    def calculate_projections(self):
        """Calculate the projections of the feature data on the eigenvectors of the PCA"""
        if self.pca is None:
            print("PCA not calculated yet. Run reduce_dimensions() first.")
            return
        pca_projection_dict = {}
        for i in range(self.n_components):
            pca_projection_dict[f"PC{i + 1}"] = project_on_eigenvector_pca(
                self.feature_only_df.values, i, self.pca
            )
        pca_projection_df = pd.DataFrame(pca_projection_dict)
        # Add non feature columns to the projection DataFrame
        non_feature_columns = [
            col
            for col in self.feature_df.columns
            if self.feature_column_keyword not in col
        ]
        for col in non_feature_columns:
            pca_projection_df[col] = self.feature_df[col].values
        self.pca_projection_df = pca_projection_df

    def get_pca_projection_df(self):
        """Return the DataFrame containing the PCA projections"""
        return self.pca_projection_df

    def save_projection_df(self, save_path):
        """Save the DataFrame containing the PCA projections to a CSV file"""
        self.pca_projection_df.to_csv(save_path, index=False)

    def load_projection_df(self, load_path):
        """Load the DataFrame containing the PCA projections from a CSV file"""
        self.pca_projection_df = pd.read_csv(load_path)

    def save_pca(self, save_path):
        """Save the PCA object to a file"""
        with open(save_path, "wb") as f:
            pickle.dump(self.pca, f)

    def load_pca(self, save_path, feature_column_keyword="DIST"):
        """Load the PCA object from a file."""
        try:
            with open(save_path, "rb") as f:
                self.pca = pickle.load(f)
            print(f"PCA object loaded successfully from {save_path}")
            self.feature_only_df = self.feature_df.filter(
                regex=feature_column_keyword, axis=1
            )
            return self.pca
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading PCA object: {e}")
            return None


class UMAPDimReducer(
    BaseDimReducer
):  # Incomplete at the moment, need to be implemented
    """
    Class to perform UMAP on features
    This class cannot calculate projections as required
    """

    def __init__(self, feature_df: pd.DataFrame, n_components: int = 2):
        self.reducer = None
        self.feature_df = feature_df
        self.n_components = n_components
        super().__init__(feature_df, n_components)
        self.umap_projection_df = None
        self.method = "UMAP"

    def reduce_dimensions(
        self, feature_column_keyword="DIST", n_neighbors=15, min_dist=0.1, **kwargs
    ) -> pd.DataFrame:
        """Reduce dimensions of feature data using UMAP"""
        # Extract all data arrays
        self.feature_only_df = self.feature_df.filter(
            regex=feature_column_keyword, axis=1
        )
        # Calculate PCA
        logging.info("Calculating UMAP.x..")
        self.reducer = umap.UMAP()
        embedding = self.reducer.fit_transform(
            self.feature_only_df.values,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        self.umap_projection_df = pd.DataFrame(
            embedding, columns=[f"UMAP{i + 1}" for i in range(self.n_components)]
        )
        self.umap_projection_df["timestep"] = self.feature_df["timestep"].values
        self.umap_projection_df["ensemble"] = self.feature_df["ensemble"].values
        self.feature_column_keyword = feature_column_keyword

    def get_umap_projection_df(self):
        """Return the DataFrame containing the UMAP projections"""
        return self.umap_projection_df

    def save_projection_df(self, save_path):
        """Save the DataFrame containing the UMAP projections to a CSV file"""
        self.umap_projection_df.to_csv(save_path, index=False)

    def load_projection_df(self, load_path):
        """Load the DataFrame containing the UMAP projections from a CSV file"""
        self.umap_projection_df = pd.read_csv(load_path)

    def save_umap(self, save_path):
        """Save the UMAP object to a file"""
        with open(save_path, "wb") as f:
            pickle.dump(self.reducer, f)

    def load_umap(self, save_path, feature_column_keyword="DIST"):
        """Load the UMAP object from a file."""
        try:
            with open(save_path, "rb") as f:
                self.reducer = pickle.load(f)
            print(f"UMAP object loaded successfully from {save_path}")
            self.feature_only_df = self.feature_df.filter(
                regex=feature_column_keyword, axis=1
            )
            return self.reducer
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading UMAP object: {e}")
            return None
