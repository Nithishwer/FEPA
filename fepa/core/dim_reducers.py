"""
This module contains classes for dimensionality reduction of features
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd
from pensa.dimensionality import calculate_pca, project_on_eigenvector_pca
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
        pca_projection_df["ensemble"] = self.feature_df["ensemble"]
        pca_projection_df["timestep"] = self.feature_df["timestep"]
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
