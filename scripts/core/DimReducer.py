import os
import umap
import pickle
import hdbscan
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import MDAnalysis.lib.distances as distances
from scipy.spatial.distance import jensenshannon
from abc import ABC, abstractmethod # Import for Abstract Base Class

class DimReducer(ABC): # Make DimReducer an Abstract Base Class
    """
    Abstract base class for dimensionality reduction techniques.

    Subclasses must implement the reduce_dimensions method.
    """
    def __init__(self, universe, metadata_df):
        """
        Initializes DimReducer with a MDAnalysis Universe object.
        """
        self.universe = universe
        self.k2ij = self._make_k2ij_dict()
        self.sda_matrix = []
        self.dimred_df = None
        self.metadata_df = metadata_df


    def _make_k2ij_dict(self):
        """
        Creates a dictionary mapping index 'k' to pairs of residue indices (i, j)
        for residues in the binding pocket.
        """
        pocket_res_ca = self.universe.select_atoms("segid BP and protein and name CA")
        n = len(pocket_res_ca)
        k2ij = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                k2ij[k] = (i, j)
                k += 1
        return k2ij

    def analyse_sda(self):
        """
        Calculates Self-Distance Arrays (SDA).

        Returns:
            numpy.ndarray: SDA matrix (NumPy array).
        """
        pocket_res_ca = self.universe.select_atoms("segid BP and protein and name CA")
        for ts in self.universe.trajectory:
            sda = distances.self_distance_array(pocket_res_ca.positions, box=None)
            self.sda_matrix.append(np.array(sda))

    @abstractmethod # Define abstract method
    def reduce_dimensions(self, n_components=2, **kwargs) -> pd.DataFrame:
        """
        Abstract method for dimensionality reduction. To be implemented by subclasses.

        Args:
            sda_matrix (numpy.ndarray): Self-Distance Array matrix.
            n_components (int, optional): Number of components to reduce to. Defaults to 2.
            **kwargs: Keyword arguments for specific dimensionality reduction methods.

        Returns:
            pandas.DataFrame: DataFrame containing the reduced components.
        """
        pass # Abstract method has no implementation in the base class

    @abstractmethod # Define abstract method
    def cluster(self, **kwargs):
        """
        Abstract method for clustering. To be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments for specific clustering methods.

        Returns:
            pandas.DataFrame: DataFrame containing the cluster labels.
        """
        pass

    def elbow_plot(self, save_path=None):
        """Plots elbow plot for clustering."""
        if self.dimred_df is None or self.dimred_df.empty:
            print(f"{self.method.upper()} data not available.")
            return

        X = self.dimred_df[self.dimred_axes_names]
        X_normalized = preprocessing.scale(X)
        sse = []

        title = f'Elbow Plot for {self.method}'
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init='auto' for sklearn > 1.2
            kmeans.fit(X_normalized)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 20), sse, marker='o', linestyle='-')
        plt.ylabel('Sum of Squared Distances')
        plt.xlabel('Parameter k for KMeans')
        plt.title(title)
        plt.xticks(range(1, 20))
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def get_output(self):
        """Returns the reduced dimensions DataFrame concatenated with the metadata DataFrame."""
        if self.dimred_df is not None:
            return pd.concat([self.dimred_df, self.metadata_df], axis=1)
        else:
            return None

    @classmethod
    def save_object(cls, dim_reducer_object, filename, output_dir): # Class method
        """Saves DimReducer object to a file."""
        save_path = os.path.join(output_dir, filename)
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(dim_reducer_object, f)
            logging.info(f"DimReducer object saved successfully to '{save_path}'")
        except Exception as e:
            logging.info(f"Error saving DimReducer object to '{save_path}': {e}")

    @classmethod
    def load_object(cls, filename, output_dir): # Class method
        """Loads DimReducer object from a file."""
        load_path = os.path.join(output_dir, filename)
        try:
            with open(load_path, 'rb') as f:
                loaded_object = pickle.load(f)
            logging.info(f"DimReducer object loaded successfully from '{load_path}'")
            return loaded_object
        except FileNotFoundError:
            logging.error(f"Error: File '{load_path}' not found.")
            return None
        except Exception as e:
            logging.error(f"Error loading DimReducer object from '{load_path}': {e}")
            return None



class PCAReducer(DimReducer): # Subclass for PCA
    """
    Performs PCA dimensionality reduction.
    """
    def __init__(self, universe, metadata_df):
        super().__init__(universe, metadata_df)
        self.method = 'PCA'

    def reduce_dimensions(self, n_components=2, **kwargs) -> pd.DataFrame:
        """
        Performs PCA dimensionality reduction on the SDA matrix.

        Args:
            sda_matrix (numpy.ndarray): Self-Distance Array matrix.
            n_components (int, optional): Number of principal components to compute. Defaults to 2.
            **kwargs:  Ignored keyword arguments.

        Returns:
            pandas.DataFrame: DataFrame containing the principal components.
        """
        try:
            reducer = PCA(n_components=n_components)
            columns = [f'principal component {i + 1}' for i in range(n_components)]
            components = reducer.fit_transform(self.sda_matrix)
            self.dimred_df = pd.DataFrame(data=components, columns=columns)
            self.dimred_axes_names = [f'principal component {i + 1}' for i in range(n_components)]
        except Exception as e:
            print(f"PCA failed: {e}")
            return None

    def cluster(self, best_k=6): # Specific clustering methods - these could potentially be moved to a separate Clustering class/strategy later if needed.
        """Performs KMeans clustering. Returns DataFrame with cluster labels."""
        try:
            columns = self.dimred_axes_names

            X = self.dimred_df[columns]
            X_normalized = preprocessing.scale(X)
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto') # n_init='auto' for sklearn > 1.2
            labels = kmeans.fit_predict(X_normalized) # Use fit_predict
            self.dimred_df['cluster'] = labels
            return self.dimred_df
        except Exception as e:
            print(f"KMeans clustering failed: {e}")
            return None
    


class TSNEReducer(DimReducer): # Subclass for TSNE
    """
    Performs TSNE dimensionality reduction.
    """
    def __init__(self, universe, metadata_df):
        super().__init__(universe, metadata_df)
        self.method = 'TSNE'

    def reduce_dimensions(self, n_components=2, **kwargs) -> pd.DataFrame:
        """
        Performs TSNE dimensionality reduction on the SDA matrix.

        Args:
            sda_matrix (numpy.ndarray): Self-Distance Array matrix.
            n_components (int, optional): Number of TSNE components to compute. Defaults to 2.
            **kwargs: Ignored keyword arguments.

        Returns:
            pandas.DataFrame: DataFrame containing the TSNE components.
        """
        try:
            reducer = TSNE(n_components=n_components)
            columns = [f'tsne component {i + 1}' for i in range(n_components)]
            components = reducer.fit_transform(self.sda_matrix)
            self.dimred_df = pd.DataFrame(data=components, columns=columns)
            self.dimred_axes_names = [f'tsne component {i + 1}' for i in range(n_components)]
        except Exception as e:
            print(f"TSNE failed: {e}")
            return None

    def cluster(self, best_k=6): # Specific clustering methods - these could potentially be moved to a separate Clustering class/strategy later if needed.
        """Performs KMeans clustering. Returns DataFrame with cluster labels."""
        try:
            columns = self.dimred_axes_names

            X = self.dimred_df[columns]
            X_normalized = preprocessing.scale(X)
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto') # n_init='auto' for sklearn > 1.2
            labels = kmeans.fit_predict(X_normalized) # Use fit_predict
            self.dimred_df['cluster'] = labels
            return self.dimred_df
        except Exception as e:
            print(f"KMeans clustering failed: {e}")
            return None
    

class UMAPReducer(DimReducer): # Subclass for UMAP
    """
    Performs UMAP dimensionality reduction.
    """
    def __init__(self, universe, metadata_df):
        super().__init__(universe, metadata_df)
        self.method = 'UMAP'

    def reduce_dimensions(self, n_components=2, n_neighbors=15, min_dist=0.1, **kwargs) -> pd.DataFrame:
        """
        Performs UMAP dimensionality reduction on the SDA matrix.

        Args:
            sda_matrix (numpy.ndarray): Self-Distance Array matrix.
            n_components (int, optional): Number of UMAP components to compute. Defaults to 2.
            n_neighbors (int, optional):  UMAP parameter, number of neighbors. Defaults to 15.
            min_dist (float, optional): UMAP parameter, minimum distance. Defaults to 0.1.
            **kwargs: Ignored keyword arguments.

        Returns:
            pandas.DataFrame: DataFrame containing the UMAP components.
        """
        
        try:
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
            columns = [f'umap component {i + 1}' for i in range(n_components)]
            components = reducer.fit_transform(self.sda_matrix)
            self.dimred_df = pd.DataFrame(data=components, columns=columns)
            self.dimred_axes_names = [f'umap component {i + 1}' for i in range(n_components)]
            self.components = components
        except Exception as e:
            print(f"UMAP failed: {e}")
            return None

    def cluster(self,min_samples=10, min_cluster_size=200): # Specific HDBSCAN + UMAP - Consider if this belongs here or in a separate Clustering class/strategy.
        """Performs HDBSCAN clustering on UMAP reduced data. Returns DataFrame with cluster labels."""
        try:
            labels = hdbscan.HDBSCAN(min_samples = min_samples, min_cluster_size = min_cluster_size).fit_predict(self.components)
            self.dimred_df['cluster'] = labels
        except Exception as e:
            print(f"HDBSCAN+UMAP clustering failed: {e}")
            return None