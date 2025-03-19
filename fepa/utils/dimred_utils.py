"""
This module contains functions that help with dimensionality reduction and clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def elbow_plot_pca(pca_projection_df, n_components=None, save_path=None):
    """Plots elbow plot for clustering."""
    if n_components is None:
        # Count the no of columns with PC in the name
        n_components = sum([1 for col in pca_projection_df.columns if "PC" in col])
    pc_columns = [f"PC{i + 1}" for i in range(n_components)]
    X = pca_projection_df[pc_columns]
    X_normalized = preprocessing.scale(X)
    sse = []
    title = "Elbow Plot for PCA"
    for k in range(1, 20):
        kmeans = KMeans(
            n_clusters=k, random_state=42, n_init="auto"
        )  # n_init='auto' for sklearn > 1.2
        kmeans.fit(X_normalized)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 20), sse, marker="o", linestyle="-")
    plt.ylabel("Sum of Squared Distances")
    plt.xlabel("Parameter k for KMeans")
    plt.title(title)
    plt.xticks(range(1, 20))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def cluster_pca(pca_projection_df, n_clusters, n_components=None):
    """
    Clusters the PCA projection data using KMeans clustering.
    """
    if n_components is None:
        n_components = sum([1 for col in pca_projection_df.columns if "PC" in col])
    pc_columns = [f"PC{i + 1}" for i in range(n_components)]
    X = pca_projection_df[pc_columns]
    X_normalized = preprocessing.scale(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_normalized)
    pca_projection_df["cluster"] = labels
    return pca_projection_df


def get_projection_df_subset_values(projection_df, key, value):
    """
    Returns a subset of the projection dataframe based on a key-value pair.
    """
    subset_projection_df = projection_df[projection_df[key] == value]
    subset_projection_df_values = subset_projection_df.drop(
        columns=["timestep", "ensemble"]
    ).values
    return subset_projection_df_values


def get_ensemble_center(projection_df, key, value):
    """
    Returns the point closest to the center of the ensemble in the projection dataframe as a row
    """
    ensemble_df = projection_df[projection_df[key] == value]
    PC_cols = [col for col in ensemble_df.columns if "PC" in col]
    # Compute the centroid of numerical columns
    centroid = ensemble_df[PC_cols].mean().values.reshape(1, -1)

    # Compute distances to centroid
    distances = cdist(ensemble_df[PC_cols], centroid, metric="euclidean")

    # Get the index of the row closest to centroid
    closest_index = distances.argmin()

    return ensemble_df.iloc[closest_index]


def make_ensemble_center_df(projection_df, key):
    """
    Returns a dataframe containing the ensemble centers for each unique value in the key column.
    """
    total_ensemble_center_df = pd.DataFrame()

    for value in np.unique(projection_df[key]):
        ensemble_center_df = (
            get_ensemble_center(projection_df=projection_df, key=key, value=value)
            .to_frame()
            .T
        )
        total_ensemble_center_df = pd.concat(
            [total_ensemble_center_df, ensemble_center_df]
        )

    return total_ensemble_center_df
