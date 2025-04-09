import numpy as np
from scipy.stats import entropy, gaussian_kde
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr  # Import for Pearson correlation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def js_divergence(
    samples_p: np.ndarray, samples_q: np.ndarray, num_points=1000
) -> float:
    """
    Compute the Jensen-Shannon divergence between two N-dimensional distributions.

    Parameters:
    samples_p (np.ndarray): Samples from the first distribution (shape: [x, N])
    samples_q (np.ndarray): Samples from the second distribution (shape: [y, N])
    num_points (int): Number of evaluation points for estimating distributions

    Returns:
    float: JS divergence
    """
    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError("Both distributions must have the same number of dimensions.")

    N = samples_p.shape[1]  # Dimensionality
    kde_p = gaussian_kde(samples_p.T)
    kde_q = gaussian_kde(samples_q.T)

    # Generate evaluation points
    min_vals = np.minimum(samples_p.min(axis=0), samples_q.min(axis=0))
    max_vals = np.maximum(samples_p.max(axis=0), samples_q.max(axis=0))
    eval_points = np.array(
        [np.linspace(min_vals[i], max_vals[i], num_points) for i in range(N)]
    ).T

    # Evaluate probability densities
    p_pdf = kde_p(eval_points.T) + 1e-10  # Avoid log(0)
    q_pdf = kde_q(eval_points.T) + 1e-10

    m_pdf = 0.5 * (p_pdf + q_pdf)

    # Compute JS divergence
    js_div = 0.5 * (entropy(p_pdf, m_pdf) + entropy(q_pdf, m_pdf))

    return js_div


def calculate_metrics(x_values, y_values):
    """
    Calculate metrics: MUE, RMSE, R2, and Pearson correlation.
    """
    # Calculate Mean Unsigned Error (MUE)
    mue = np.mean(np.abs(x_values - y_values))

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(x_values, y_values))

    # Calculate R squared (R2)
    r2 = r2_score(x_values, y_values)

    # Calculate Pearson Correlation
    pearson_corr, _ = pearsonr(x_values, y_values)

    return {"mue": mue, "rmse": rmse, "r2": r2, "pearson_corr": pearson_corr}


def find_optimal_clusters_elbow(l, max_clusters=10, save_path=None):
    """
    Finds the optimal number of clusters for a univariate distribution using the Elbow Method.

    Args:
        l (list): A list representing the univariate distribution.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        int: The estimated optimal number of clusters.
    """
    data = np.array(l).reshape(-1, 1)  # Reshape for scikit-learn
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    if save_path is not None:
        plt.plot(range(1, max_clusters + 1), wcss, marker="o")
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.xticks(range(1, max_clusters + 1))
        plt.grid(True)
        plt.savefig(save_path)

    # Heuristic approach to find the elbow point (can be subjective)
    # We can look for the point where the percentage decrease in WCSS starts to diminish.
    if len(wcss) < 2:
        return 1
    for i in range(len(wcss) - 1):
        if i > 0:
            improvement = wcss[i - 1] - wcss[i]
            previous_improvement = wcss[i] - wcss[i + 1] if i + 1 < len(wcss) else 0
            if (
                improvement < 0.1 * wcss[0]
            ):  # Consider a threshold for significant improvement
                return i + 1
        elif len(wcss) == 2 and wcss[0] > 0:
            return 2
    return 1  # Default to 1 if no clear elbow is found


def find_optimal_clusters_silhouette(l, max_clusters=10, save_path=None):
    """
    Finds the optimal number of clusters for a univariate distribution using the Silhouette Score.

    Args:
        l (list): A list representing the univariate distribution.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        int: The estimated optimal number of clusters.
    """
    data = np.array(l).reshape(-1, 1)
    silhouette_scores = []
    for n_clusters in range(
        2, max_clusters + 1
    ):  # Silhouette score needs at least 2 clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the Silhouette Scores
    if save_path is not None:
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
        plt.title("Silhouette Score")
        plt.xlabel("Number of clusters")
        plt.ylabel("Average Silhouette Score")
        plt.xticks(range(2, max_clusters + 1))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    # Find the number of clusters with the highest silhouette score
    if silhouette_scores:
        optimal_clusters = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        return optimal_clusters
    else:
        return 1  # If max_clusters is less than 2


def find_optimal_clusters_bic(l, max_components=10, save_path=None):
    """
    Finds the optimal number of clusters for a univariate distribution using BIC.

    Args:
        l (list): A list representing the univariate distribution.
        max_components (int): The maximum number of components (clusters) to consider.

    Returns:
        int: The estimated optimal number of clusters based on BIC.
    """
    data = np.array(l).reshape(-1, 1)
    bic_scores = []
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))

    # Plot BIC scores
    if save_path is not None:
        plt.plot(range(1, max_components + 1), bic_scores, marker="o")
        plt.title("BIC for Gaussian Mixture Model")
        plt.xlabel("Number of components (clusters)")
        plt.ylabel("BIC Score")
        plt.xticks(range(1, max_components + 1))
        plt.grid(True)
        plt.savefig("bic_scores.png")

    # The optimal number of clusters is the one with the lowest BIC score
    optimal_n_components = np.argmin(bic_scores) + 1
    return optimal_n_components
