import numpy as np
from scipy.stats import entropy, gaussian_kde


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
