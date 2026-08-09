"""Sandwich covariance estimation and cluster-robust standard errors."""

from __future__ import annotations

import numpy as np


def sandwich_covariance(
    H: np.ndarray,
    u_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the sandwich variance and its standard errors.

    .. math::

        \mathrm{cov} = H^{-1} S H^{-1},
        \qquad
        S = \sum_i u_i u_i^{\top},

    where each per-trip score :math:`u_i = Z_{\mathrm{act}}^{\top} \hat P
    \big(\mathrm{scale} \odot (y_i - \hat x_i)\big)` is a :math:`K`-vector, so
    the meat accumulates as a sum of rank-one terms.

    Args:
        H: Accumulated Hessian :math:`\sum_i H_i`, shape (K, K).
        u_vectors: Stacked score vectors :math:`u_i`, shape (N, K).

    Returns:
        cov: Covariance matrix, shape (K, K).
        se: Standard errors, shape (K,).
    """
    S = u_vectors.T @ u_vectors

    H_inv = np.linalg.inv(H)
    cov = H_inv @ S @ H_inv

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return cov, se


def cluster_robust_covariance(
    H: np.ndarray,
    u_vectors: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the cluster-robust sandwich variance and its standard errors.

    Scores are summed within each cluster before the meat is formed, so
    within-cluster dependence is carried through to the variance:

    .. math::

        S_{\mathrm{cl}} = \frac{J}{J-1} \sum_{j=1}^{J}
        \Big(\sum_{i \in j} u_i\Big)
        \Big(\sum_{i \in j} u_i\Big)^{\top},
        \qquad
        \mathrm{cov} = H^{-1} S_{\mathrm{cl}} H^{-1},

    over :math:`J` clusters. The CR1 finite-cluster scaling :math:`J/(J-1)` is
    included here, so a caller that applies its own would double-count it.

    Both ``H`` and the meat are accumulated over observations, so the returned
    covariance is already scaled as a sampling variance and needs no further
    division by the sample size.

    Wald intervals built from these standard errors take their critical value
    from :math:`t_{J-1}`.

    Args:
        H: Accumulated Hessian, shape (K, K).
        u_vectors: Stacked u_i vectors, shape (N, K).
        cluster_ids: Cluster (e.g., user) ID per observation, shape (N,).

    Returns:
        cov: Cluster-robust covariance, shape (K, K).
        se: Cluster-robust standard errors, shape (K,).

    Raises:
        ValueError: If cluster_ids and u_vectors disagree in length, or if there
            are fewer than two clusters (the CR1 correction is undefined).
    """
    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape[0] != u_vectors.shape[0]:
        raise ValueError(
            f"cluster_ids has length {cluster_ids.shape[0]} but u_vectors has "
            f"{u_vectors.shape[0]} rows; they must be aligned per observation."
        )

    unique_clusters = np.unique(cluster_ids)
    n_clusters = unique_clusters.size
    if n_clusters < 2:
        raise ValueError(f"cluster-robust covariance needs at least 2 clusters, got {n_clusters}.")

    K = H.shape[0]
    S = np.zeros((K, K), dtype=float)

    for g in unique_clusters:
        mask = cluster_ids == g
        u_g = u_vectors[mask].sum(axis=0)
        S += np.outer(u_g, u_g)

    S *= n_clusters / (n_clusters - 1.0)

    H_inv = np.linalg.inv(H)
    cov = H_inv @ S @ H_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return cov, se
