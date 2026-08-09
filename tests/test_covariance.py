"""Sandwich and cluster-robust covariance against hand-written reference formulas.

Every assertion here is an exact algebraic identity, so the tolerances sit at
the level of floating-point noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from micropurc.core.covariance import cluster_robust_covariance, sandwich_covariance

K = 3


def _spd(rng, k=K):
    """A random SPD bread matrix; the k*I shift keeps it comfortably invertible."""
    A = rng.normal(size=(k, k))
    return A.T @ A + k * np.eye(k)


def _reference_eq21(H, u, cluster_ids):
    """The cluster-robust sandwich written out directly.

    Sums the scores within each cluster, builds the meat from the outer products
    of those cluster sums, applies the J/(J-1) finite-cluster factor, and
    sandwiches the result between two copies of H^-1.
    """
    groups = np.unique(cluster_ids)
    S = np.zeros((H.shape[0], H.shape[0]))
    for g in groups:
        u_g = u[cluster_ids == g].sum(axis=0)
        S += np.outer(u_g, u_g)
    S *= groups.size / (groups.size - 1.0)
    H_inv = np.linalg.inv(H)
    return H_inv @ S @ H_inv


def test_matches_eq21_including_cr1_factor():
    """cluster_robust_covariance reproduces the reference sandwich, factor included."""
    rng = np.random.default_rng(0)
    H = _spd(rng)
    u = rng.normal(size=(60, K))
    cluster_ids = rng.integers(0, 7, size=60)

    cov, se = cluster_robust_covariance(H, u, cluster_ids)

    expected = _reference_eq21(H, u, cluster_ids)
    assert np.allclose(cov, expected, rtol=0, atol=1e-14)
    assert np.allclose(se, np.sqrt(np.diag(expected)), rtol=0, atol=1e-14)


def test_cr1_factor_is_actually_applied():
    """Dropping the J/(J-1) factor would shrink every standard error by sqrt((J-1)/J)."""
    rng = np.random.default_rng(1)
    H = _spd(rng)
    u = rng.normal(size=(40, K))
    cluster_ids = rng.integers(0, 5, size=40)
    n_clusters = np.unique(cluster_ids).size

    cov, _ = cluster_robust_covariance(H, u, cluster_ids)

    H_inv = np.linalg.inv(H)
    S_raw = np.zeros((K, K))
    for g in np.unique(cluster_ids):
        u_g = u[cluster_ids == g].sum(axis=0)
        S_raw += np.outer(u_g, u_g)
    uncorrected = H_inv @ S_raw @ H_inv

    assert np.allclose(cov, uncorrected * n_clusters / (n_clusters - 1.0), atol=1e-14)
    assert not np.allclose(cov, uncorrected)


def test_singleton_clusters_rescale_the_naive_sandwich():
    """One observation per cluster: each cluster sum is a single score.

    The cluster meat then coincides with the naive meat term by term, so the
    only difference between the two covariances is the J/(J-1) factor.
    """
    rng = np.random.default_rng(2)
    H = _spd(rng)
    u = rng.normal(size=(25, K))
    cluster_ids = np.arange(25)

    cov_cl, _ = cluster_robust_covariance(H, u, cluster_ids)
    cov_naive, _ = sandwich_covariance(H, u)

    assert np.allclose(cov_cl, cov_naive * 25 / 24.0, rtol=0, atol=1e-14)


@pytest.mark.parametrize("trips_per_cluster", [2, 5, 10])
def test_identical_rows_within_cluster_give_exact_se_ratio(trips_per_cluster):
    """Identical rows within a cluster inflate the SE by exactly sqrt(T * J/(J-1)).

    When all T rows of a cluster coincide, the cluster meat is T^2 times the
    per-cluster outer product while the naive meat is T times it, so the ratio
    of standard errors is the same in every coordinate. This is the closed form
    behind the repeated-trips study, where a traveler's T habitual trips are
    identical draws.
    """
    rng = np.random.default_rng(3)
    n_clusters = 8
    H = _spd(rng)
    per_cluster = rng.normal(size=(n_clusters, K))

    u = np.repeat(per_cluster, trips_per_cluster, axis=0)
    cluster_ids = np.repeat(np.arange(n_clusters), trips_per_cluster)

    _, se_cl = cluster_robust_covariance(H, u, cluster_ids)
    _, se_naive = sandwich_covariance(H, u)

    expected = np.sqrt(trips_per_cluster * n_clusters / (n_clusters - 1.0))
    assert np.allclose(se_cl / se_naive, expected, rtol=0, atol=1e-12)


def test_accumulated_convention_equals_the_papers_normalized_form():
    """The accumulated convention already equals the per-observation-averaged form.

    Write H_tilde = H/N and S_tilde = S/N for the averaged bread and meat. The
    averaged sandwich expands to
    N^-1 (N H^-1) (S/N) (N H^-1) = H^-1 S H^-1, so accumulating H and the meat
    over observations needs no further division by N.
    """
    rng = np.random.default_rng(4)
    n_obs = 50
    H = _spd(rng)
    u = rng.normal(size=(n_obs, K))
    cluster_ids = rng.integers(0, 6, size=n_obs)
    n_clusters = np.unique(cluster_ids).size

    cov, _ = cluster_robust_covariance(H, u, cluster_ids)

    H_tilde = H / n_obs
    S_tilde = np.zeros((K, K))
    for g in np.unique(cluster_ids):
        u_g = u[cluster_ids == g].sum(axis=0)
        S_tilde += np.outer(u_g, u_g)
    S_tilde *= (n_clusters / (n_clusters - 1.0)) / n_obs

    H_tilde_inv = np.linalg.inv(H_tilde)
    paper_form = (H_tilde_inv @ S_tilde @ H_tilde_inv) / n_obs

    assert np.allclose(cov, paper_form, rtol=1e-12, atol=0)


def test_single_cluster_raises():
    """One cluster leaves J/(J-1) undefined, so the call must fail loudly."""
    rng = np.random.default_rng(5)
    H = _spd(rng)
    u = rng.normal(size=(10, K))

    with pytest.raises(ValueError, match="at least 2 clusters"):
        cluster_robust_covariance(H, u, np.zeros(10, dtype=int))


def test_misaligned_cluster_ids_raise():
    """Cluster ids are matched to scores by position, so a length mismatch is fatal."""
    rng = np.random.default_rng(6)
    H = _spd(rng)
    u = rng.normal(size=(10, K))

    with pytest.raises(ValueError, match="must be aligned"):
        cluster_robust_covariance(H, u, np.arange(9))


def test_cluster_ids_need_not_be_integers_or_sorted():
    """Grouping goes through np.unique, so any comparable label in any order works."""
    rng = np.random.default_rng(7)
    H = _spd(rng)
    u = rng.normal(size=(6, K))
    labels = np.array(["b", "a", "b", "c", "a", "c"])

    cov, _ = cluster_robust_covariance(H, u, labels)

    assert np.allclose(cov, _reference_eq21(H, u, labels), rtol=0, atol=1e-14)
