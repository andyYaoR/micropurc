"""Core estimation.

The estimator and its supporting numerical machinery: the perturbed-utility
fixed-point solver (:class:`~micropurc.core.estimator.MicroPURCEstimator`),
the range-space projection, sandwich and cluster-robust covariance, active-set
detection, convergence diagnostics, and the lazy design representation.
"""

from .covariance import cluster_robust_covariance, sandwich_covariance
from .estimator import MicroPURCEstimator
from .lazy_design import LazyDesign

__all__ = [
    "MicroPURCEstimator",
    "sandwich_covariance",
    "cluster_robust_covariance",
    "LazyDesign",
]
