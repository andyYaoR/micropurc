"""OD demand distributions for simulation and empirical data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import Network


@dataclass
class ODDistribution:
    """Distribution over origin-destination pairs.

    Attributes:
        origins: Origin node indices (internal), shape (M,).
        dests: Destination node indices (internal), shape (M,).
        weights: Sampling probabilities, shape (M,), normalized on
            construction, so a caller may pass raw counts.
    """

    origins: np.ndarray
    dests: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes and normalize the weights to a probability vector."""
        if self.origins.shape != self.dests.shape:
            raise ValueError("origins and dests must have the same shape.")
        if self.origins.shape != self.weights.shape:
            raise ValueError("origins and weights must have the same shape.")
        total = float(self.weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights must sum to a positive finite number.")
        self.weights = self.weights / total

    @property
    def size(self) -> int:
        """Number of OD pairs in the distribution."""
        return int(self.origins.shape[0])

    def sample(
        self,
        rng: np.random.Generator,
        n_trips: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n_trips`` OD pairs with replacement, proportional to the weights.

        Returns the sampled origins and destinations, each of shape
        ``(n_trips,)``.
        """
        idx = rng.choice(self.size, size=n_trips, replace=True, p=self.weights)
        return self.origins[idx], self.dests[idx]


def od_from_csv(
    network: Network,
    trips_csv: str,
    *,
    origin_col: str = "origin",
    dest_col: str = "destination",
    weight_col: str | None = "sampling_weight",
    use_node_index: bool = True,
) -> ODDistribution:
    """Build an ODDistribution from a CSV file.

    Args:
        network: Network for mapping raw node IDs.
        trips_csv: Path to CSV with OD demands.
        origin_col: Column name for origin node IDs.
        dest_col: Column name for destination node IDs.
        weight_col: Column holding the weights. When it is ``None`` or absent
            from the file, the OD pairs are weighted equally.
        use_node_index: Map the raw IDs through ``network.node_index``. With
            this off, or with no index on the network, the IDs are cast to int
            and used as internal indices.

    Returns:
        The origin-destination distribution parsed from the file.

    Raises:
        ValueError: If the origin or destination column is absent.
    """
    df = pd.read_csv(trips_csv)

    if origin_col not in df.columns or dest_col not in df.columns:
        raise ValueError(
            f"origin_col='{origin_col}' or dest_col='{dest_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    origin_raw = df[origin_col].to_numpy()
    dest_raw = df[dest_col].to_numpy()

    if use_node_index and network.node_index is not None:
        origins = network.node_index.map_array(origin_raw)
        dests = network.node_index.map_array(dest_raw)
    else:
        origins = origin_raw.astype(int)
        dests = dest_raw.astype(int)

    if weight_col is not None and weight_col in df.columns:
        weights = df[weight_col].to_numpy(float)
    else:
        weights = np.ones_like(origins, dtype=float)

    return ODDistribution(origins=origins, dests=dests, weights=weights)


def od_uniform_all_pairs(
    network: Network,
    *,
    allow_self_trips: bool = False,
) -> ODDistribution:
    """Uniform distribution over all OD pairs.

    Args:
        network: Network whose nodes define the OD universe.
        allow_self_trips: Keep the pairs whose origin equals their destination.

    Returns:
        The uniform distribution over the retained OD pairs.
    """
    V = network.num_nodes
    origins, dests = np.meshgrid(np.arange(V, dtype=int), np.arange(V, dtype=int), indexing="ij")
    origins = origins.ravel()
    dests = dests.ravel()

    if not allow_self_trips:
        mask = origins != dests
        origins = origins[mask]
        dests = dests[mask]

    weights = np.ones_like(origins, dtype=float)
    return ODDistribution(origins=origins, dests=dests, weights=weights)
