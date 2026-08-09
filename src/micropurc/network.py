"""Network representation with sparse CSC incidence matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class NodeIndex:
    """Mapping between raw node IDs and internal 0..V-1 indices.

    Attributes:
        raw_ids: Raw node IDs in internal-index order, shape (V,).
        id_to_idx: Inverse mapping, raw ID to internal index.
    """

    raw_ids: np.ndarray
    id_to_idx: dict[Any, int]

    @classmethod
    def from_edges(cls, tail_raw: np.ndarray, head_raw: np.ndarray) -> NodeIndex:
        """Build an index from every raw node ID appearing in the edges.

        Internal indices follow the sorted order of the unique raw IDs.
        """
        unique_raw = np.unique(np.concatenate([tail_raw, head_raw]))
        id_to_idx = {rid: i for i, rid in enumerate(unique_raw)}
        return cls(raw_ids=unique_raw, id_to_idx=id_to_idx)

    def map_array(self, arr: np.ndarray) -> np.ndarray:
        """Map an array of raw node IDs to internal indices.

        An ID outside the index raises ``KeyError`` naming the offending ID.
        """
        out = np.empty(arr.shape[0], dtype=int)
        for k, rid in enumerate(arr):
            try:
                out[k] = self.id_to_idx[rid]
            except KeyError:
                raise KeyError(f"Raw node ID {rid!r} not found in NodeIndex")
        return out


@dataclass
class Network:
    """Directed network with sparse incidence and link attributes.

    Attributes:
        nodes: Internal node indices (0..V-1), shape (V,).
        edges: Directed edges in internal indices, shape (L, 2).
        Z: Link attribute matrix, shape (L, K).
        A: Node-link incidence matrix, sparse CSC, shape (V, L).
            Convention: A[i, ell] = -1 (tail), A[j, ell] = +1 (head).
        node_index: Mapping between raw node IDs and internal indices, ``None``
            for a network built without raw IDs.
        attribute_names: Name of each column of Z, length K. Defaults to
            ``attr_0, attr_1, ...`` when not supplied. Model specifications
            reference attributes by these names.
        tail: Tail node indices, shape (L,).
        head: Head node indices, shape (L,).
    """

    nodes: np.ndarray
    edges: np.ndarray
    Z: np.ndarray
    A: sp.csc_matrix
    node_index: NodeIndex | None = None
    attribute_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize ``A`` to CSC form and fill in default attribute names.

        A supplied ``attribute_names`` whose length disagrees with the column
        count of ``Z`` raises ``ValueError``.
        """
        if not sp.issparse(self.A):
            self.A = sp.csc_matrix(self.A)
        elif not sp.isspmatrix_csc(self.A):
            self.A = self.A.tocsc()
        if self.attribute_names is None:
            self.attribute_names = [f"attr_{k}" for k in range(self.Z.shape[1])]
        elif len(self.attribute_names) != self.Z.shape[1]:
            raise ValueError(
                f"attribute_names has {len(self.attribute_names)} entries "
                f"but Z has {self.Z.shape[1]} columns"
            )

    def attribute_index(self, name: str) -> int:
        """Return the column index of ``Z`` holding the named attribute."""
        try:
            return self.attribute_names.index(name)
        except ValueError:
            raise KeyError(f"attribute '{name}' not in {self.attribute_names}") from None

    @property
    def tail(self) -> np.ndarray:
        """Tail node of each link."""
        return self.edges[:, 0]

    @property
    def head(self) -> np.ndarray:
        """Head node of each link."""
        return self.edges[:, 1]

    @property
    def num_links(self) -> int:
        """Number of links (rows of ``Z``)."""
        return self.Z.shape[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes (rows of ``A``)."""
        return self.A.shape[0]

    @property
    def num_attributes(self) -> int:
        """Number of raw link attributes (columns of ``Z``)."""
        return self.Z.shape[1]

    @classmethod
    def grid(cls, rows: int, cols: int, K: int, **kwargs: Any) -> Network:
        """Construct a ``rows`` x ``cols`` grid network with synthetic attributes.

        Every grid neighbour pair contributes two directed links, one per
        direction.

        Args:
            rows: Number of grid rows.
            cols: Number of grid columns.
            K: Number of synthetic attribute columns.
            **kwargs: Only ``rng_seed`` is read, seeding the attribute draw
                (default 42).

        Returns:
            The grid network, with node IDs equal to their internal indices.

        Raises:
            ValueError: If ``rows`` or ``cols`` is not positive.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")

        V = rows * cols
        nodes = np.arange(V, dtype=int)

        def node_id(r: int, c: int) -> int:
            return r * cols + c

        edges_list: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                u = node_id(r, c)
                if c + 1 < cols:
                    v = node_id(r, c + 1)
                    edges_list.append((u, v))
                    edges_list.append((v, u))
                if r + 1 < rows:
                    v = node_id(r + 1, c)
                    edges_list.append((u, v))
                    edges_list.append((v, u))

        edges = np.asarray(edges_list, dtype=int)
        L = edges.shape[0]

        node_index = NodeIndex(
            raw_ids=nodes.copy(),
            id_to_idx={int(rid): int(rid) for rid in nodes},
        )

        row_idx = np.concatenate([edges[:, 0], edges[:, 1]])
        col_idx = np.concatenate([np.arange(L), np.arange(L)])
        data = np.concatenate([np.full(L, -1.0), np.full(L, 1.0)])
        A = sp.csc_matrix((data, (row_idx, col_idx)), shape=(V, L))

        rng_seed = kwargs.get("rng_seed", 42)
        rng = np.random.default_rng(rng_seed)

        # Standardized, independent synthetic attributes: each column is drawn
        # independently and scaled to unit variance, then shifted by its own
        # minimum so every attribute is non-negative. Independent columns give a
        # full-rank, well-conditioned design that can be estimated on directly,
        # and the non-negative shift keeps the forward route-choice flows
        # non-degenerate (a mean-zero design leaves routing costs mixed-sign and
        # yields no valid trips).
        Z = rng.standard_normal((L, K))
        if L > 1 and K > 0:
            std = Z.std(axis=0)
            std[std == 0.0] = 1.0
            Z = (Z - Z.mean(axis=0)) / std
        if L > 0 and K > 0:
            Z = Z - Z.min(axis=0)

        return cls(nodes=nodes, edges=edges, Z=Z, A=A, node_index=node_index)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        *,
        tail_col: str = "init_node",
        head_col: str = "term_node",
        attribute_cols: list[str] | None = None,
    ) -> Network:
        """Construct a Network from a link CSV file, one row per directed link.

        Args:
            csv_path: Path to the link file.
            tail_col: Column holding the raw tail node ID.
            head_col: Column holding the raw head node ID.
            attribute_cols: Columns to load into ``Z``. When omitted, every
                numeric column apart from ``tail_col`` and ``head_col`` is
                taken, in file order.

        Returns:
            The network, with raw node IDs reachable through ``node_index``.

        Raises:
            ValueError: If a tail, head, or attribute column is absent, or if
                no attribute column is left to load.
        """
        df = pd.read_csv(csv_path)

        if tail_col not in df.columns or head_col not in df.columns:
            raise ValueError(
                f"tail_col='{tail_col}' or head_col='{head_col}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        tail_raw = df[tail_col].to_numpy()
        head_raw = df[head_col].to_numpy()

        node_index = NodeIndex.from_edges(tail_raw, head_raw)
        tail = node_index.map_array(tail_raw)
        head = node_index.map_array(head_raw)
        edges = np.column_stack([tail, head]).astype(int)
        L = edges.shape[0]

        V = node_index.raw_ids.shape[0]
        nodes = np.arange(V, dtype=int)

        row_idx = np.concatenate([edges[:, 0], edges[:, 1]])
        col_idx = np.concatenate([np.arange(L), np.arange(L)])
        data = np.concatenate([np.full(L, -1.0), np.full(L, 1.0)])
        A = sp.csc_matrix((data, (row_idx, col_idx)), shape=(V, L))

        if attribute_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            attribute_cols = [c for c in numeric_cols if c not in [tail_col, head_col]]

        if not attribute_cols:
            raise ValueError(
                "No attribute columns selected. Provide attribute_cols explicitly "
                "or ensure there are numeric columns besides tail/head."
            )

        for col in attribute_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Attribute column '{col}' not found in {csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        Z = df[attribute_cols].to_numpy(float)
        return cls(
            nodes=nodes,
            edges=edges,
            Z=Z,
            A=A,
            node_index=node_index,
            attribute_names=list(attribute_cols),
        )

    @classmethod
    def from_dense(
        cls,
        A_dense: np.ndarray,
        edges: np.ndarray,
        Z: np.ndarray,
        node_index: NodeIndex | None = None,
    ) -> Network:
        """Construct a network from a dense node-link incidence matrix.

        ``A_dense`` is converted to CSC and its row count fixes the node set.
        """
        V = A_dense.shape[0]
        nodes = np.arange(V, dtype=int)
        A = sp.csc_matrix(A_dense)
        return cls(nodes=nodes, edges=edges, Z=Z, A=A, node_index=node_index)
