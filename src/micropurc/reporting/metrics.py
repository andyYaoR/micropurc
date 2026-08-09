r"""Prediction and goodness-of-fit metrics for the microPURC route-choice model.

The forward solve returns a per-trip link-flow vector :math:`x` whose entries are
the model's predicted link-choice probabilities, since
:math:`\mathbb{E}[y] = x` under the Markov route sampler. The canonical residual
is :math:`y - x`, the observed route indicator minus the predicted flow. These
functions are pure and dataset-agnostic; they turn :math:`x` and an observed
route into per-trip prediction scores:

- :func:`route_overlap` -- predicted flow mass on, and coverage of, the observed
  route.
- :func:`path_loglik` -- Markov flow-decomposition log-likelihood of the observed
  path, :math:`\sum_{v} \log\big(x_e / \sum_{e' \in \mathrm{out}(v)} x_{e'}\big)`.
- :func:`route_attr_total` -- route total of a link attribute,
  :math:`\mathrm{attr}^{\top} w`, weighted either by the predicted flow :math:`x`
  or by the observed indicator :math:`y`.
- :func:`brier_score` and :func:`auc_roc` -- calibration and discrimination of
  the predicted link probabilities over a pooled set of (trip, link) instances.

:func:`build_out_edges` and :func:`order_route_edges` are helpers for the path
log-likelihood.
"""

from __future__ import annotations

import math

import numpy as np


def route_overlap(x: np.ndarray, obs_links: np.ndarray, flow_tol: float = 1e-8) -> dict:
    """Measures predicted flow mass on, and coverage of, a trip's observed route.

    Args:
        x: Predicted link-flow vector, shape ``(L,)`` (link-choice probabilities).
        obs_links: Distinct link ids on the observed route, shape ``(m,)``; a
            repeated id contributes its flow once per occurrence.
        flow_tol: Flow above which a link counts as predicted-used.

    Returns:
        Dict with ``n_obs_links``, ``n_pred_links``, ``obs_links_carrying_flow``,
        ``flow_on_route``, ``flow_share_on_route`` (mass on observed links / total
        predicted mass, in [0, 1]; NaN if the prediction is all zero), and
        ``min_flow_on_route``.
    """
    x = np.asarray(x, dtype=float)
    obs = np.asarray(obs_links, dtype=int).ravel()
    on_route = x[obs]
    total = float(x.sum())
    return {
        "n_obs_links": int(obs.size),
        "n_pred_links": int((x > flow_tol).sum()),
        "obs_links_carrying_flow": int((on_route > flow_tol).sum()),
        "flow_on_route": float(on_route.sum()),
        "flow_share_on_route": float(on_route.sum() / total) if total > 0 else float("nan"),
        "min_flow_on_route": float(on_route.min()) if obs.size else float("nan"),
    }


def route_attr_total(attr: np.ndarray, weights: np.ndarray) -> float:
    r"""Computes the route total of a link attribute.

    The total is :math:`\sum_e \mathrm{attr}_e w_e`. Pass the predicted flow
    ``x`` as weights for the predicted total, or the observed link-count vector
    ``y`` for the observed total.

    Args:
        attr: Per-link attribute values, shape ``(L,)``.
        weights: Per-link weights (predicted flow ``x`` or observed counts ``y``),
            shape ``(L,)``.

    Returns:
        The weighted attribute total.
    """
    return float(np.asarray(attr, dtype=float) @ np.asarray(weights, dtype=float))


def build_out_edges(tail: np.ndarray, num_nodes: int) -> list[np.ndarray]:
    """Groups link ids by their tail node.

    Args:
        tail: Tail node of each link, shape ``(L,)``.
        num_nodes: Number of nodes ``V``.

    Returns:
        A list of length ``V``; entry ``v`` is the array of link ids with
        ``tail == v``.
    """
    tail = np.asarray(tail, dtype=int).ravel()
    out: list[list[int]] = [[] for _ in range(num_nodes)]
    for e, v in enumerate(tail):
        out[int(v)].append(e)
    return [np.array(lst, dtype=int) for lst in out]


def order_route_edges(
    obs_links: np.ndarray,
    origin: int,
    dest: int,
    tail: np.ndarray,
    head: np.ndarray,
) -> list[int]:
    """Orders an unordered observed link set into a path from origin to dest.

    Walks from ``origin`` following observed links, consuming each at most once,
    which is the traversal order :func:`path_loglik` needs. The walk stops on
    reaching ``dest`` or when no remaining link leaves the current node, so an
    observed set that fails to form one simple path yields a prefix.

    Args:
        obs_links: Link ids on the observed route (unordered), shape ``(m,)``.
        origin: Origin node id.
        dest: Destination node id.
        tail: Tail node of each link, shape ``(L,)``.
        head: Head node of each link, shape ``(L,)``.

    Returns:
        Link ids in traversal order (possibly a prefix if the set is not a path).
    """
    tail = np.asarray(tail, dtype=int)
    head = np.asarray(head, dtype=int)
    remaining = set(int(e) for e in np.asarray(obs_links).ravel())
    ordered: list[int] = []
    node = int(origin)
    while remaining:
        nxt = next((e for e in remaining if int(tail[e]) == node), None)
        if nxt is None:
            break
        ordered.append(nxt)
        remaining.discard(nxt)
        node = int(head[nxt])
        if node == int(dest):
            break
    return ordered


def path_loglik(
    x: np.ndarray,
    ordered_edges: list[int] | np.ndarray,
    tail: np.ndarray,
    out_edges: list[np.ndarray],
    eps: float = 1e-300,
) -> tuple[float, int]:
    r"""Scores an observed path by the Markov flow-decomposition log-likelihood.

    At each visited node :math:`v = \mathrm{tail}[e]` the model leaves along link
    :math:`e` with probability
    :math:`x_e / \sum_{e' \in \mathrm{out}(v)} x_{e'}`, so

    .. math::

        \ell = \sum_{e \in \mathrm{path}}
        \log \frac{x_e}{\sum_{e' \in \mathrm{out}(v)} x_{e'}} .

    A node whose outgoing mass falls to ``eps`` or below contributes
    :math:`\log(\mathrm{eps})`, which keeps the sum finite for a path the model
    assigns no flow to.

    Args:
        x: Predicted link-flow vector, shape ``(L,)``.
        ordered_edges: Observed links in traversal order (see
            :func:`order_route_edges`).
        tail: Tail node of each link, shape ``(L,)``.
        out_edges: Per-node outgoing-link ids from :func:`build_out_edges`.
        eps: Floor for the outgoing-mass denominator and the probability, guarding
            ``log(0)``.

    Returns:
        A tuple ``(loglik, n_steps)`` with the summed log-probability and the number
        of path steps scored.
    """
    x = np.asarray(x, dtype=float)
    tail = np.asarray(tail, dtype=int)
    loglik = 0.0
    n = 0
    for e in ordered_edges:
        e = int(e)
        v = int(tail[e])
        denom = float(x[out_edges[v]].sum())
        p = x[e] / denom if denom > eps else 0.0
        loglik += math.log(max(p, eps))
        n += 1
    return loglik, n


def brier_score(x: np.ndarray, y: np.ndarray) -> float:
    r"""Computes the Brier score, the mean squared error of the predictions.

    .. math:: \mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2

    with predicted probability :math:`x_i \in [0, 1]` and binary outcome
    :math:`y_i \in \{0, 1\}`. The Brier score is a strictly proper scoring rule
    (uniquely minimised in expectation by the true probabilities) and is bounded
    in :math:`[0, 1]`; lower is better. It stays finite when :math:`x_i = 0` for
    an event that occurs, where the per-instance cost reaches its maximum of 1.

    For microPURC each instance ``i`` is a (trip, link) pair: ``x_i`` is the
    predicted link flow (already a probability via the unit-demand box
    ``0 <= x <= 1``) and ``y_i = 1`` iff link ``i`` lies on the observed route.
    No link/logit transform is applied, so the score is well defined for the
    quadratic-perturbation model's exact-zero flows.

    Args:
        x: Predicted probabilities, shape ``(N,)``.
        y: Binary outcomes in ``{0, 1}``, shape ``(N,)``.

    Returns:
        The mean squared error (Brier score).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((x - y) ** 2))


def _mid_ranks(x: np.ndarray) -> np.ndarray:
    """Returns 1-based ascending ranks of ``x``, ties sharing their mid-rank.

    The stable mergesort keeps equal values in input order, so a tie block is
    contiguous in ``xs`` and can be assigned its mid-rank in one sweep.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0  # mid-rank of the tie block
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def auc_roc(x: np.ndarray, y: np.ndarray) -> float:
    r"""Computes the area under the ROC curve via the Mann-Whitney U statistic.

    .. math:: \mathrm{AUC} = \Pr(x_{+} > x_{-}) + \tfrac{1}{2}\Pr(x_{+} = x_{-})
        = \frac{S_{+} - n_{+}(n_{+} + 1)/2}{n_{+}\, n_{-}}

    where all ``N`` instances are ranked by ``x`` ascending, :math:`S_{+}` is the
    sum of the mid-ranks (ties averaged) of the :math:`n_{+}` positive instances
    and :math:`n_{-}` is the number of negatives. Equivalently, the probability
    that a randomly chosen positive is scored above a randomly chosen negative.

    The AUC is rank-based, hence invariant to any monotone transform of ``x``,
    which frees it from any probability-scale or logit assumption and suits the
    quadratic-perturbation model. It is also a function of ``(x, y)`` alone, so
    it applies to a moment or M-estimator. It measures discrimination and is by
    construction blind to calibration.

    Args:
        x: Scores, e.g. predicted link flows, shape ``(N,)``.
        y: Binary outcomes in ``{0, 1}``, shape ``(N,)``.

    Returns:
        The AUC in ``[0, 1]`` (0.5 = chance), or NaN if either class is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _mid_ranks(x)
    s_pos = float(ranks[pos].sum())
    return float((s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
