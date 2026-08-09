"""Prediction metrics: worked examples with the expected value derived by hand.

Each assertion carries the arithmetic it checks, so a failure points at the
metric definition that changed.
"""

import math

import numpy as np
import pytest

from micropurc.reporting.metrics import (
    auc_roc,
    brier_score,
    build_out_edges,
    order_route_edges,
    path_loglik,
    route_attr_total,
    route_overlap,
)

# Tiny fixed network: nodes {0,1,2}; e0=0->1, e1=1->2, e2=0->2.
TAIL = np.array([0, 1, 0])
HEAD = np.array([1, 2, 2])
V = 3


def test_route_overlap_all_mass_on_route():
    x = np.array([0.5, 0.5, 0.0])
    r = route_overlap(x, obs_links=np.array([0, 1]))
    assert r["n_obs_links"] == 2
    assert r["obs_links_carrying_flow"] == 2
    assert r["n_pred_links"] == 2
    assert math.isclose(r["flow_share_on_route"], 1.0)
    assert math.isclose(r["min_flow_on_route"], 0.5)


def test_route_overlap_partial_mass_off_route():
    # half the mass goes on the direct link e2, which is not on the observed route
    x = np.array([0.5, 0.5, 0.5])
    r = route_overlap(x, obs_links=np.array([0, 1]))
    assert math.isclose(r["flow_share_on_route"], 1.0 / 1.5)


def test_route_overlap_zero_prediction_is_nan():
    r = route_overlap(np.zeros(3), obs_links=np.array([0, 1]))
    assert math.isnan(r["flow_share_on_route"])


def test_route_attr_total_predicted_and_observed():
    attr = np.array([2.0, 3.0, 5.0])
    assert math.isclose(route_attr_total(attr, np.array([0.5, 0.5, 0.0])), 2.5)
    assert math.isclose(route_attr_total(attr, np.array([1.0, 1.0, 0.0])), 5.0)  # observed y


def test_build_out_edges():
    out = build_out_edges(TAIL, V)
    assert set(out[0].tolist()) == {0, 2}
    assert out[1].tolist() == [1]
    assert out[2].tolist() == []


def test_order_route_edges_from_unordered_set():
    ordered = order_route_edges(np.array([1, 0]), origin=0, dest=2, tail=TAIL, head=HEAD)
    assert ordered == [0, 1]


def test_path_loglik_deterministic_path_is_zero():
    out = build_out_edges(TAIL, V)
    x = np.array([1.0, 1.0, 0.0])  # no branching mass -> every step prob 1
    ll, n = path_loglik(x, [0, 1], TAIL, out)
    assert n == 2
    assert math.isclose(ll, 0.0, abs_tol=1e-12)


def test_path_loglik_branching_node():
    out = build_out_edges(TAIL, V)
    x = np.array([0.5, 1.0, 0.5])  # at node 0, mass splits 0.5/0.5 between e0 and e2
    ll, n = path_loglik(x, [0, 1], TAIL, out)
    assert n == 2
    # step 1: log(0.5/(0.5+0.5)) = log(0.5); step 2: log(1.0) = 0
    assert math.isclose(ll, math.log(0.5), rel_tol=1e-9)


def test_brier_score_perfect_and_known():
    assert math.isclose(brier_score([1.0, 0.0, 1.0], [1, 0, 1]), 0.0)
    # two 0.5 forecasts, one hit one miss: mean((0.5-1)^2,(0.5-0)^2) = 0.25
    assert math.isclose(brier_score([0.5, 0.5], [1, 0]), 0.25)
    # bounded: the worst case x=0 for y=1 costs 1 per instance, so the score stays finite
    assert math.isclose(brier_score([0.0], [1]), 1.0)


def test_auc_roc_separation_and_ties():
    # perfect separation: every positive scored above every negative
    assert math.isclose(auc_roc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
    # perfectly wrong ordering: positives scored below negatives
    assert math.isclose(auc_roc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
    # all scores tied -> pure chance, AUC = 0.5 (mid-ranks)
    assert math.isclose(auc_roc([0.4, 0.4, 0.4, 0.4], [1, 0, 1, 0]), 0.5)
    # a used link with predicted 0 flow ranks below the considered (positive-flow) link
    assert math.isclose(auc_roc([0.0, 0.3], [1, 0]), 0.0)
    # one class empty -> NaN
    assert math.isnan(auc_roc([0.2, 0.5], [1, 1]))


def test_brier_auc_match_sklearn():
    """Oracle check: brier_score and auc_roc equal sklearn's on adversarial data.

    The draws deliberately contain ties, exact zeros, and exact ones, which are
    the inputs where rank handling and clipping conventions diverge.
    """
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(20, 300))
        x = np.round(rng.random(n), 2)  # ties
        x[rng.random(n) < 0.3] = 0.0  # exact zeros (off-support)
        x[rng.random(n) < 0.1] = 1.0  # exact ones
        y = (rng.random(n) < 0.5).astype(int)
        if y.sum() in (0, n):
            continue
        assert math.isclose(brier_score(x, y), skm.brier_score_loss(y, x), abs_tol=1e-12)
        assert math.isclose(auc_roc(x, y), skm.roc_auc_score(y, x), abs_tol=1e-12)
