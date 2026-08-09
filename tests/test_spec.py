"""Tests for the model-specification schema, DSL, and compiler."""

import numpy as np
import pytest

from micropurc.network import Network
from micropurc.spec import (
    ModelSpec,
    attr,
    beta,
    compile_design,
    log,
    power,
    seg,
    standardize,
)
from micropurc.spec.schema import Segment, Term


def _toy_network() -> Network:
    """A small grid with non-constant, positive named attributes."""
    net = Network.grid(2, 2, K=3)
    L = net.num_links
    rng = np.random.default_rng(0)
    net.Z = np.column_stack(
        [
            np.linspace(1.0, 5.0, L),  # length
            np.linspace(2.0, 6.0, L),  # time
            rng.uniform(0.5, 2.0, L),  # toll
        ]
    )
    net.attribute_names = ["length", "time", "toll"]
    return net


def test_dsl_lowers_to_schema():
    U = beta("time") * attr("time") + beta("cost") * attr("length")
    spec = U.to_spec()
    assert spec.parameters() == ["time", "cost"]
    assert spec.terms[0] == Term(parameter="time", attributes=("time",))
    assert spec.terms[1] == Term(parameter="cost", attributes=("length",))


def test_schema_json_round_trip():
    spec = ModelSpec(
        terms=[
            Term("toll", ("toll",), segment=Segment("income", "<=", 3.0)),
            Term("logdist", ("length",), transform="log"),
            Term("inter", ("toll", "time"), scale=2.0),
        ]
    )
    restored = ModelSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert ModelSpec.from_json(spec.to_json()) == spec


def test_dsl_schema_equivalence():
    U = beta("toll") * attr("toll") * seg("income", "<=", 3.0) + beta("logdist") * log(
        attr("length")
    )
    hand = ModelSpec(
        terms=[
            Term("toll", ("toll",), segment=Segment("income", "<=", 3.0)),
            Term("logdist", ("length",), transform="log"),
        ]
    )
    assert U.to_spec() == hand


def test_compile_plain_attributes_selects_columns():
    net = _toy_network()
    spec = (beta("b_len") * attr("length") + beta("b_toll") * attr("toll")).to_spec()
    cd = compile_design(spec, net)
    assert cd.parameters == ["b_len", "b_toll"]
    assert cd.Z.shape == (net.num_links, 2)
    np.testing.assert_allclose(cd.Z[:, 0], net.Z[:, 0])
    np.testing.assert_allclose(cd.Z[:, 1], net.Z[:, 2])


def test_compile_interaction_and_transforms():
    net = _toy_network()
    spec = (
        beta("inter") * attr("length") * attr("toll")
        + beta("logt") * log(attr("time"))
        + beta("p2") * power(attr("toll"), 2.0)
        + beta("std") * standardize(attr("length"))
    ).to_spec()
    cd = compile_design(spec, net)
    L = net.Z[:, 0]
    T = net.Z[:, 1]
    K = net.Z[:, 2]
    np.testing.assert_allclose(cd.Z[:, 0], L * K)
    np.testing.assert_allclose(cd.Z[:, 1], np.log(T))
    np.testing.assert_allclose(cd.Z[:, 2], K**2)
    np.testing.assert_allclose(cd.Z[:, 3], (L - L.mean()) / L.std())


def test_compile_shared_parameter_sums_terms():
    net = _toy_network()
    spec = (beta("b") * attr("length") + beta("b") * attr("toll")).to_spec()
    cd = compile_design(spec, net)
    assert cd.parameters == ["b"]
    np.testing.assert_allclose(cd.Z[:, 0], net.Z[:, 0] + net.Z[:, 2])


def test_compile_segmentation_variants():
    net = _toy_network()
    spec = (
        beta("toll_lo") * attr("toll") * seg("income", "<=", 2.0)
        + beta("toll_hi") * attr("toll") * seg("income", ">", 2.0)
        + beta("time") * attr("time")
    ).to_spec()
    variant_cov = np.array([[1.0], [3.0]])  # variant 0: income 1, variant 1: income 3
    trip_variant = np.array([0, 1, 0])
    cd = compile_design(
        spec,
        net,
        variant_covariates=variant_cov,
        covariate_names=["income"],
        trip_variant=trip_variant,
    )
    assert cd.z_variants.shape == (2, net.num_links, 3)
    toll = net.Z[:, 2]
    time = net.Z[:, 1]
    # variant 0 (income<=2): toll_lo active, toll_hi zero
    np.testing.assert_allclose(cd.z_variants[0, :, 0], toll)
    np.testing.assert_allclose(cd.z_variants[0, :, 1], 0.0)
    # variant 1 (income>2): toll_hi active, toll_lo zero
    np.testing.assert_allclose(cd.z_variants[1, :, 0], 0.0)
    np.testing.assert_allclose(cd.z_variants[1, :, 1], toll)
    # time non-segmented: present in both
    np.testing.assert_allclose(cd.z_variants[0, :, 2], time)
    np.testing.assert_allclose(cd.z_variants[1, :, 2], time)


def test_segmented_spec_requires_variant_data():
    net = _toy_network()
    spec = (beta("t") * attr("toll") * seg("income", "<=", 2.0)).to_spec()
    with pytest.raises(ValueError):
        compile_design(spec, net)


def test_spec_lazy_matches_explicit_in_estimator():
    """A segmented spec compiled to lazy terms yields the same H, g as explicit Z."""
    import scipy.sparse as sp

    from micropurc.config import ProjectionConfig, SolverConfig
    from micropurc.core.estimator import MicroPURCEstimator
    from micropurc.forward.piqp_solver import PIQPFlowSolver

    net = Network.grid(3, 3, K=3)
    L = net.num_links
    rng = np.random.default_rng(1)
    net.Z = np.column_stack(
        [
            1.0 + rng.uniform(size=L),
            0.5 + rng.uniform(size=L),
            0.2 + rng.uniform(size=L),
        ]
    )
    net.attribute_names = ["length", "time", "toll"]

    spec = (
        beta("time") * attr("time")
        + beta("toll_lo") * attr("toll") * seg("income", "<=", 2.0)
        + beta("toll_hi") * attr("toll") * seg("income", ">", 2.0)
    ).to_spec()
    variant_cov = np.array([[1.0], [3.0]])
    trip_variant = np.array([0, 1, 0, 1], dtype=np.int64)

    cd_mat = compile_design(
        spec,
        net,
        variant_covariates=variant_cov,
        covariate_names=["income"],
        trip_variant=trip_variant,
        materialize=True,
    )
    cd_lazy = compile_design(
        spec,
        net,
        variant_covariates=variant_cov,
        covariate_names=["income"],
        trip_variant=trip_variant,
        materialize=False,
    )
    assert cd_mat.parameters == cd_lazy.parameters == ["time", "toll_lo", "toll_hi"]

    beta_vec = np.array([0.5, 0.3, 0.4])
    origins = np.zeros(4, dtype=np.int64)
    dests = np.full(4, 8, dtype=np.int64)
    y = sp.csr_matrix((4, L), dtype=float)
    scale_m = net.Z[:, 0].copy()

    def _accum(design_kwargs):
        est = MicroPURCEstimator(
            net,
            PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp")),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        return est._accumulate(beta_vec, y, None, origins=origins, dests=dests, **design_kwargs)

    H_mat, g_mat, _, _ = _accum(
        {"z_variants": cd_mat.z_variants, "trip_variant": cd_mat.trip_variant}
    )
    H_lazy, g_lazy, _, _ = _accum({"lazy_design": cd_lazy.lazy})

    np.testing.assert_allclose(H_lazy, H_mat, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(g_lazy, g_mat, atol=1e-10, rtol=1e-10)


def test_variant_attribute_materializes_per_variant():
    """A variant-varying attribute selects the right row per trip variant."""
    net = _toy_network()
    L = net.num_links
    tt = np.stack([np.full(L, 1.0), np.full(L, 2.0)])  # (2, L): two time-of-day rows
    spec = (beta("tt") * attr("travel_time") + beta("len") * attr("length")).to_spec()
    cd = compile_design(
        spec,
        net,
        variant_attrs={"travel_time": tt},
        variant_attr_index=np.array([0, 1]),
        trip_variant=np.array([0, 1, 0]),
    )
    assert cd.z_variants.shape == (2, L, 2)
    np.testing.assert_allclose(cd.z_variants[0, :, 0], tt[0])  # variant 0 -> row 0
    np.testing.assert_allclose(cd.z_variants[1, :, 0], tt[1])  # variant 1 -> row 1
    # static attribute is identical across variants
    np.testing.assert_allclose(cd.z_variants[0, :, 1], net.Z[:, 0])
    np.testing.assert_allclose(cd.z_variants[1, :, 1], net.Z[:, 0])


def _accum_grid(net, design_kwargs, beta_vec, N=4, dest=8):
    import scipy.sparse as sp

    from micropurc.config import ProjectionConfig, SolverConfig
    from micropurc.core.estimator import MicroPURCEstimator
    from micropurc.forward.piqp_solver import PIQPFlowSolver

    est = MicroPURCEstimator(
        net,
        PIQPFlowSolver(net, net.Z[:, 0].copy(), SolverConfig(backend="piqp")),
        projection_config=ProjectionConfig(mode="robust_pinv"),
    )
    return est._accumulate(
        beta_vec,
        sp.csr_matrix((N, net.num_links), dtype=float),
        None,
        origins=np.zeros(N, dtype=np.int64),
        dests=np.full(N, dest, dtype=np.int64),
        **design_kwargs,
    )


def test_variant_attribute_lazy_matches_explicit():
    """A variant-varying attribute gives identical H, g via lazy and explicit paths."""
    net = Network.grid(3, 3, K=3)
    L = net.num_links
    rng = np.random.default_rng(2)
    net.Z = np.column_stack([1 + rng.uniform(size=L), rng.uniform(size=L), rng.uniform(size=L)])
    net.attribute_names = ["length", "time", "toll"]

    tt = np.stack([0.5 + rng.uniform(size=L), 1.5 + rng.uniform(size=L)])  # (2, L)
    spec = (beta("tt") * attr("travel_time") + beta("len") * attr("length")).to_spec()
    kw = dict(
        variant_attrs={"travel_time": tt},
        variant_attr_index=np.array([0, 1]),
        trip_variant=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    cd_mat = compile_design(spec, net, materialize=True, **kw)
    cd_lazy = compile_design(spec, net, materialize=False, **kw)

    beta_vec = np.array([0.4, 0.3])
    H_mat, g_mat, _, _ = _accum_grid(
        net, {"z_variants": cd_mat.z_variants, "trip_variant": cd_mat.trip_variant}, beta_vec
    )
    H_lazy, g_lazy, _, _ = _accum_grid(net, {"lazy_design": cd_lazy.lazy}, beta_vec)
    np.testing.assert_allclose(H_lazy, H_mat, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(g_lazy, g_mat, atol=1e-10, rtol=1e-10)


def test_variant_attribute_with_segmentation_lazy_matches_explicit():
    """A variant-varying attribute combined with covariate segmentation: lazy == explicit.

    Exercises both lazy-design opcodes in one spec, so an evaluation order that
    works for either alone but mixes them up is caught here.
    """
    net = Network.grid(3, 3, K=3)
    L = net.num_links
    rng = np.random.default_rng(3)
    net.Z = np.column_stack(
        [1 + rng.uniform(size=L), rng.uniform(size=L), 0.5 + rng.uniform(size=L)]
    )
    net.attribute_names = ["length", "time", "toll"]

    tt = np.stack([0.5 + rng.uniform(size=L), 1.5 + rng.uniform(size=L)])  # (2, L)
    spec = (
        beta("tt") * attr("travel_time")
        + beta("toll_lo") * attr("toll") * seg("income", "<=", 2.0)
        + beta("toll_hi") * attr("toll") * seg("income", ">", 2.0)
    ).to_spec()
    kw = dict(
        variant_attrs={"travel_time": tt},
        variant_attr_index=np.array([0, 1]),
        variant_covariates=np.array([[1.0], [3.0]]),
        covariate_names=["income"],
        trip_variant=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    cd_mat = compile_design(spec, net, materialize=True, **kw)
    cd_lazy = compile_design(spec, net, materialize=False, **kw)
    assert cd_mat.parameters == ["tt", "toll_lo", "toll_hi"]

    beta_vec = np.array([0.4, 0.3, 0.2])
    H_mat, g_mat, _, _ = _accum_grid(
        net, {"z_variants": cd_mat.z_variants, "trip_variant": cd_mat.trip_variant}, beta_vec
    )
    H_lazy, g_lazy, _, _ = _accum_grid(net, {"lazy_design": cd_lazy.lazy}, beta_vec)
    np.testing.assert_allclose(H_lazy, H_mat, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(g_lazy, g_mat, atol=1e-10, rtol=1e-10)
