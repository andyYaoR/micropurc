"""Run-archive round-trip: writer -> loader -> analysis helpers -> HTML."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from micropurc.config import EstimationConfig
from micropurc.network import Network
from micropurc.reporting import (
    RunWriter,
    environment_info,
    load_run,
    next_run_dir,
    render_batch_report,
    render_fit_report,
)


@pytest.fixture()
def grid_net() -> Network:
    return Network.grid(3, 3, K=2)


def _write_run(tmp_path, net, *, with_obs=True):
    """Write a complete run archive on a 3x3 grid: config, data, one record, result."""
    scale_m = np.ones(net.num_links)
    beta_hat = np.array([0.5, 0.2])
    writer = RunWriter(
        tmp_path,
        "unit",
        config={"estimation": EstimationConfig(max_iterations=7), "seed": 3},
    )
    data = {
        "edges": net.edges,
        "Z": net.Z,
        "A": sp.csc_matrix(net.A),
        "scale_m": scale_m,
    }
    if with_obs:
        rows, cols = [0, 0, 1, 2, 2], [0, 1, 2, 0, 3]
        y = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(3, net.num_links))
        data.update(
            y=y,
            origins=np.array([0, 0, 0]),
            dests=np.array([8, 8, 8]),
        )
    writer.snapshot_data(**data)
    writer.add_record("task_0", beta_hat=beta_hat, converged_legacy=True)
    writer.finalize(
        result={
            "beta_hat": beta_hat,
            "se_hat": np.array([0.05, 0.02]),
            "converged": True,
            "iterations": 7,
            "Q_history": np.array([1e-2, 1e-4, 1e-7]),
        }
    )
    return writer


class TestRunWriter:
    def test_auto_increment_never_overwrites(self, tmp_path):
        d0 = next_run_dir(tmp_path, "m")
        d0.mkdir()
        d1 = next_run_dir(tmp_path, "m")
        d1.mkdir()
        d2 = next_run_dir(tmp_path, "m")
        assert (d0.name, d1.name, d2.name) == ("m", "m~00", "m~01")

    def test_config_snapshot_serializes_dataclasses(self, tmp_path, grid_net):
        writer = _write_run(tmp_path, grid_net)
        cfg = json.loads((writer.run_dir / "config.json").read_text())
        assert cfg["estimation"]["max_iterations"] == 7
        assert cfg["estimation"]["__dataclass__"] == "EstimationConfig"
        assert cfg["seed"] == 3

    def test_environment_and_compute_info(self, tmp_path, grid_net):
        writer = _write_run(tmp_path, grid_net)
        env = json.loads((writer.run_dir / "environment.json").read_text())
        comp = json.loads((writer.run_dir / "compute.json").read_text())
        assert env["cpu_count"] >= 1 and env["numpy"]
        assert comp["peak_rss_mb"] > 0 and comp["status"] == "completed"

    def test_hash_reference_mode_for_large_data(self, tmp_path, grid_net):
        writer = RunWriter(tmp_path, "big", config={})
        out = writer.snapshot_data(Z=grid_net.Z, max_full_bytes=0)
        ref = json.loads(out.read_text())
        assert ref["mode"] == "hash_reference"
        assert len(ref["arrays"]["Z"]["sha256"]) == 64

    def test_record_exists_for_resume(self, tmp_path, grid_net):
        writer = _write_run(tmp_path, grid_net)
        assert writer.record_exists("task_0")
        assert not writer.record_exists("task_1")

    def test_resume_does_not_rewrite_config(self, tmp_path, grid_net):
        writer = _write_run(tmp_path, grid_net)
        before = (writer.run_dir / "config.json").read_text()
        resumed = RunWriter(tmp_path, "unit", resume=writer.run_dir)
        resumed.add_record("task_1", beta_hat=np.zeros(2))
        assert (writer.run_dir / "config.json").read_text() == before
        assert resumed.record_exists("task_0") and resumed.record_exists("task_1")


class TestLoaderAndHelpers:
    def test_round_trip_result_and_records(self, tmp_path, grid_net):
        writer = _write_run(tmp_path, grid_net)
        run = load_run(writer.run_dir)
        np.testing.assert_allclose(run.beta_hat, [0.5, 0.2])
        np.testing.assert_allclose(run.se_hat, [0.05, 0.02])
        recs = run.records()
        assert set(recs) == {"task_0"}
        assert bool(recs["task_0"]["converged_legacy"])

    def test_network_reconstruction(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        net2 = run.network()
        assert net2.num_links == grid_net.num_links
        np.testing.assert_allclose(net2.Z, grid_net.Z)
        assert (net2.A != grid_net.A).nnz == 0

    def test_predicted_flow_conserves_unit_demand(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        flows = run.predicted_flow([(0, 8)])
        x = flows[(0, 8)]
        net = run.network()
        resid = net.A @ x
        expect = np.zeros(net.num_nodes)
        expect[0], expect[8] = -1.0, 1.0
        np.testing.assert_allclose(resid, expect, atol=1e-8)

    def test_active_subnetwork_carries_the_flow(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        sub = run.active_subnetwork((0, 8))
        x = run.predicted_flow([(0, 8)])[(0, 8)]
        # active links (dual-slack test) must carry at least 99% of the flow mass
        assert x[sub["links"]].sum() >= 0.99 * x.sum()

    def test_compare_routes_bounds(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        rows = run.compare_routes(trips=[0, 1, 2])
        assert [r["trip"] for r in rows] == [0, 1, 2]
        for r in rows:
            assert 0.0 <= r["flow_share_on_route"] <= 1.0 + 1e-12
            assert r["obs_links_carrying_flow"] <= r["n_obs_links"]

    def test_od_resolution_validation(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        with pytest.raises(ValueError, match="exactly one"):
            run.predicted_flow([(0, 8)], trips=[0])
        with pytest.raises(ValueError, match="exactly one"):
            run.predicted_flow()


class TestHtmlReports:
    def test_fit_report(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        out = render_fit_report(run, param_names=["b0", "b1"])
        html = out.read_text()
        assert "b0" in html and "0.500000" in html
        assert "Q-history" in html and "peak RSS" in html
        assert "sha256" not in html  # full snapshot stores arrays inline, so no digests

    def test_batch_report_gates(self, tmp_path, grid_net):
        run = load_run(_write_run(tmp_path, grid_net).run_dir)
        out = render_batch_report(
            run,
            title="unit batch",
            tables=[("T", ["a", "b"], [[1, 2.0]])],
            gates=[("g1", True), ("g2", False)],
        )
        html = out.read_text()
        assert "1 GATE(S) FAILED" in html and "unit batch" in html


def test_environment_info_is_json_serializable():
    json.dumps(environment_info())
