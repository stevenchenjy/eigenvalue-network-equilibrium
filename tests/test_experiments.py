"""Checks of exported evidence and failure behavior, not research conclusions."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest

from network_models.experiments import find_gap_matches, prepare_config, run_experiment


PROJECT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT / "run_experiment.py"


def small_config():
    return {
        "schema_version": 1, "experiment": "starting_distributions", "metric": "l1",
        "epsilon": 0.001, "max_steps": 100,
        "cases": [{"id": "c10", "model": "four_room", "parameters": {"c": 0.1},
                   "starts": [{"id": "A", "vector": [1, 0, 0, 0]},
                              {"id": "C", "vector": [0.5, 0, 0.5, 0]},
                              {"id": "D", "vector": [0.25, 0.25, 0.25, 0.25]}]}],
    }


def write_config(tmp_path, config=None):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config or small_config()), encoding="utf-8")
    return path


def test_cli_exports_crossing_trace_and_valid_hashes(tmp_path):
    config_path = write_config(tmp_path)
    output_root = tmp_path / "outputs"
    command = [sys.executable, str(WRAPPER), "--config", str(config_path),
               "--run-id", "test-cli", "--output-root", str(output_root)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=45)
    assert process.returncode == 0, process.stderr
    assert "Not approved for assessed use" in process.stdout
    with (output_root / "data/test-cli/summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    by_start = {row["start_id"]: row for row in summary}
    assert int(by_start["A"]["convergence_time"]) == 31
    assert int(by_start["C"]["convergence_time"]) == 1
    assert int(by_start["D"]["convergence_time"]) == 0
    with (output_root / "data/test-cli/traces.csv").open() as stream:
        traces = list(csv.DictReader(stream))
    point_trace = [row for row in traces if row["start_id"] == "A"]
    assert len(point_trace) == 32
    assert float(point_trace[-2]["distance"]) >= 0.001
    assert float(point_trace[-1]["distance"]) < 0.001
    assert int(point_trace[0]["t"]) == 0
    np.testing.assert_allclose(json.loads(point_trace[1]["distribution"]), [.45, .45, .05, .05])
    provenance = json.loads((output_root / "logs/test-cli/provenance.json").read_text())
    assert provenance["academic_use"] == "exploratory_unapproved"
    assert provenance["config_file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert provenance["command"] == command
    assert provenance["source_sha256"]
    for relative_path, expected_hash in provenance["artifact_sha256"].items():
        assert hashlib.sha256((output_root / relative_path).read_bytes()).hexdigest() == expected_hash
    assert len(list((output_root / "figures/test-cli").glob("*.png"))) == 2


def test_identical_design_repeats_tables_and_refuses_overwrite(tmp_path):
    path = write_config(tmp_path)
    output = tmp_path / "output"
    first = run_experiment(path, "first", output)
    second = run_experiment(path, "second", output)
    for filename in ["summary.csv", "traces.csv", "case_details.json"]:
        assert (first["data_dir"] / filename).read_bytes() == (second["data_dir"] / filename).read_bytes()
    original = (first["data_dir"] / "summary.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(path, "first", output)
    assert (first["data_dir"] / "summary.csv").read_bytes() == original


@pytest.mark.parametrize("mutation", ["bad_last_start", "entry_over_one", "unknown_key", "nan_epsilon", "duplicate_case", "bad_optional", "path_run_id"])
def test_invalid_design_has_no_partial_output(tmp_path, mutation):
    config = small_config()
    run_id = "bad"
    if mutation == "bad_last_start":
        config["cases"][0]["starts"].append({"id": "bad", "vector": [1, 1, 0, 0]})
    elif mutation == "entry_over_one":
        vector = [1 + 3e-13, 1e-13, 0, 0]
        assert np.isclose(sum(vector), 1, atol=1e-12, rtol=0)
        config["cases"][0]["starts"].append({"id": "bad", "vector": vector})
    elif mutation == "unknown_key":
        config["epslion"] = .01
    elif mutation == "nan_epsilon":
        config["epsilon"] = float("nan")
    elif mutation == "duplicate_case":
        config["cases"].append(copy.deepcopy(config["cases"][0]))
    elif mutation == "bad_optional":
        config["optional_cases"] = [{"id": "bad_optional", "model": "does_not_exist", "starts": []}]
    else:
        run_id = "../escape"
    path = write_config(tmp_path, config)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        run_experiment(path, run_id, output)
    assert not output.exists()


def test_gap_matching_enforces_tolerance_actual_start_and_metric():
    def row(case_id, gap, time, p0=(1., 0.), epsilon=.001, metric="l1"):
        return dict(case_id=case_id, start_id="point0", n=2, gap=gap, convergence_time=time,
                    p0=list(p0), epsilon=epsilon, metric=metric, status="converged")
    rows = [row("left", .2, 31), row("near", .205, 19), row("far", .22, 16),
            row("other_start", .2, 4, p0=(0., 1.)), row("other_epsilon", .2, 3, epsilon=.01),
            row("other_metric", .2, 2, metric="tv")]
    pairs = find_gap_matches(rows, .01)
    assert len(pairs) == 1
    assert pairs[0]["left"] == "left:point0"
    assert pairs[0]["right"] == "near:point0"
    assert pairs[0]["different_times"] is True
    assert pairs[0]["absolute_gap_difference"] <= .01
    assert find_gap_matches([row("a", .2, 1), row("b", .4, 1)], .01) == []


def test_optional_periodic_case_is_diagnosed_without_false_time(tmp_path):
    config = small_config()
    config["optional_cases"] = [{"id": "periodic", "model": "four_room", "parameters": {"c": 1.},
                                 "starts": [{"id": "A", "vector": [1, 0, 0, 0]}]}]
    assert len(prepare_config(config)) == 1
    assert len(prepare_config(config, include_optional=True)) == 2
    result = run_experiment(write_config(tmp_path, config), "periodic", tmp_path / "out", include_optional=True)
    periodic = [row for row in result["rows"] if row["case_id"] == "periodic"][0]
    assert periodic["status"] == "diagnostic_nonergodic"
    assert periodic["convergence_time"] is None
    assert periodic["period"] == 2
    assert "periodic:A" not in result["traces"]


def test_every_shipped_configuration_validates():
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "experiments").glob("*/config.json"))
    assert len(paths) == 4
    for path in paths:
        config = json.loads(path.read_text())
        assert prepare_config(config)
        assert prepare_config(config, include_optional=True)


def test_cli_reports_numerically_unresolved_stationary_system(tmp_path):
    config = small_config()
    config["cases"][0]["parameters"]["c"] = 1e-10
    path = write_config(tmp_path, config)
    output = tmp_path / "output"
    process = subprocess.run(
        [sys.executable, str(WRAPPER), "--config", str(path),
         "--run-id", "ill-conditioned", "--output-root", str(output)],
        text=True, capture_output=True, timeout=30,
    )
    assert process.returncode == 2
    assert "Experiment not completed" in process.stderr
    assert "ill-conditioned" in process.stderr
    assert "Traceback" not in process.stderr
    assert not output.exists()


def test_launcher_from_other_directory_uses_current_source_and_project_outputs(tmp_path):
    path = write_config(tmp_path)
    other_directory = tmp_path / "different-working-directory"
    other_directory.mkdir()
    run_id = f"test-cwd-{uuid.uuid4().hex}"
    output_dirs = [PROJECT / "outputs" / kind / run_id for kind in ("data", "figures", "logs")]
    assert not any(p.exists() for p in output_dirs)
    command = [sys.executable, str(WRAPPER), "--config", str(path), "--run-id", run_id]
    try:
        process = subprocess.run(command, cwd=other_directory, text=True, capture_output=True, timeout=45)
        assert process.returncode == 0, process.stderr
        assert all(p.is_dir() for p in output_dirs)
        assert not (other_directory / "outputs").exists()
        provenance = json.loads((output_dirs[2] / "provenance.json").read_text())
        assert provenance["output_root"] == str(PROJECT / "outputs")
        assert Path(provenance["working_directory"]).resolve() == other_directory.resolve()
        assert provenance["command"] == command
        for relative_path in ["run_experiment.py", "src/network_models/experiments.py", "requirements-lock.txt", "pyproject.toml"]:
            assert provenance["source_sha256"][relative_path] == hashlib.sha256((PROJECT / relative_path).read_bytes()).hexdigest()
    finally:
        for path in output_dirs:
            if path.exists():
                shutil.rmtree(path)
