"""Small deterministic exploratory experiments with a local, auditable CLI.

All generated outputs are AI-assisted exploratory aids. Instructor approval is
not inferred from successful numerical validation. No output is copied into a
manuscript or treated as independently authored work.
"""

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import matplotlib
import numpy as np

from . import matrices
from .convergence import simulate, validate_distribution
from .diagnostics import diagnose_chain, validate_stochastic
from .eigen_analysis import spectral_summary, stationary_distribution
from .visualization import render_figures


ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
MODELS = {
    "four_room": matrices.four_room,
    "uniform_network": matrices.uniform_network,
    "two_cluster": matrices.two_cluster,
    "bottleneck_network": matrices.bottleneck_network,
    "hub_spoke": matrices.hub_spoke,
    "asymmetric_network": matrices.asymmetric_network,
}
EXPERIMENTS = {"baseline", "starting_distributions", "network_structures", "spectral_gap_limits"}
SUMMARY_FIELDS = [
    "case_id", "start_id", "model", "n", "parameters", "p0", "stationary", "eigenvalues",
    "slem", "gap", "metric", "epsilon", "max_steps", "status", "convergence_time",
    "iterations", "initial_distance", "final_distance", "stationary_residual_l1",
    "irreducible", "aperiodic", "period", "academic_use",
]


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} must use 1–80 letters, digits, underscores or hyphens and start with a letter/digit")
    return value


def _keys(obj, allowed, name):
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(obj) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def _starting_vector(spec, matrix, stationary):
    _keys(spec, {"id", "vector", "kind", "node"}, "start")
    _identifier(spec.get("id"), "start id")
    n = matrix.shape[0]
    if "vector" in spec:
        if "kind" in spec or "node" in spec:
            raise ValueError("A start must specify either vector or kind, not both")
        p0 = spec["vector"]
    elif spec.get("kind") == "point":
        node = spec.get("node", 0)
        if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < n:
            raise ValueError("Point-start node must be a valid integer index")
        p0 = np.zeros(n)
        p0[node] = 1.0
    elif spec.get("kind") == "uniform":
        if "node" in spec:
            raise ValueError("node is only valid for a point start")
        p0 = np.ones(n) / n
    elif spec.get("kind") == "stationary":
        if "node" in spec or stationary is None:
            raise ValueError("Stationary start needs a uniquely calculated stationary vector and no node")
        p0 = stationary.copy()
    else:
        raise ValueError("A start needs a vector or kind point/uniform/stationary")
    return validate_distribution(p0, n)


def prepare_config(config, include_optional=False):
    """Validate the entire selected design before any output directories exist."""
    _keys(config, {
        "schema_version", "experiment", "description", "metric", "epsilon", "max_steps",
        "cases", "optional_cases", "selected_traces", "gap_match_tolerance",
    }, "configuration")
    if config.get("schema_version") != 1 or isinstance(config.get("schema_version"), bool):
        raise ValueError("schema_version must be 1")
    if config.get("experiment") not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {sorted(EXPERIMENTS)}")
    metric = config.get("metric", "l1")
    if metric not in {"l1", "tv"}:
        raise ValueError("metric must be l1 or tv")
    epsilon = _positive_finite(config.get("epsilon", 0.001), "epsilon")
    max_steps = config.get("max_steps", 100000)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if "gap_match_tolerance" in config:
        _positive_finite(config["gap_match_tolerance"], "gap_match_tolerance")
    cases = config.get("cases")
    optional = config.get("optional_cases", [])
    if not isinstance(cases, list) or not cases or not isinstance(optional, list):
        raise ValueError("cases must be a nonempty list and optional_cases must be a list")
    selected = config.get("selected_traces", [])
    if not isinstance(selected, list) or not all(isinstance(v, str) for v in selected):
        raise ValueError("selected_traces must be a list of case_id:start_id strings")
    prepared, seen, trace_ids = [], set(), set()
    # Validate optional schema too, while constructing only requested cases.
    for index, case in enumerate(cases + optional):
        _keys(case, {"id", "model", "parameters", "starts", "epsilon"}, "case")
        case_id = _identifier(case.get("id"), "case id")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        model = case.get("model")
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        parameters = case.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        try:
            matrix = validate_stochastic(MODELS[model](**parameters))
        except TypeError as error:
            raise ValueError(f"Invalid parameters for {case_id}: {error}") from error
        diagnostic = diagnose_chain(matrix)
        stationary = stationary_distribution(matrix) if diagnostic.irreducible else None
        spectrum = spectral_summary(matrix)
        case_epsilon = _positive_finite(case.get("epsilon", epsilon), f"epsilon for {case_id}")
        starts = case.get("starts")
        if not isinstance(starts, list) or not starts:
            raise ValueError("Each case needs a nonempty starts list")
        start_vectors = {}
        for start in starts:
            p0 = _starting_vector(start, matrix, stationary)
            start_id = start["id"]
            if start_id in start_vectors:
                raise ValueError(f"Duplicate start id in {case_id}: {start_id}")
            start_vectors[start_id] = p0
            trace_ids.add(f"{case_id}:{start_id}")
        if index < len(cases) or include_optional:
            prepared.append({
                "case_id": case_id, "model": model, "parameters": parameters,
                "matrix": matrix, "diagnostic": diagnostic, "spectrum": spectrum,
                "stationary": stationary, "starts": start_vectors,
                "epsilon": case_epsilon, "metric": metric, "max_steps": max_steps,
            })
    if not set(selected) <= trace_ids:
        raise ValueError(f"Unknown selected trace IDs: {sorted(set(selected) - trace_ids)}")
    return prepared


def calculate_cases(prepared):
    """Calculate all results without writing files; rejected chains are diagnosed."""
    rows, traces, details = [], {}, []
    for case in prepared:
        matrix, diagnostic, spectrum = case["matrix"], case["diagnostic"], case["spectrum"]
        stationary = case["stationary"]
        details.append({
            "case_id": case["case_id"], "model": case["model"], "parameters": case["parameters"],
            "transition_matrix": matrix, "starts": case["starts"], "diagnostics": diagnostic,
            "stationary": stationary, "spectrum": spectrum,
        })
        for start_id, p0 in case["starts"].items():
            row = {
                "case_id": case["case_id"], "start_id": start_id, "model": case["model"],
                "n": matrix.shape[0], "parameters": case["parameters"], "p0": p0.tolist(),
                "stationary": None if stationary is None else stationary.tolist(),
                "eigenvalues": _jsonable(spectrum.eigenvalues), "slem": spectrum.slem, "gap": spectrum.gap,
                "metric": case["metric"], "epsilon": case["epsilon"], "max_steps": case["max_steps"],
                "irreducible": diagnostic.irreducible, "aperiodic": diagnostic.aperiodic,
                "period": diagnostic.period, "academic_use": "exploratory_unapproved",
                "stationary_residual_l1": None if stationary is None else float(np.abs(stationary @ matrix - stationary).sum()),
            }
            if not diagnostic.ergodic:
                row.update(status="diagnostic_nonergodic", convergence_time=None, iterations=0, initial_distance=None, final_distance=None)
            else:
                result = simulate(matrix, p0, epsilon=case["epsilon"], max_steps=case["max_steps"], metric=case["metric"])
                row.update(
                    status=result.status, convergence_time=result.convergence_time,
                    iterations=result.iterations, initial_distance=float(result.distances[0]),
                    final_distance=float(result.distances[-1]),
                )
                traces[f"{case['case_id']}:{start_id}"] = {
                    "case_id": case["case_id"], "start_id": start_id, "p0": p0.tolist(),
                    "distances": result.distances, "distributions": result.distributions,
                    "epsilon": case["epsilon"], "metric": case["metric"],
                    "convergence_time": result.convergence_time,
                }
            rows.append(row)
    return rows, traces, details


def find_gap_matches(rows, tolerance):
    """Find different-network pairs under shared n, start, metric and threshold.

    The absolute tolerance is explicit; a candidate pair does not establish a
    causal effect. Actual starting vectors must agree, not merely their labels.
    """
    tolerance = _positive_finite(tolerance, "gap matching tolerance")
    pairs = []
    eligible = [r for r in rows if r["status"] == "converged"]
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            if left["case_id"] == right["case_id"]:
                continue
            if (left["n"], left["metric"], left["epsilon"], left["start_id"]) != (right["n"], right["metric"], right["epsilon"], right["start_id"]):
                continue
            if not np.allclose(left["p0"], right["p0"], rtol=0, atol=1e-12):
                continue
            difference = abs(left["gap"] - right["gap"])
            if difference <= tolerance:
                pairs.append({
                    "left": f"{left['case_id']}:{left['start_id']}",
                    "right": f"{right['case_id']}:{right['start_id']}",
                    "left_gap": left["gap"], "right_gap": right["gap"], "absolute_gap_difference": difference,
                    "tolerance": tolerance, "left_convergence_time": left["convergence_time"],
                    "right_convergence_time": right["convergence_time"], "metric": left["metric"],
                    "epsilon": left["epsilon"], "n": left["n"], "p0": left["p0"],
                    "different_times": left["convergence_time"] != right["convergence_time"],
                })
    return pairs


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(config_path, run_id=None, output_root=None, include_optional=False, command=None):
    """Run a validated small design and export data, figures and provenance."""
    config_path = Path(config_path).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    prepared = prepare_config(config, include_optional=include_optional)
    now = datetime.now(timezone.utc)
    run_id = _identifier(run_id or f"{config['experiment']}-{now.strftime('%Y%m%dT%H%M%S%fZ')}", "run id")
    output_root = Path(output_root).resolve() if output_root else ROOT / "outputs"
    data_dir, figure_dir, log_dir = [output_root / kind / run_id for kind in ("data", "figures", "logs")]
    if any(path.exists() for path in (data_dir, figure_dir, log_dir)):
        raise FileExistsError(f"Run {run_id!r} already has output; choose a fresh --run-id")
    rows, traces, details = calculate_cases(prepared)
    tolerance = config.get("gap_match_tolerance", 0.01)
    pairs = find_gap_matches(rows, tolerance) if config["experiment"] == "network_structures" else []
    for path in (data_dir, figure_dir, log_dir):
        path.mkdir(parents=True, exist_ok=False)

    with (data_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical_json(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    with (data_dir / "traces.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "start_id", "t", "distance", "distribution", "metric", "epsilon"])
        writer.writeheader()
        for trace in traces.values():
            for t, distance in enumerate(trace["distances"]):
                writer.writerow({
                    "case_id": trace["case_id"], "start_id": trace["start_id"], "t": t,
                    "distance": float(distance), "distribution": canonical_json(trace["distributions"][t]),
                    "metric": trace["metric"], "epsilon": trace["epsilon"],
                })
    _write_json(data_dir / "case_details.json", details)
    if config["experiment"] == "network_structures":
        _write_json(data_dir / "gap_matches.json", {"absolute_tolerance": tolerance, "pairs": pairs, "match_count": len(pairs)})
    render_figures(rows, traces, figure_dir, config["experiment"], config.get("metric", "l1"), config.get("selected_traces"))

    converged = sum(r["status"] == "converged" for r in rows)
    limited = sum(r["status"] == "iteration_limit" for r in rows)
    rejected = sum(r["status"] == "diagnostic_nonergodic" for r in rows)
    report = [
        "# Exploratory run receipt", "", "## Material Passport", "",
        f"- Run ID: `{run_id}`", f"- Started (UTC): {now.isoformat()}",
        "- Origin: AI-generated exploratory software and outputs.",
        "- Student independent verification: pending.", "- Instructor approval for assessed use: pending.",
        "- Method: deterministic row-vector probability propagation; no random walkers or RNG.",
        "- Publication/delivery: local files only; nothing inserted into the manuscript.", "",
        "## Execution", "", f"Configuration: `{config_path}`",
        f"Cases/starts: {len(rows)}; converged: {converged}; iteration limit: {limited}; diagnosed nonergodic: {rejected}.",
        "The status 'converged' refers to the configured strict numerical threshold for that starting vector; it is not a worst-case mixing-time claim.", "",
        "| Case | Start | Metric | ε | Gap | Status | First t < ε |", "|---|---|---|---|---|---|---|",
    ]
    report.extend(f"| {r['case_id']} | {r['start_id']} | {r['metric']} | {r['epsilon']:g} | {r['gap']:.8g} | {r['status']} | {r['convergence_time'] if r['convergence_time'] is not None else '—'} |" for r in rows)
    if config["experiment"] == "network_structures":
        report.extend(["", "## Similar-gap candidate search", "", f"Absolute gap tolerance: {tolerance:g}. Pairs must share matrix size, actual starting vector, start label, metric and ε."])
        report.append(f"Found {len(pairs)} candidate pairs; {sum(p['different_times'] for p in pairs)} have different measured times. See `gap_matches.json`. This search is descriptive and does not isolate every structural difference." if pairs else "No suitable matched pairs were found in this configured grid.")
    report.extend(["", "## Next verification", "", "Independently check selected transitions, spectrum, stationary residuals and first-passage rows. Form a student-authored explanatory question before adding further experiments. Do not copy these results or plots into assessed work without instructor approval.", "", "Log plots use a display floor equal to one tenth of the smallest selected epsilon. Distances below that floor, including zero, are displayed at the floor with downward triangles; their actual values remain in traces.csv. Starts already below threshold at t=0 remain in the legend without a plotted trajectory. The numeric floor is printed on each trace figure. Displayed starting vectors use three significant digits and an approximation mark; complete vectors remain in the exported data. Summary and trajectory data are never rounded or clipped by these display rules. Optional cases are excluded unless explicitly requested.", ""])
    (log_dir / "run_report.md").write_text("\n".join(report), encoding="utf-8")
    _write_json(log_dir / "config.json", config)
    artifacts = sorted([*data_dir.glob("*"), *figure_dir.glob("*"), log_dir / "run_report.md", log_dir / "config.json"])
    source_dir = Path(__file__).parent
    source_paths = [*source_dir.glob("*.py"), *(ROOT / name for name in (
        "run_experiment.py", "requirements.txt", "requirements-lock.txt", "pyproject.toml",
    ))]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(source_paths) if p.is_file()}
    if command is None:
        command = [sys.executable, "-m", "network_models.experiments", "--config", str(config_path),
                   "--run-id", run_id, "--output-root", str(output_root)]
        if include_optional:
            command.append("--include-optional")
    provenance = {
        "schema_version": 1, "run_id": run_id, "experiment": config["experiment"],
        "started_utc": now.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "working_directory": str(Path.cwd()), "config_path": str(config_path),
        "config_file_sha256": hashlib.sha256(raw_config).hexdigest(),
        "config_canonical_sha256": hashlib.sha256(canonical_json(config).encode()).hexdigest(),
        "config": config, "include_optional": include_optional,
        "academic_use": "exploratory_unapproved", "student_authored": False,
        "instructor_approval": None, "deterministic": True, "random_seed": None,
        "row_stochastic": True, "update_rule": "p_next = p @ P", "stopping_rule": "distance < epsilon; t=0 checked",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "source_sha256": source_hashes,
        "artifact_sha256": {str(p.relative_to(output_root)): sha256_file(p) for p in artifacts},
        "counts": {"total": len(rows), "converged": converged, "iteration_limit": limited, "diagnostic_nonergodic": rejected},
        "gap_match_count": len(pairs), "output_root": str(output_root),
        "reproducibility_note": "Numerical tables/traces repeat exactly on the same software/runtime. Timestamps, paths and run IDs differ between runs; byte equality of plot files across Matplotlib versions is not promised.",
    }
    _write_json(log_dir / "provenance.json", provenance)
    return {"run_id": run_id, "rows": rows, "traces": traces, "pairs": pairs, "provenance": provenance, "data_dir": data_dir, "figure_dir": figure_dir, "log_dir": log_dir}


def main(argv=None, command=None):
    parser = argparse.ArgumentParser(description="Run an AI-assisted exploratory Markov-chain experiment (not approved for submission).")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", help="Fresh output name; existing run folders are never overwritten")
    parser.add_argument("--output-root", type=Path, help="Default: repository outputs directory")
    parser.add_argument("--include-optional", action="store_true", help="Include separately listed optional extension cases")
    args = parser.parse_args(argv)
    if command is None:
        command = [sys.executable, "-m", "network_models.experiments", *(sys.argv[1:] if argv is None else argv)]
    try:
        result = run_experiment(args.config, args.run_id, args.output_root, args.include_optional, command=command)
    except (ValueError, TypeError, OSError, KeyError, ArithmeticError) as error:
        parser.exit(2, f"Experiment not completed: {error}\n")
    print(f"Exploratory run: {result['run_id']}")
    print("Not approved for assessed use. No manuscript content was generated.")
    print("case_id,start_id,gap,status,convergence_time")
    for row in result["rows"]:
        print(f"{row['case_id']},{row['start_id']},{row['gap']:.8g},{row['status']},{row['convergence_time']}")
    print(f"Data: {result['data_dir']}\nFigures: {result['figure_dir']}\nProvenance: {result['log_dir'] / 'provenance.json'}")
    if result["provenance"]["experiment"] == "network_structures":
        print(f"Similar-gap candidates: {len(result['pairs'])} (explicit tolerance in gap_matches.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
