"""Exploratory plots only; nothing here exports into the manuscript directory."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NOTICE = "Exploratory • AI-assisted • Not approved for submission"


def _vector_label(vector):
    """Display only: three significant digits, with ≈ supplied by the caller."""
    return "[" + ", ".join(f"{float(value):.3g}" for value in vector) + "]"


def _save(fig, path, note=None):
    fig.text(0.5, 0.015, NOTICE, ha="center", fontsize=8, color="#7b3333")
    if note:
        fig.text(0.02, 0.055, note, fontsize=7, color="0.35", va="bottom")
    # Reserve physical space for every footer line; a long legend must not
    # push tick labels into the notes. Tight bounding boxes also preserve
    # external legends when exporting an unfamiliar, larger configuration.
    footer_margin = 0.055 + (note.count("\n") + 1) * 9 / (72 * fig.get_figheight()) + 0.025 if note else 0.045
    fig.tight_layout(rect=(0, footer_margin, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.15,
                metadata={"Software": "network_models exploratory workflow"})
    plt.close(fig)
    return path


def render_figures(rows, traces, output_dir, experiment, metric, selected_trace_ids=None):
    """Render consistent metrics with an explicitly labeled logarithmic floor.

    Summary tables retain zero values and convergence at t=0. Starts already
    below threshold at t=0 are listed in the legend without plotting their
    floating-point residuals. Other values below one tenth of the smallest
    selected epsilon are displayed at the floor with downward triangles.
    Raw distances are never modified by this plotting convention.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    valid = [r for r in rows if r["status"] == "converged"]
    metric_label = "L1 distance" if metric == "l1" else "Total variation distance"
    epsilons = sorted({r["epsilon"] for r in rows})
    condition = f"{metric_label} < " + (f"{epsilons[0]:g}" if len(epsilons) == 1 else "ε (shown per case)")
    matrix_specs = {(r["model"], json.dumps(r["parameters"], sort_keys=True)) for r in rows}
    same_matrix = len(matrix_specs) <= 1
    limited_count = sum(r["status"] == "iteration_limit" for r in rows)
    nonergodic_count = sum(r["status"] == "diagnostic_nonergodic" for r in rows)
    omitted_note = (f"Summary plots omit {limited_count} iteration-limit and {nonergodic_count} nonergodic rows; see summary.csv."
                    if limited_count or nonergodic_count else None)

    if experiment == "baseline":
        ordered = sorted(valid, key=lambda r: r["parameters"]["c"])
        for xkey, ykey, name, xlabel, ylabel in [
            ("c", "gap", "c_vs_gap", "Cross-group probability c", "Spectral gap (1 − SLEM)"),
            ("c", "convergence_time", "c_vs_convergence", "Cross-group probability c", "First iteration below threshold"),
            ("gap", "convergence_time", "gap_vs_convergence", "Spectral gap (1 − SLEM)", "First iteration below threshold"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            x = [r["parameters"]["c"] if xkey == "c" else r[xkey] for r in ordered]
            ax.plot(x, [r[ykey] for r in ordered], "o-", color="#25678c")
            ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Four-room parameter sweep\n{condition}; p₀=[1, 0, 0, 0]")
            ax.grid(alpha=0.2)
            paths.append(_save(fig, output_dir / f"{name}.png", omitted_note))
    elif experiment in {"starting_distributions", "spectral_gap_limits"}:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = [f"{r['start_id']}\np₀≈{_vector_label(r['p0'])}" if experiment == "starting_distributions" and same_matrix else
                  f"{r['case_id']} / {r['start_id']}\nε={r['epsilon']:g}" for r in valid]
        ax.bar(labels, [r["convergence_time"] for r in valid], color="#25678c")
        if experiment == "starting_distributions":
            title = "Starting distributions with the same matrix" if same_matrix else "Starting-distribution and matrix comparisons"
        else:
            title = "Threshold and start comparisons with the same matrix" if same_matrix else "Threshold, start and optional parameter comparisons"
        ax.set(ylabel="First iteration below threshold", title=f"{title}\n{condition}")
        ax.tick_params(axis="x", labelsize=8 if same_matrix else 7, labelrotation=20 if len(valid) > 6 else 0)
        ax.grid(axis="y", alpha=0.2)
        filename = "starting_distribution_times.png" if experiment == "starting_distributions" else "threshold_comparisons.png"
        paths.append(_save(fig, output_dir / filename, omitted_note))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted({r["model"] for r in valid})
        colors = plt.get_cmap("tab10")
        for index, model in enumerate(models):
            group = [r for r in valid if r["model"] == model]
            for start_index, start_id in enumerate(sorted({r["start_id"] for r in group})):
                points = [r for r in group if r["start_id"] == start_id]
                ax.scatter(
                    [r["gap"] for r in points], [r["convergence_time"] for r in points],
                    color=colors(index), marker=["o", "s", "^", "D"][start_index % 4],
                    label=f"{model} / {start_id}", s=45, alpha=0.85,
                )
        ax.set(xlabel="Spectral gap (1 − SLEM)", ylabel="First iteration below threshold", title=f"Exploratory convergence comparisons\n{condition}")
        ax.grid(alpha=0.2)
        if valid:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        paths.append(_save(fig, output_dir / "gap_vs_convergence.png", omitted_note))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    selected = set(selected_trace_ids or traces)
    thresholds = {trace["epsilon"] for trace_id, trace in traces.items() if trace_id in selected}
    display_floor = min(thresholds) / 10 if thresholds else None
    if display_floor == 0:
        # Only relevant for subnormal custom epsilons whose division underflows.
        display_floor = min(thresholds)
    for trace_id, trace in traces.items():
        if trace_id not in selected:
            continue
        distances = np.asarray(trace["distances"], dtype=float)
        label = f"{trace_id}\np₀≈{_vector_label(trace['p0'])}"
        if trace["convergence_time"] == 0:
            label += "\n(below threshold at t=0)"
            ax.semilogy([], [], marker="o", linestyle="none", label=label)
        else:
            displayed = np.maximum(distances, display_floor)
            line, = ax.semilogy(np.arange(len(distances)), displayed, marker="." if len(distances) < 5 else None, label=label)
            below_floor = distances < display_floor
            if np.any(below_floor):
                ax.scatter(np.flatnonzero(below_floor), displayed[below_floor], marker="v",
                           color=line.get_color(), s=35, zorder=3)
    for threshold in sorted(thresholds):
        ax.axhline(threshold, linestyle="--", color="0.4", linewidth=0.8, label=f"ε={threshold:g}")
    ax.set(xlabel="Iteration t (initial distribution at t=0)", ylabel=metric_label, title="Distance from the stationary distribution")
    if display_floor is not None:
        ax.set_ylim(bottom=display_floor / 1.5 if display_floor / 1.5 > 0 else display_floor)
    ax.grid(alpha=0.2, which="both")
    if traces:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    trace_note = (f"Values below display floor {display_floor:g} are shown at the floor with ▼, not at their actual values."
                  if display_floor is not None else "No selected simulated traces.")
    trace_note += "\nStarts below threshold at t=0 are legend-only; all raw values remain in traces.csv."
    trace_note += "\nLegend vectors are rounded to three significant digits (≈); full vectors remain in the data."
    if limited_count or nonergodic_count:
        trace_note += f"\n{nonergodic_count} nonergodic rows have no simulated trace; {limited_count} iteration-limit traces end at their configured cap."
    paths.append(_save(fig, output_dir / "distance_traces.png", trace_note))
    return paths
