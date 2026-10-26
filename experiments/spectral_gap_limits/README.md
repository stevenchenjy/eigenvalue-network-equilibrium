# Experiment D: controlled follow-up scaffold

AI-generated exploratory design, not assessed evidence. The student's observation, explanatory hypothesis and interpretation remain blank until actual evidence is independently reviewed.

Core question: how does changing the error threshold affect measured first passage while P and its spectrum remain fixed? Use the fixed four-room matrix at c=0.10 and two starts A/C, with L1 thresholds 0.01, 0.001 and 0.0001. Each starting vector has its own threshold comparison. The metric, matrix and iteration cap remain fixed. These are deliberately different thresholds and must not be silently pooled as equal experimental conditions.

```sh
python run_experiment.py --config experiments/spectral_gap_limits/config.json --run-id thresholds-study
```

Validate strict inequality against the raw trace, including the row immediately before each crossing. The threshold comparison is a proposed follow-up, not proof that an unexplained anomaly has occurred. Add a student-authored question and prediction before adopting it as part of the investigation.

Optional cases study c above 0.5, including c=1. They are excluded from the core command. Only after core experiments and the convergence assumptions have been studied:

```sh
python run_experiment.py --config experiments/spectral_gap_limits/config.json --run-id negative-modes-study --include-optional
```

The periodic c=1 case is diagnosed and marked `diagnostic_nonergodic`; the main convergence simulator is not used for that case. It receives no finite convergence-time claim. This optional command does not itself visualize indefinitely alternating trajectories; a separately justified, finite propagation exercise would be needed for that purpose.

Expected artifacts: threshold-specific tables/traces and plots, graph diagnostics for optional cases, configuration and provenance. Actual output status appears only in `outputs/logs/<run-id>/run_report.md`. Follow the protocol Observation → Question → Controlled follow-up → Student-authored evidence-based explanation. Do not force anomalies or remove inconvenient data.

Trace plots use a labeled display floor of one tenth of the smallest selected epsilon. Values below it are shown at the floor with downward triangles; the markers are display bounds, not the actual distances. Starts below threshold at t=0 remain legend-only. Raw values are unchanged in traces.csv.

Plotted starting-vector labels use three significant digits and an approximation mark (≈). Complete vectors remain in the data; display rounding does not alter any calculation.
