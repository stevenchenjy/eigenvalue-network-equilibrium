# Experiment A: four-room parameter sweep

AI-generated exploratory design. No outputs are approved for assessed use. The mathematical relationships supplied in the project prompt are validation references, not new discoveries. The student's own hypothesis and interpretation are still pending.

Question: how do cross-group probability, SLEM, spectral gap and first-passage convergence time change together in the controlled four-room family?

- Input: `config.json`, seven supplied c values, fixed p₀=[1,0,0,0].
- Independent variable: c. Controls: four nodes, construction rule, initial vector, deterministic propagation, L1 distance, strict D(t)<0.001, t=0 check, maximum 100000 updates.
- Calculations: stochastic and graph diagnostics; numerical spectrum; stationary residual; every distribution and distance until the first threshold crossing.
- Working expectation for validation: compare with the analytical references supplied by the user. Independently derive and check these before treating them as understood mathematics.
- Output: summary.csv, traces.csv, case_details.json, four plots, config snapshot, run report and SHA-256 provenance.
- Validation: automated reference cases, check the row before crossing and the crossing row, and independently calculate selected first updates.
- Limitations: this highly symmetric family does not establish a universal finite-time gap formula.
- Follow-up: hold P fixed and change p₀ in Experiment B.

From the project root, after installing the dependencies:

```sh
python run_experiment.py --config experiments/baseline/config.json --run-id baseline-study
```

Use a new run ID each time. Read `outputs/logs/<run-id>/run_report.md` and `provenance.json` for actual execution status; the existence of this design is not evidence of a completed experiment. Graphs use selected c values for legibility, while the CSV files include every configured case.

Trace plots use a labeled display floor of one tenth of the smallest selected epsilon. Values below it are shown at the floor with downward triangles; the markers are display bounds, not the actual distances. Starts below threshold at t=0 remain legend-only. Raw values are unchanged in traces.csv.

Plotted starting-vector labels use three significant digits and an approximation mark (≈). Complete vectors remain in the data; display rounding does not alter any calculation.
