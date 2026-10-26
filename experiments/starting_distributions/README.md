# Experiment B: starting-distribution sensitivity

AI-generated exploratory design, unapproved for assessed use. The student must independently formulate the hypothesis and explain the evidence.

Question: how much can measured convergence vary while the same transition matrix and spectral gap are held fixed?

- Input: P(0.10), and the four supplied starting vectors A=[1,0,0,0], B=[0,1,0,0], C=[0.5,0,0.5,0], D=[0.25,0.25,0.25,0.25].
- Independent variable: starting distribution. Controls: all matrix entries, stationary target, L1 metric, strict threshold 0.001, iteration cap and numerical conventions.
- Calculations: spectrum and stationary distribution of the fixed matrix, complete deterministic traces, first crossing including t=0.
- Working expectations for software checks: compare against user-supplied transition references and equilibrium; no student research hypothesis has been recorded yet.
- Output: one comparative time plot, labeled distance traces, raw traces, summary, explicit p₀ vectors and provenance.
- Validation: compare A/B symmetry, manually check the first update of C, and check D at t=0. Retain all four starts.
- Limitations: these four starts are not a numerical search over the probability simplex and do not establish worst-case mixing time.
- Follow-up questions: which starts contain the between-group imbalance? How could eigenvectors explain this? Which changes would isolate a different mode?

```sh
python run_experiment.py --config experiments/starting_distributions/config.json --run-id starts-study
```

The actual run receipt is `outputs/logs/<run-id>/run_report.md`; do not mark the experiment complete from this design alone. Stationary starts remain in the table at t=0 and appear only in the trace legend. Other distances below the logarithmic display floor (one tenth of the smallest selected epsilon) are shown at the floor with downward triangles. This includes exact zeros and tiny numerical residuals; the markers do not indicate equality with the floor. The numeric floor is printed on the figure, and all raw values remain unchanged in traces.csv.

Plotted starting-vector labels use three significant digits and an approximation mark (≈). Complete vectors remain in the data; display rounding does not alter any calculation.
