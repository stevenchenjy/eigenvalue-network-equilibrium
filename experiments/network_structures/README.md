# Experiment C: small network families

AI-generated exploratory design. Assessed use and student interpretation remain pending.

Question: does the relationship in the baseline extend across different reversible constructions, and do similar gaps correspond to different measured times under shared controls?

All default networks have six nodes. Their transparent construction rules are implemented in `src/network_models/matrices.py`:

- `uniform_network`: each transition selects a node uniformly, including the current node.
- `two_cluster`: two equal groups; choose the same or other group using c, then select uniformly within it. This is the baseline rule with three rooms per group.
- `bottleneck_network`: two weighted cliques with self-loops and one symmetric weighted bridge, then row normalization. The bridge weight is an edge weight, not the per-row cross-group probability c.
- `hub_spoke`: random walk on a star mixed with a hold probability (`laziness`) to ensure aperiodicity.

The grid changes c, bridge weight, or laziness within its own family. Each of the 13 core matrices is run from point mass at node 0, point mass at node 1, uniform, and its own stationary vector: 52 core comparisons. Node 0 is the hub and node 1 a leaf in the star family, making this an additional within-matrix initial-state control. Node roles vary across families; point-start comparisons are documented choices, not role-matched topology comparisons. Uniform vectors and stationary vectors are separate controls when stationary probabilities are nonuniform. Independently chosen hypotheses are still pending.

Outputs include construction parameters, actual P and p₀, π, complex-valued spectra, residuals, graph diagnostics, traces and convergence times. Candidate matches require absolute gap difference ≤0.01, equal n, metric, epsilon, start label, and actual p₀. `gap_matches.json` lists all candidates and flags different measured times. An empty search is explicitly reported. Approximate gap matching does not control all other mathematical properties and does not establish causation.

```sh
python run_experiment.py --config experiments/network_structures/config.json --run-id structures-study
```

The optional asymmetric ring is disabled by default. Its directional bias introduces a nonreversible model; study the additional caveats in [numerical_conventions.md](../../docs/numerical_conventions.md) first. This circulant ring can have complex eigenvalues but does not represent every nonreversible chain or demonstrate general nonnormal transients. The optional command is:

```sh
python run_experiment.py --config experiments/network_structures/config.json --run-id structures-optional --include-optional
```

Validation requirements: confirm row sums and graph assumptions for every matrix; verify detailed balance independently before classifying an unfamiliar construction; inspect every matched candidate's actual inputs. This grid does not prove a universal theorem. Read the actual local receipt under `outputs/logs/<run-id>/`; this design alone is not evidence that any pair was found.

Follow-up: choose a candidate based on recorded evidence, formulate an explanatory question, and vary a single plausible factor. Retain inconvenient or null comparisons.

Trace plots use a labeled display floor of one tenth of the smallest selected epsilon. Values below it are shown at the floor with downward triangles; the markers are display bounds, not the actual distances. Starts below threshold at t=0 remain legend-only. Raw values are unchanged in traces.csv.

Plotted starting-vector labels use three significant digits and an approximation mark (≈). Complete vectors remain in the data; display rounding does not alter any calculation.
