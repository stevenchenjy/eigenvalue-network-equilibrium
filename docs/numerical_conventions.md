# Numerical conventions — exploratory software reference

AI-generated implementation notes. Not approved for assessed use; not manuscript text. Date: 2026-10-25.

## Inputs and probability propagation

P[i,j] means the probability of next node j given current node i. Use row vectors throughout. Validators return owned arrays and never alter caller input. All numbers must be finite and real. Matrices must be nonempty, square, have entries in [0,1] and have each row sum to 1 within absolute tolerance 1e-12 (relative tolerance zero). Negative probabilities, even small negatives, are rejected. Distributions use the same sum tolerance and a one-dimensional shape. Numerical parameters reject booleans and nonfinite values.

Weighted adjacency construction explicitly divides each row by its positive finite weight sum; this is a documented construction step, not a hidden repair of an invalid stochastic matrix. Simulation does not renormalize each update. Mass drift above 1e-8 or nonfinite states raise an error. Small networks and ordinary thresholds are the intended scope; extremely long runs or ill-conditioned near-reducible cases require independent scrutiny.

## Structural diagnostics

Edges correspond to entries strictly greater than zero; the graph checker does not discard tiny positive edges. Forward and reverse reachability check strong connectivity. For an irreducible chain, the gcd of BFS-depth edge differences gives its period. A chain passes the main gate when irreducible and period 1. Reducible diagnostics report period=null and aperiodic=false: no single global period is established by this implementation. This does not assert that each closed component is periodic.

Main simulations explicitly reject nonergodic inputs. Diagnostic-only output for such a case is not a failed-to-converge trajectory or a convergent result. A finite iteration limit on an eligible chain yields status `iteration_limit` and convergence_time=null. Never substitute the limit for a measured convergence time.

## Stationary vector and spectra

Solve the augmented least-squares system `(P.T-I) @ pi = 0`, `sum(pi)=1`. The solver requires irreducibility to avoid choosing an arbitrary stationary distribution from a reducible system; it permits periodic irreducible chains. It checks numerical rank, finite/nonnegative probabilities, normalization and the maximum absolute residual of `pi @ P - pi` against 1e-10. Tiny negative solver values within that tolerance are clipped, then the vector is renormalized and its residual rechecked. No corresponding repair is applied to the matrix.

A tiny residual alone does not guarantee an accurate stationary vector near reducibility. The augmented system's singular-value condition number must also be at most `1/sqrt(machine_epsilon)` (about 6.71e7 for float64). Larger values raise an explicit ArithmeticError before a convergence claim. This conservative software guard is not a formal forward-error guarantee. For example, c=1e-14 has a structurally positive bridge but is refused by the double-precision solver; the supplied grid c>=.01 is well within the supported range. Very strict thresholds and near-degenerate spectra still require independent analysis or higher precision.

Eigenvalues come from `numpy.linalg.eigvals` in complex arithmetic as needed. Sort by decreasing modulus with real/imaginary tie breakers. Identify an eigenvalue nearest 1 within absolute tolerance 1e-8, remove **exactly one**, and take the maximum remaining modulus. Additional eigenvalues near 1 stay in the SLEM computation. The count within that tolerance is a numerical diagnostic, not proof of exact multiplicity. SLEM above 1 only by eigenvalue tolerance is clipped to 1; a larger violation raises an error. For a one-state matrix, the explicitly chosen empty-spectrum convention is SLEM=0, gap=1.

This gap is the absolute gap `1-SLEM`. It accounts for slow negative or complex modes. An algebraic ordering of real eigenvalues can give a different quantity and is not substituted. Only the baseline range 0<c<0.5 uses the analytical expression gap=2c. For other c use the modulus definition.

These diagnostics do not implement general spectral convergence bounds, eigenvector condition-number analysis, Jordan decompositions, or formal mixing-time proofs. Numerical eigenvalues alone cannot establish a universal finite-time prediction. The optional directed-ring example can be nonreversible yet normal; it is not a demonstration of general nonnormal transients.

## Distances and strict stopping

L1 is `sum(abs(p-pi))`; TV is half this number. The default epsilon is 0.001. Simulation stores the initial state at t=0, checks it first and stops at the first **computed** distance strictly less than epsilon. A zero initial distance has time zero. max_steps counts updates and may be zero. Changing epsilon or metric creates a different configuration and must be labeled in tables and figures.

Floating-point stationary vectors and distances may differ slightly from exact arithmetic. A distance theoretically equal to a threshold may evaluate a little above or below it; the software compares the actual float with `<`, without rounding or an extra stopping tolerance. Boundary tests use equality to the computed value to verify this rule. For evidence close to the threshold, inspect the surrounding distances and independently verify the mathematics. Tests and references are exploratory validation aids, not assessed findings.

Zero distances cannot appear on a logarithmic y-axis. Trace plots use a display floor equal to one tenth of the smallest selected epsilon. Values below that floor, including zero, are drawn at the floor with downward triangles; a numeric footnote states that these are not the actual values. Trajectories already passing at t=0 remain in the legend without plotted points. These display rules keep tiny stationary-solver residuals from determining the logarithmic range. Tabular traces retain every actual value without clipping or rounding. Read the trajectory and summary alongside each plot.

Trace legends format starting vectors to three significant digits with an approximation mark for legibility. case_details.json and the CSVs retain the full input values. Figure export reserves space for notes and includes the complete legend in its bounding box.

## Network controls

The uniform model includes self transitions. The two-cluster model chooses a destination group then chooses a room uniformly. The bottleneck model normalizes symmetric weighted cliques with one bridge. The star is made lazy explicitly. Equal node counts and the same start-node rule are preferred for comparisons; differing size, stationary law or start are possible confounders and must be recorded. Similar gaps are matched by a predeclared absolute tolerance, not by rounding displayed gap values. Matching is descriptive; it does not hold all other variables constant.

No randomness is used. Reproducibility checks compare numerical tables and trajectories in the recorded environment; run timestamps and IDs are expected to differ. Artifact hashes identify exact files. Versions, source hashes, config and output hashes must accompany any retained run.
