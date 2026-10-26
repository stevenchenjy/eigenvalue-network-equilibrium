# Coding workspace instructions

Use English for user-facing writing, documentation and agent instructions. Read ACADEMIC_INTEGRITY.md before making changes.

- Maintain AI-assisted exploratory code, tests, configurations and software documentation here. Keep assessed manuscript writing and learning notes in their separate research workspace.
- Never write or rewrite graded mathematical explanations, proofs, findings or conclusions, or automatically insert generated outputs into assessed work. Preserve independently authored text.
- Instructor approval, student-independent checks, human source reading and completed research are never inferred from code publication or a passing test.
- Use row-stochastic matrices and row distributions with p_next=p@P. Default distance is L1, epsilon=0.001, strict distance<epsilon, and t=0 checked. Keep matrix and starting vector separate.
- Use eigenvalue moduli, remove exactly one stationary eigenvalue, and preserve negative/complex spectra. Main simulations require finite irreducible aperiodic chains. Keep diagnostic and iteration-limit statuses explicit.
- Retain probability, stationary residual and conditioning checks. Core propagation is deterministic. Any future randomness needs an explicit seed and justification.
- Keep the implementation lightweight. Run appropriate tests and inspect actual CLI/export/plot results and adjacent workflows before reporting success. Distinguish source, generated artifacts, GitHub delivery and assessed use.
- Never commit generated datasets/figures/logs, local environments, manuscript files, private research notes or personal machine paths by default. Run manifests can contain local paths and need deliberate review before sharing.
- This repository is the active software workspace. The separate research workspace's earlier source copies are historical prototypes associated with its original run receipts; do not silently edit both copies or treat them as synchronized.
- Preserve software-origin history. Git commits identify published versions; retain per-run source hashes for exploratory output reproducibility.
