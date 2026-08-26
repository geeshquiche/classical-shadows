# Learning the Quantum Dynamics of Classical Shadows

Code and reproducibility companion for the Imperial College London MRes project **MLBD_2025_10**
(Vageesh Singh, 2025–26).

## What the project is about

Classical shadows compress quantum measurements into classically stored snapshots from which many
properties of a system can later be estimated — but each estimate is noisy and refers to a single
instant. This project reconstructs the *time dependence*: observable trajectories ⟨O⟩(t) and reduced
density matrices ρ(t) of simulated spin chains, from shadows taken at a finite set of times, using
Gaussian-process regression with a predictive uncertainty that is checked against the truth by empirical
coverage rather than assumed.

## What was achieved

- **A calibrated reconstruction method.** Gaussian-process regression on matched-count shadow estimates
  with the observation noise supplied from the measured snapshot scatter halves the raw estimator's error
  on both a local observable and a two-body correlator, with 95% bands that never under-cover at any
  sampling density tested (`studies/final_headline_recon.py`, `studies/final_config_coverage.py`).
- **A route comparison.** Against a generative Bayesian model of the joint measurement-outcome
  distribution (the conditional/autoregressive classifier route), the regression is the most accurate route in
  every comparison — 20 paired seeds, three budgets, four qubits, and all eight Hamiltonians of a dynamics
  library on all 160 individual paired seeds — at a small fraction of the cost. Quadrupling the conditional
  route's capacity and quintupling its resampling closes only 5% of the gap for a local observable and 19%
  for the correlator, leaving it worse than even the raw estimator: the deficit is structural
  (`studies/routes3_final.py`, `studies/ham_gp_recompute.py`, `estimator_comparison/`).
- **The noise finding.** Fitting the observation noise by marginal likelihood under-estimates it; supplying
  the physically known noise instead is what makes the bands calibrated, improves full-state
  reconstruction by 9% over shared hyperparameters, and flips the budget-allocation verdict
  (`studies/rho_final_program.py`, `studies/budget_final.py`).
- **Scaling and scope.** Shadow variance grows as 3^k with observable weight but is independent of system
  size; within a reduced block, partial reconstruction gives no sparsity shortcut
  (`per_element_rho/mll_k_vs_n.py`, `studies/partial_final.py`).

## Where to look

| Document / folder | What it is |
|---|---|
| **[`docs/seeding_and_reproducibility.pdf`](docs/seeding_and_reproducibility.pdf)** | Companion document: how randomness is assigned across experiments, the audits applied to the pipeline, the experimental-design rationale, and the implementation details of the final estimator. The report's Section 3 and its evaluation protocol summarise this document; read it first. (`.tex` source alongside.) |
| `studies/` | The final-method studies reported in the MRes report: eleven scripts, each with the summary CSV of its results next to it (see the table below). |
| `estimator_comparison/` | Earlier studies whose results the report still uses: the 8-Hamiltonian sweep (conditional arm), the conditional-route capacity sweep, the nested-vs-independent seeding demonstration, the element-grid and lengthscale-map figures, and the fitted-vs-analytic noise check. |
| `figures/` | Scripts that turn the CSVs into the report's figures (`csv_figs_vector.py` for most, plus four smaller ones). Outputs go to `figures/out/`. |
| `per_element_rho/` | Per-element ρ(t) fitting variants (shared / per-element / empirical-noise) and the locality study. |
| `rho_reconstruction/` | Joint shadow matrices and the conditional (autoregressive classifier) route. |
| `calibration_and_adaptive/` | Coverage-evaluation utilities. |
| repository root | Measurement simulation, shadow construction, collision-free per-cell seeding (`make_cell_seed`), classifier machinery. |

### The studies in `studies/` and the report sections they produce

| Script | Produces | Report |
|---|---|---|
| `final_headline_recon.py` | raw vs GP reconstruction, coverage, classical-smoother baselines | §4.1, §6 |
| `final_config_coverage.py` | sensitivity ladder: 16 design arms × 5 sampling densities | §4.1 (ladder figure) |
| `routes3_final.py` | three-route comparison: table, budget robustness, 4-qubit check | §4.5 (Table 1, figures) |
| `ham_gp_recompute.py` | GP arm of the 8-Hamiltonian sweep, paired with the stored conditional arm | §4.5 |
| `rho_final_program.py` | ρ(t) program: 2×2 de-confound, shot scaling, n-qubit, variants ladder | §4.6, §4.2 |
| `partial_final.py` | partial vs full block reconstruction | §4.8 |
| `budget_final.py` | 1×500 vs 5×100 allocation under fitted and empirical noise | §4.3 |
| `coverage_basis_ablation.py`, `coverage_xi_fix_probe.py` | fixed- vs fresh-basis protocol; mechanism check (alignment count vs scale and coverage) | §5.1 |
| `mom_vs_mean.py` | median-of-means vs sample mean | Appendix A |
| `allocation_sweep.py` | matched-budget allocation: observation times against shadows per time | §4.2 |
| `uncertainty_split.py` | aleatoric/epistemic decomposition by refitting independent datasets | §4.2 |
| `band_construction.py`, `band_switch_table.py` | latent vs predictive uncertainty band | §5.1 |
| `split_mechanism.py` | bias/variance decomposition of the split-budget penalty | §5.1 |
| `conditional_fairness.py` | conditional-route capacity and resampling sweep | §5.5 |
| `verify_pipeline.py` | end-to-end checks that the code computes what the report states | — |
| `gp_vs_classical_smoothers.py` | spline and Savitzky–Golay baselines on the route-comparison data | §5 |

## Running

Python 3.11 with `numpy`, `scipy`, `pandas`, `qutip`, `scikit-learn`, `torch`, `gpytorch`, `botorch`,
`matplotlib`. Every study script supports a fast smoke test via `QUICK=1`, and several take a `MODE`
environment variable (documented in each script's header):

```bash
QUICK=1 python studies/routes3_final.py          # smoke test
MODE=table python studies/routes3_final.py       # full 20-seed route comparison
MODE=core  python studies/rho_final_program.py   # full-state 2x2 de-confound
python figures/csv_figs_vector.py                # regenerate most report figures from the CSVs
```

Scripts are deterministic given their seed configuration: every experimental cell draws an independent,
collision-free seed, and results are written to CSVs that record the configuration, seeds and wall time.
The exactly simulated dynamics is used only to sample measurement outcomes and to score final metrics.

## Provenance

Developed within a two-person MRes project; the measurement-simulation and classifier-route utilities
build on code shared within the project team.

**AI assistance:** substantial portions of this codebase and its documentation were developed with the
assistance of Claude (Anthropic), working under the author's direction; experimental designs, results and
conclusions were specified, checked and verified by the author, and every reported number can be
regenerated from the scripts in this repository.
