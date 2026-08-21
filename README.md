# Learning the Quantum Dynamics of Classical Shadows

Code for the Imperial College London MRes project **MLBD_2025_10** (Vageesh Singh, 2025–26):
Gaussian-process reconstruction of quantum dynamics — observable trajectories ⟨O⟩(t) and sub-system
density matrices ρ(t) — from classical-shadow measurements on simulated spin chains.

**Start here:** [`docs/seeding_and_reproducibility.pdf`](docs/seeding_and_reproducibility.pdf) — the
companion document describing how randomness is assigned across experiments, the audits applied to the
pipeline, and the implementation details of the final estimator (matched-count normalisation, smoothed
variance inputs, exact pattern-count sampling). The scientific results and theory live in the MRes
report; this repository carries the code that produced them.

## What is here

| Path | Contents |
|---|---|
| repository root | Measurement simulation, shadow construction, collision-free per-cell seeding (`make_cell_seed`), classifier-route machinery |
| `rho_reconstruction/` | Joint shadow matrices; conditional (autoregressive classifier) route |
| `calibration_and_adaptive/` | Coverage-evaluation utilities |
| `for_fred_latest/` | Per-element ρ(t) fitting variants (shared / per-element / empirical-noise) |
| `estimator_comparison/` | Noise-scale consistency check (fitted vs analytic) |
| `review_ablations/` | The final-method studies reported in the MRes report, with summary CSVs of their results |

Only code that is directly relevant to the reported results is included; exploratory notebooks and
superseded pipelines are deliberately omitted.

## Running

Python 3.11 with `numpy`, `scipy`, `pandas`, `qutip`, `scikit-learn`, `torch`, `gpytorch`, `botorch`,
`matplotlib`. Every study script supports a fast smoke test via `QUICK=1`, and several take a `MODE`
environment variable (documented in each script's header), e.g.

```bash
QUICK=1 python review_ablations/routes3_final.py          # smoke test
MODE=table python review_ablations/routes3_final.py       # full 20-seed route comparison
MODE=core  python review_ablations/rho_final_program.py   # full-state 2x2 de-confound
```

Scripts are deterministic given their seed configuration: every experimental cell draws an independent,
collision-free seed, and results are written to CSVs that record the configuration, seeds and wall time.
The exactly simulated dynamics is used only to sample measurement outcomes and to score final metrics.

## Provenance

This code was developed within a two-person MRes project; parts of the measurement-simulation and
classifier-route utilities originate from the joint codebase with my project partner, Fred Xu.

**AI assistance:** substantial portions of this codebase and its documentation were developed with the
assistance of Claude (Anthropic), working under the author's direction; experimental designs, results
and conclusions were specified, checked and verified by the author, and every reported number can be
regenerated from the scripts in this repository.
