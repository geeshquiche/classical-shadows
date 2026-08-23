#!/usr/bin/env python3
"""Fitted-vs-analytic noise consistency check (provenance for report Sections 3 and 4.1).

Compares the observation-noise variance selected by the marginal-likelihood fit of the matrix/observable
GP route against the trajectory-averaged analytic shadow variance (3 - <X0>(t)^2)/N, for the house
configuration (XI, 2-qubit TFIM, 100 observed times x 200 shadows, RBF kernel), 8 independent seeds via
make_cell_seed.

Reports the ratio fitted/analytic over the seed list (NSEED env selects the number of seeds).
The fit UNDER-estimates the noise systematically (right order, low by ~1/3), consistent with the
under-coverage of the bands and with the budget noise-fit artefact (budget_empnoise_test).
"""
import sys, os
import numpy as np
import qutip as qt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from Synthetic_Error_Uncertainty_Check import (build_hamiltonian, build_initial_state,
                                               generate_measurement_df, pauli_string_operator,
                                               make_cell_seed)
from classical_shadow_matrix import construct_classical_shadow_matrices_by_time
from bayesian_matrix_inference_botorch import bayesian_infer_matrix_over_time

N_TRUE, N_OBS, SHADOW = 400, 100, 200
# NSEED env override: NSEED=20 -> seeds 10,20,...,200
SEEDS = list(range(10, 10 * int(os.environ.get("NSEED", "8")) + 1, 10))

tl = np.linspace(0, 2 * np.pi, N_TRUE)
states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tl, []).states
op = pauli_string_operator("XI", 2); opm = op.full()
truth = np.real(np.asarray(qt.expect(op, states)))
oi = np.linspace(0, N_TRUE - 1, N_OBS, dtype=int)
analytic = np.mean((3.0 - truth[oi] ** 2) / SHADOW)

ratios = []
for seed in SEEDS:
    mdf = generate_measurement_df(states, tl, oi, 2, SHADOW,
                                  seed=make_cell_seed(seed, "XI", SHADOW, N_OBS, 0))
    _, mt, mats = construct_classical_shadow_matrices_by_time(mdf, 2)
    y = np.array([np.real(np.trace(opm @ m)) for m in mats]).reshape(-1, 1, 1)
    r = bayesian_infer_matrix_over_time(observations=y, time_index=mt, target_time_index=mt,
                                        kernel="rbf", return_hyperparameters=True)
    fitted = float(np.asarray(r["hyperparameters"]["noise"]).ravel()[0])
    ratios.append(fitted / analytic)
    print(f"seed {seed}: fitted={fitted:.5f} analytic={analytic:.5f} ratio={ratios[-1]:.3f}", flush=True)

ratios = np.array(ratios)
print(f"\nRATIO fitted/analytic: mean={ratios.mean():.3f} min={ratios.min():.3f} "
      f"max={ratios.max():.3f} ({len(SEEDS)} seeds)")
