#!/usr/bin/env python3
"""Does tuning close the conditional route's gap to the matrix route on <Z0Z1>?

The route comparisons run the conditional route at fixed settings (40 inducing points, 80 iterations).
This sweep tests whether more capacity/optimisation narrows its gap: paired data per seed, 5 seeds,
configurations (inducing, iters) in {40,80} x {80,160}, against the matrix route on the same data.
Answers the viva question "did you tune the conditional before recommending the matrix route?".
"""
from __future__ import annotations
import os, sys
from time import perf_counter
import numpy as np
import torch          # seeded per (seed, config) cell for reproducible, order-independent SVGP inits
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.append(_PARENT)
from Synthetic_Error_Uncertainty_Check import (Config, make_cell_seed, build_hamiltonian,
    build_initial_state, pauli_string_operator, generate_measurement_df,
    run_conditional_model, summarize_prediction_samples)
from classical_shadow_matrix import construct_classical_shadow_matrices_by_time
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch

SP = "/private/tmp/claude-501/-Users-vzs-MLBD/6ff4ff12-f3ce-46f8-a65b-5e2fc95a74f1/scratchpad"
QUICK = os.environ.get("QUICK", "0") == "1"
NT, NP, NO, SH = (200, 40, 30, 40) if QUICK else (400, 100, 100, 200)
# 8 independent seeds (house standard for paired comparisons) and a capacity/optimisation grid whose
# baseline cell (40, 80) is EXACTLY the configuration used by the route comparisons, so the sweep is
# directly comparable to Table/Fig of Section 4.5. Runtime is dominated by the iters=320 cells; the
# overnight budget allows it.
SEEDS = [10] if QUICK else [10, 20, 30, 40, 50, 60, 70, 80]
CONFIGS = [(40, 80), (80, 80), (40, 160), (80, 160), (40, 320), (80, 320)]
if QUICK:
    CONFIGS = CONFIGS[:2]


def make_cfg(ind, it):
    return Config(base_seed=SEEDS[0], qubit_num=2, observable_string="ZZ", dynamics_name="tfim",
                  num_true_time_points=NT, prediction_time_count=NP, shadow_size_grid=(SH,),
                  num_time_index_grid=(NO,), num_dataset_repeats=1, prediction_resamples=20,
                  gp_kernel="matern32", normal_training_iter=100, conditional_training_iter=it,
                  num_inducing=ind, gp_lr=0.05, use_matched_basis_estimator_for_pauli_strings=False,
                  independent_seeds=True)


def main():
    t0 = perf_counter()
    tl = np.linspace(0, 2*np.pi, NT)
    pt = np.linspace(0, 2*np.pi, NP)
    states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tl, []).states
    op = pauli_string_operator("ZZ", 2)
    truth = np.interp(pt, tl, np.real(np.asarray(qt.expect(op, states))))
    oi = np.linspace(0, NT-1, NO, dtype=int)

    rows = ["config,seed,route,rmse"]
    res = {c: [] for c in CONFIGS}
    mat = []
    for seed in SEEDS:
        mdf = generate_measurement_df(states, tl, oi, 2, SH,
                                      seed=make_cell_seed(seed, "ZZ", SH, NO, 0))
        _, mt, mats = construct_classical_shadow_matrices_by_time(mdf, 2)
        r = infer_observable_from_shadow_with_botorch(observations=mats, operator=op,
              time_index=mt, target_time_index=pt, kernel="matern32", credible_mass=0.95)
        m_rmse = float(np.sqrt(np.mean((np.real(np.asarray(r["posterior_mean"], dtype=complex)) - truth) ** 2)))
        mat.append(m_rmse)
        rows.append(f"matrix,{seed},matrix,{m_rmse:.6f}")
        for c in CONFIGS:
            # reproducible + order-independent torch state per cell (mirrors make_cell_seed hygiene)
            torch.manual_seed(make_cell_seed(seed, f"ZZtune{c[0]}x{c[1]}", SH, NO, 1))
            sm = run_conditional_model(mdf, pt, make_cfg(*c), SH, op)
            _, cm, _ = summarize_prediction_samples(sm, truth)
            rmse = float(np.sqrt(np.mean((cm - truth) ** 2)))
            res[c].append(rmse)
            rows.append(f"ind{c[0]}_it{c[1]},{seed},conditional,{rmse:.6f}")
            print(f"seed {seed} ind={c[0]} it={c[1]}: cond={rmse:.4f} (matrix={m_rmse:.4f}) "
                  f"({perf_counter()-t0:.0f}s)", flush=True)

    print(f"\nmatrix: {np.mean(mat):.4f} +/- {np.std(mat, ddof=1)/np.sqrt(len(mat)):.4f}")
    for c in CONFIGS:
        v = np.array(res[c]); d = v - np.array(mat)
        sed = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"cond ind={c[0]:3d} it={c[1]:3d}: {v.mean():.4f} +/- "
              f"{v.std(ddof=1)/np.sqrt(len(v)):.4f}   paired gap to matrix {d.mean():+.4f}+/-{sed:.4f}")
    for d in (_HERE, SP):
        open(os.path.join(d, "conditional_tuning_sweep.csv"), "w").write("\n".join(rows) + "\n")
    print("saved conditional_tuning_sweep.csv")


if __name__ == "__main__":
    main()
