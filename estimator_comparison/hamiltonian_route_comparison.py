#!/usr/bin/env python3
"""Conditional vs matrix-GP route across the June Hamiltonian library, at house standard.

The June notebooks (gp_sweep / hamiltonian_comparison, "Project Conditional Probability") compared the
two routes on <X0> across 8 Hamiltonians at a SINGLE seed (42) with a globally shared numpy stream; the
one case where the conditional route "won" (weak: 0.0926 vs 0.0929) was a coin flip. This re-run uses
independent collision-free per-cell seeds (make_cell_seed), 5 seeds, paired data per seed, and reports
mean +/- SE with a 2xSE significance call per Hamiltonian.

Run:  QUICK=1 python hamiltonian_route_comparison.py    # 1 Ham, 1 seed smoke
      python hamiltonian_route_comparison.py            # full 8 x 5
Out:  hamiltonian_route_comparison.csv / .png (+ copies in the session scratchpad)
"""
from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np
import qutip as qt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from Synthetic_Error_Uncertainty_Check import (
    Config,
    make_cell_seed,
    pauli_string_operator,
    generate_measurement_df,
    run_conditional_model,
    summarize_prediction_samples,
)
from classical_shadow_matrix import construct_classical_shadow_matrices_by_time
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch

import os as _os
SP = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "out")  # repo-relative output dir
_os.makedirs(SP, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"
OBS = "XI"
QUBIT_NUM = 2
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
if QUICK:
    NUM_TRUE, NUM_PRED, NUM_OBS, SHADOW, SEEDS = 200, 40, 30, 40, [10]
    COND_ITER, NUM_INDUCING, RESAMPLES = 30, 15, 8
else:
    NUM_TRUE, NUM_PRED, NUM_OBS, SHADOW, SEEDS = 400, 100, 100, 200, [10, 20, 30, 40, 50]
    COND_ITER, NUM_INDUCING, RESAMPLES = 80, 40, 20

ZZ = qt.tensor(qt.sigmaz(), qt.sigmaz())
XI_ = qt.tensor(qt.sigmax(), qt.qeye(2))
IX_ = qt.tensor(qt.qeye(2), qt.sigmax())
ZI_ = qt.tensor(qt.sigmaz(), qt.qeye(2))
IZ_ = qt.tensor(qt.qeye(2), qt.sigmaz())
XX = qt.tensor(qt.sigmax(), qt.sigmax())
YY = qt.tensor(qt.sigmay(), qt.sigmay())

# (name, Hamiltonian, collapse operators) -- the June library, unchanged
HAMS = [
    ("ZZ_slow",    0.5 * ZZ,                                []),
    ("ZZ_fast",    2.5 * ZZ,                                []),
    ("ZZ_plus_XI", ZZ + 0.5 * XI_,                          []),
    ("weak",       0.05 * ZZ,                               []),
    ("entangling", XX + YY,                                 []),
    ("TFIM",       ZZ + 0.5 * XI_ + 0.5 * IX_,              []),
    ("beating",    0.4 * ZI_ + 0.6 * IZ_ + 0.3 * ZZ,        []),
    ("dephasing",  0.5 * ZZ,                                [np.sqrt(0.3) * ZI_]),
]
if QUICK:
    HAMS = HAMS[:1]
if os.environ.get("ONLY_HAM"):                       # e.g. ONLY_HAM=weak for a targeted top-up
    HAMS = [h for h in HAMS if h[0] in os.environ["ONLY_HAM"].split(",")]
if os.environ.get("SEED_LIST"):                      # e.g. SEED_LIST=60,70,...  (top-up seeds)
    SEEDS = [int(s) for s in os.environ["SEED_LIST"].split(",")]


def make_cfg() -> Config:
    return Config(
        base_seed=SEEDS[0], qubit_num=QUBIT_NUM, observable_string=OBS,
        dynamics_name="tfim", num_true_time_points=NUM_TRUE,
        prediction_time_count=NUM_PRED,
        shadow_size_grid=(SHADOW,), num_time_index_grid=(NUM_OBS,),
        num_dataset_repeats=1, prediction_resamples=RESAMPLES, gp_kernel="matern32",
        normal_training_iter=100, conditional_training_iter=COND_ITER,
        num_inducing=NUM_INDUCING, gp_lr=0.05,
        use_matched_basis_estimator_for_pauli_strings=False, independent_seeds=True,
    )


def main():
    t0 = perf_counter()
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = qt.tensor(plus, plus)
    tlist = np.linspace(TIME_MIN, TIME_MAX, NUM_TRUE)
    pred_times = np.linspace(TIME_MIN, TIME_MAX, NUM_PRED)
    obs_idx = np.linspace(0, NUM_TRUE - 1, NUM_OBS, dtype=int)
    operator = pauli_string_operator(OBS, QUBIT_NUM)
    cfg = make_cfg()

    print(f"QUICK={QUICK}  <{OBS}> | {NUM_OBS} times x {SHADOW} shadows | seeds={SEEDS} | "
          f"{len(HAMS)} Hamiltonians", flush=True)

    rows = ["hamiltonian,route,seed,rmse"]
    results = {}
    for name, H, c_ops in HAMS:
        states = qt.mesolve(H, psi0, tlist, c_ops).states
        true_full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth = np.interp(pred_times, tlist, true_full)
        r_mat, r_cond = [], []
        for seed in SEEDS:
            data_seed = make_cell_seed(seed, OBS + name, SHADOW, NUM_OBS, 0)
            mdf = generate_measurement_df(states, tlist, obs_idx, QUBIT_NUM, SHADOW,
                                          seed=data_seed)
            # matrix route (joint shadow matrices -> MLL-fit GP on Tr(O rho_hat))
            _, mtimes, mats = construct_classical_shadow_matrices_by_time(mdf, QUBIT_NUM)
            res = infer_observable_from_shadow_with_botorch(
                observations=mats, operator=operator, time_index=mtimes,
                target_time_index=pred_times, kernel="rbf", credible_mass=0.95)
            m_mean = np.real(np.asarray(res["posterior_mean"], dtype=complex))
            r_mat.append(float(np.sqrt(np.mean((m_mean - truth) ** 2))))
            # conditional route (autoregressive classifier, resampled shadows)
            sm = run_conditional_model(mdf, pred_times, cfg, SHADOW, operator)
            _, c_mean, _ = summarize_prediction_samples(sm, truth)
            r_cond.append(float(np.sqrt(np.mean((c_mean - truth) ** 2))))
            rows.append(f"{name},matrix,{seed},{r_mat[-1]:.6f}")
            rows.append(f"{name},conditional,{seed},{r_cond[-1]:.6f}")
            print(f"  {name:11s} seed {seed}: matrix={r_mat[-1]:.4f} cond={r_cond[-1]:.4f} "
                  f"({perf_counter()-t0:.0f}s)", flush=True)
        m, c = np.array(r_mat), np.array(r_cond)
        d = c - m
        sed = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        verdict = ("matrix better" if d.mean() > 0 else "conditional better")
        sig = "SIGNIFICANT" if (len(d) > 1 and abs(d.mean()) > 2 * sed) else "not resolved"
        results[name] = (m, c, d.mean(), sed, f"{verdict} ({sig})")
        print(f"  -> {name}: matrix {m.mean():.4f}  cond {c.mean():.4f}  "
              f"diff {d.mean():+.4f}+/-{sed:.4f}  {verdict} ({sig})", flush=True)

    for d in (_HERE, SP):
        open(os.path.join(d, "hamiltonian_route_comparison.csv"), "w").write("\n".join(rows) + "\n")

    names = [n for n, _, _ in HAMS]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    mm = [results[n][0].mean() for n in names]
    ms = [results[n][0].std(ddof=1) / np.sqrt(len(SEEDS)) if len(SEEDS) > 1 else 0 for n in names]
    cm = [results[n][1].mean() for n in names]
    cs = [results[n][1].std(ddof=1) / np.sqrt(len(SEEDS)) if len(SEEDS) > 1 else 0 for n in names]
    ax.bar(x - w / 2, mm, w, yerr=ms, capsize=3, color="#5dade2", edgecolor="k", lw=.5,
           label="matrix GP (joint shadow matrix)")
    ax.bar(x + w / 2, cm, w, yerr=cs, capsize=3, color="#e59866", edgecolor="k", lw=.5,
           label="conditional (autoregressive classifier)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"RMSE of $\langle X_0\rangle(t)$")
    ax.set_title(f"Route comparison across Hamiltonians ({NUM_OBS} times x {SHADOW} shadows, "
                 f"{len(SEEDS)} independent seeds; error bars 1 SE)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    for d in (_HERE, SP):
        fig.savefig(os.path.join(d, "hamiltonian_route_comparison.png"), dpi=130)
    print("saved hamiltonian_route_comparison.csv + .png")


if __name__ == "__main__":
    main()
