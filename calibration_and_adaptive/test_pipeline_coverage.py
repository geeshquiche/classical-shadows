#!/usr/bin/env python3
"""Calibration coverage in the REAL pipeline (matrix-GP route), not the sklearn toy.

Earlier (test_calibration.py) the shot-only-vs-fit-inclusive band contrast was shown on an
sklearn GP with a synthetic Bernoulli model. This runs the SAME contrast through the actual
matrix-GP reconstruction of <O>(t) on the 2-qubit TFIM, so the calibration claim is about the
method we actually use:

  shot-only band      : 1.96 * (shot standard error of the single-shot shadow estimator)
  fit-inclusive band  : 1.96 * sqrt(GP-posterior variance + shot^2)   [adds fit uncertainty]

Empirical coverage = fraction of target times whose truth lies inside the band, averaged over
seeds (target 0.95). Reports both bands for XI and ZZ.

Run:  python test_pipeline_coverage.py
"""
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_RHO = os.path.join(_PARENT, "rho_reconstruction")   # conditional_rho lives here
for _p in (_PARENT, _RHO):
    if _p not in sys.path:
        sys.path.append(_p)

import conditional_rho as cr
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch

Z95 = 1.959963984540054
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
TRUE_T, PRED_T = 400, 60
N_TIMES, SHADOWS, SEEDS = 20, 60, list(range(10, 30))
KERNEL = "matern32"
SUBSYS = (0, 1)

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)
OPS = {"XI": np.kron(_sx, _I), "ZZ": np.kron(_sz, _sz)}


def exact_curve(states, tlist, op, target_times):
    vals = np.array([np.trace(op @ cr.reduced_density_matrix(s, SUBSYS)).real for s in states])
    return np.interp(target_times, tlist, vals)


def shot_standard_error(mdf, op, obs_times):
    """Per observed time: std over single-shot estimators Tr(O * one-shot shadow) / sqrt(N)."""
    se = np.zeros(len(obs_times))
    for ti, t in enumerate(obs_times):
        block = mdf[mdf["time"] == t]
        vals = []
        for _, shot_rows in block.groupby(["shadow_id", "shot_repeat"]):
            m = cr.subsystem_shadow_matrix(shot_rows, SUBSYS)
            vals.append(np.trace(op @ m).real)
        vals = np.asarray(vals)
        se[ti] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return se


def main():
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tlist, []).states

    print(f"matrix-GP route, {N_TIMES} obs times x {SHADOWS} shadows, {len(SEEDS)} seeds\n")
    for name, op in OPS.items():
        truth = exact_curve(states, tlist, op, target_times)
        cov_shot, cov_fit = [], []
        for seed in SEEDS:
            observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
            mdf = cr.generate_repeated_measurement_df(states, tlist, observed, 2, SHADOWS,
                                                      shots_per_setting=1, seed=seed)
            obs_times, shadow_mats = cr.averaged_subsystem_shadow_matrices(mdf, SUBSYS)
            res = infer_observable_from_shadow_with_botorch(
                observations=shadow_mats, operator=op, time_index=obs_times,
                target_time_index=target_times, kernel=KERNEL, credible_mass=0.95)
            mean = np.real(np.asarray(res["posterior_mean"], dtype=complex))
            fit_var = np.real(np.asarray(res["posterior_variance"], dtype=float))
            se_obs = shot_standard_error(mdf, op, obs_times)
            se_tgt = np.interp(target_times, obs_times, se_obs)
            band_shot = Z95 * se_tgt
            band_fit = Z95 * np.sqrt(fit_var + se_tgt ** 2)
            err = np.abs(mean - truth)
            cov_shot.append(np.mean(err <= band_shot))
            cov_fit.append(np.mean(err <= band_fit))
        cs, cf = np.mean(cov_shot), np.mean(cov_fit)
        print(f"{name}:  shot-only coverage = {cs:.3f}   fit-inclusive coverage = {cf:.3f}   "
              f"(target 0.95)")
        verdict = ("fit-inclusive closer to 0.95" if abs(cf - 0.95) < abs(cs - 0.95)
                   else "shot-only closer to 0.95")
        print(f"      -> {verdict}\n")


if __name__ == "__main__":
    main()
