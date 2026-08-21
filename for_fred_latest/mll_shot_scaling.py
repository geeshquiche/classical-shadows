#!/usr/bin/env python3
"""How does the per-element-vs-shared gap scale with the number of shots?

Tests the analysis behind result 5 (mll_per_element_variants.py): the naive per-element GP loses to
shared because it estimates a noise hyperparameter per element from noisy data, and that estimate
sharpens as the per-point measurement noise (3^k-<O>^2)/N shrinks -- i.e. as SHOTS N grow. Prediction:
  * naive per-element's gap vs shared should SHRINK as N grows (cleaner points -> less wobble);
  * empnoise (fix the noise to the measured shadow variance, fit only the lengthscale) should stay
    ahead of shared at every N, since it never estimates the noise at all.
Time points M are held fixed (denser M does not add smoothness information past Nyquist; shots do).

2-qubit TFIM rho(t), M=500 time points, N swept, scored by Frobenius distance to exact rho(t).
Reports per-N mean +/- SE for shared / per-elem / empnoise and their gaps vs shared.

Run:  python mll_shot_scaling.py        (full)
      QUICK=1 python mll_shot_scaling.py (small smoke test)
"""
import os, sys, time, resource, random, csv, warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_RHO = os.path.join(_PARENT, "rho_reconstruction")
for _p in (_PARENT, _RHO):
    if _p not in sys.path:
        sys.path.append(_p)
import conditional_rho as cr

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
if QUICK:
    TRUE_T, N_TIMES, TARGET_T, N_GRID, SEEDS = 200, 40, 120, [60, 240], [10, 20]
else:
    TRUE_T, N_TIMES, TARGET_T, N_GRID, SEEDS = 500, 500, 200, [125, 250, 500, 1000, 2000], list(range(10, 61, 10))
TOTAL_QUBITS, SUBSYS = 2, (0, 1)
LS_GRID = np.geomspace(0.02, 1.0, 7)
NOISE_GRID = np.geomspace(1e-3, 3e-1, 5)
VARIANTS = ["shared", "per-elem", "empnoise"]


def avg_shadow_matrices(states, obs_idx, seed, shadow):
    np.random.seed(seed); random.seed(seed)
    dim = 2 ** len(SUBSYS)
    mean = np.zeros((len(obs_idx), dim, dim), dtype=np.complex128)
    var_r = np.zeros((len(obs_idx), dim, dim)); var_i = np.zeros((len(obs_idx), dim, dim))
    for ti, idx in enumerate(obs_idx):
        st = states[int(idx)]
        rho_t = st if st.isoper else st * st.dag()
        acc = np.zeros((dim, dim), dtype=np.complex128)
        acc2r = np.zeros((dim, dim)); acc2i = np.zeros((dim, dim))
        for _ in range(shadow):
            setting = cr.create_random_pauli_obs(TOTAL_QUBITS)
            meas = cr.pauli_measurement(rho_t, setting, TOTAL_QUBITS)
            m = cr.local_shadow_matrix(meas[SUBSYS[0]][0], meas[SUBSYS[0]][1])
            for q in SUBSYS[1:]:
                m = np.kron(m, cr.local_shadow_matrix(meas[q][0], meas[q][1]))
            acc += m; acc2r += m.real ** 2; acc2i += m.imag ** 2
        mu = acc / shadow
        mean[ti] = mu
        var_r[ti] = np.clip(acc2r / shadow - mu.real ** 2, 0, None) / shadow
        var_i[ti] = np.clip(acc2i / shadow - mu.imag ** 2, 0, None) / shadow
    return mean, var_r, var_i


def truth_rhos(states, tlist, target):
    d = 2 ** len(SUBSYS)
    full = np.array([cr.reduced_density_matrix(s, SUBSYS) for s in states])
    out = np.empty((len(target), d, d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            out[:, i, j] = (np.interp(target, tlist, full[:, i, j].real)
                            + 1j * np.interp(target, tlist, full[:, i, j].imag))
    return out


def channels(mean, var_r, var_i):
    n = mean.shape[1]
    ch = []
    for i in range(n):
        for j in range(n):
            for part in ("real", "imag"):
                if part == "real":
                    y = mean[:, i, j].real; vmean = var_r[:, i, j]
                else:
                    y = mean[:, i, j].imag; vmean = var_i[:, i, j]
                mu, sd = float(y.mean()), float(y.std()) or 1.0
                alpha = np.clip(vmean / sd ** 2, 1e-8, None)
                ch.append(((i, j), part, (y - mu) / sd, mu, sd, alpha))
    return ch


def fit_shared(to, tn, ch, n):
    best, bl, bz = -np.inf, LS_GRID[0], NOISE_GRID[0]
    for ls in LS_GRID:
        for nz in NOISE_GRID:
            k = C(1.0, "fixed") * RBF(ls, "fixed") + WhiteKernel(nz, "fixed")
            tot = sum(GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False)
                      .fit(to.reshape(-1, 1), ys).log_marginal_likelihood()
                      for _, _, ys, _, _, _ in ch)
            if tot > best:
                best, bl, bz = tot, ls, nz
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    k = C(1.0, "fixed") * RBF(bl, "fixed") + WhiteKernel(bz, "fixed")
    for (i, j), part, ys, mu, sd, _ in ch:
        m = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False).fit(
            to.reshape(-1, 1), ys).predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def fit_per_element(to, tn, ch, n):
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for (i, j), part, ys, mu, sd, _ in ch:
        k = C(1.0, "fixed") * RBF(0.2, (0.02, 1.0)) + WhiteKernel(0.05, (1e-3, 3e-1))
        m = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=1).fit(
            to.reshape(-1, 1), ys).predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def fit_empnoise(to, tn, ch, n):
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for (i, j), part, ys, mu, sd, alpha in ch:
        k = C(1.0, "fixed") * RBF(0.2, (0.02, 1.0))
        m = GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=False,
                                     n_restarts_optimizer=1).fit(
            to.reshape(-1, 1), ys).predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def frob(pred, true):
    return float(np.linalg.norm(pred - true, axis=(1, 2)).mean())


def main():
    t0 = time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target = np.linspace(TIME_MIN, TIME_MAX, TARGET_T)
    tn = (target - TIME_MIN) / (TIME_MAX - TIME_MIN)
    states = qt.mesolve(cr.build_ising_hamiltonian(TOTAL_QUBITS),
                        cr.build_plus_initial_state(TOTAL_QUBITS), tlist, []).states
    truth = truth_rhos(states, tlist, target)
    obs_idx = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    to = np.linspace(0, 1, N_TIMES)
    n = 2 ** len(SUBSYS)
    print(f"QUICK={QUICK}  2-qubit rho, M={N_TIMES} time points, N in {N_GRID}, "
          f"{len(SEEDS)} seeds\n", flush=True)

    # scores[N][variant] = list over seeds
    scores = {N: {v: [] for v in VARIANTS} for N in N_GRID}
    for N in N_GRID:
        for seed in SEEDS:
            mean, var_r, var_i = avg_shadow_matrices(states, obs_idx, seed, N)
            ch = channels(mean, var_r, var_i)
            preds = {"shared": fit_shared(to, tn, ch, n),
                     "per-elem": fit_per_element(to, tn, ch, n),
                     "empnoise": fit_empnoise(to, tn, ch, n)}
            for v in VARIANTS:
                scores[N][v].append(frob(preds[v], truth))
        row = "  ".join(f"{v}={np.mean(scores[N][v]):.4f}" for v in VARIANTS)
        print(f"  N={N:5d} done  ({time.perf_counter()-t0:.0f}s)  {row}", flush=True)

    ns = len(SEEDS)
    means = {N: {v: float(np.mean(scores[N][v])) for v in VARIANTS} for N in N_GRID}
    ses = {N: {v: float(np.std(scores[N][v], ddof=1) / np.sqrt(ns)) for v in VARIANTS} for N in N_GRID}

    # ---- two panels: RMSE vs N, and gap-vs-shared (%) vs N ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for v, col in zip(VARIANTS, ["tab:blue", "tab:gray", "tab:green"]):
        a1.errorbar(N_GRID, [means[N][v] for N in N_GRID], yerr=[ses[N][v] for N in N_GRID],
                    marker="o", lw=1.6, capsize=3, color=col, label=v)
    a1.set_xscale("log"); a1.set_xlabel("shots per time N"); a1.set_ylabel("Frobenius to exact rho(t)")
    a1.set_title(f"Reconstruction error vs shots (M={N_TIMES})"); a1.grid(alpha=0.3); a1.legend()
    for v, col in zip(["per-elem", "empnoise"], ["tab:gray", "tab:green"]):
        gap = [(1 - means[N][v] / means[N]["shared"]) * 100 for N in N_GRID]
        a2.plot(N_GRID, gap, marker="o", lw=1.6, color=col, label=f"{v} vs shared")
    a2.axhline(0, color="tab:blue", lw=0.9, ls="--", label="shared")
    a2.set_xscale("log"); a2.set_xlabel("shots per time N"); a2.set_ylabel("improvement vs shared (%)")
    a2.set_title("Per-element gap shrinks with shots; empnoise stays ahead")
    a2.grid(alpha=0.3); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(_HERE, "mll_shot_scaling.png"), dpi=150)

    wall = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(os.path.join(_HERE, "mll_shot_scaling.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# 2-qubit TFIM rho(t); M={N_TIMES} time points; N swept; {ns} seeds; "
                    f"shared vs naive per-element vs empnoise (noise fixed to shadow variance); "
                    f"wall={wall:.0f}s; peak_mem={peak_mb:.0f}MB"])
        w.writerow(["N", "shared", "shared_SE", "per_elem", "per_elem_SE", "empnoise", "empnoise_SE",
                    "per_elem_vs_shared_pct", "empnoise_vs_shared_pct"])
        for N in N_GRID:
            w.writerow([N,
                        round(means[N]["shared"], 4), round(ses[N]["shared"], 4),
                        round(means[N]["per-elem"], 4), round(ses[N]["per-elem"], 4),
                        round(means[N]["empnoise"], 4), round(ses[N]["empnoise"], 4),
                        round((1 - means[N]["per-elem"] / means[N]["shared"]) * 100, 1),
                        round((1 - means[N]["empnoise"] / means[N]["shared"]) * 100, 1)])
    print(f"\nwall={wall:.0f}s  peak_mem={peak_mb:.0f}MB  -> saved figure + CSV")


if __name__ == "__main__":
    main()
