#!/usr/bin/env python3
"""Can per-element GP hyperparameters be made to BEAT one shared set for rho(t)?

Baseline result (mll_per_element_vs_shared.py, 20 seeds): fitting a free (lengthscale, noise) per
matrix-element channel is SIGNIFICANTLY worse than one shared set (0.0396 vs 0.0366). The cause is
variance: each channel estimates its own hyperparameters from ~M short, noisy points, and the
continuous optimiser wanders (picks a too-short lengthscale that fits noise, or too-long that
oversmooths). This script tests concrete fixes that keep per-element flexibility but tame that
variance, all on the SAME generated shadows per seed so the comparison is fair:

  shared          : one (ls, noise) grid-chosen to explain ALL channels        [baseline]
  per-elem        : free (ls, noise) per channel, continuous L-BFGS, 1 restart [the thing to beat]
  per-elem-grid   : per channel, best (ls, noise) from the SAME coarse grid    [is it the optimiser?]
  per-elem-restart: per-elem but 8 optimiser restarts                          [is it local optima?]
  per-elem-bound  : per-elem but lengthscale constrained near the shared value [partial pooling]
  grouped         : two shared sets -- one for diagonal, one for off-diagonal  [structured pooling]
  per-elem-empnoise: fit ls only; noise FIXED to the measured per-point shadow  [use the known noise]
                     variance (heteroscedastic alpha), so no noise hyperparam to estimate

Scored by Frobenius distance to exact rho(t). 2-qubit TFIM, house settings. Reports each variant's
mean +/- SE and a 2*SE significance verdict vs shared. Outputs a bar chart + CSV + timing.

Run:  python mll_per_element_variants.py        (full)
      QUICK=1 python mll_per_element_variants.py (small smoke test)
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

# Per-element fits deliberately push the optimiser to its bounds (that IS the pathology under study),
# so sklearn's "close to bound" warnings are expected and would otherwise flood the log.
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
    TRUE_T, N_TIMES, TARGET_T, SHADOW, SEEDS = 200, 40, 120, 120, [10, 20]
else:
    TRUE_T, N_TIMES, TARGET_T, SHADOW, SEEDS = 500, 500, 200, 500, list(range(10, 151, 10))  # 15 seeds
TOTAL_QUBITS, SUBSYS = 2, (0, 1)
LS_GRID = np.geomspace(0.02, 1.0, 7)
NOISE_GRID = np.geomspace(1e-3, 3e-1, 5)
VARIANTS = ["shared", "per-elem", "per-elem-grid", "per-elem-restart",
            "per-elem-bound", "grouped", "per-elem-empnoise"]
if os.environ.get("NSEED"):          # optional: bump seed count, e.g. NSEED=20
    SEEDS = list(range(10, 10 * int(os.environ["NSEED"]) + 1, 10))


def avg_shadow_matrices(states, obs_idx, seed):
    """Mean shadow matrix per time PLUS the variance of that mean for each real/imag channel.

    The per-point variance is the measured shadow noise (var across the SHADOW snapshots)/SHADOW --
    the same quantity the 'quality of the estimate' analysis models as (3^k-<O>^2)/N, but read
    straight off the data per element. Used to fix the noise in the empnoise variant.
    """
    np.random.seed(seed); random.seed(seed)
    dim = 2 ** len(SUBSYS)
    mean = np.zeros((len(obs_idx), dim, dim), dtype=np.complex128)
    var_r = np.zeros((len(obs_idx), dim, dim))   # variance of the MEAN, real part
    var_i = np.zeros((len(obs_idx), dim, dim))   # variance of the MEAN, imag part
    for ti, idx in enumerate(obs_idx):
        st = states[int(idx)]
        rho_t = st if st.isoper else st * st.dag()
        acc = np.zeros((dim, dim), dtype=np.complex128)
        acc2r = np.zeros((dim, dim)); acc2i = np.zeros((dim, dim))
        for _ in range(SHADOW):
            setting = cr.create_random_pauli_obs(TOTAL_QUBITS)
            meas = cr.pauli_measurement(rho_t, setting, TOTAL_QUBITS)
            m = cr.local_shadow_matrix(meas[SUBSYS[0]][0], meas[SUBSYS[0]][1])
            for q in SUBSYS[1:]:
                m = np.kron(m, cr.local_shadow_matrix(meas[q][0], meas[q][1]))
            acc += m; acc2r += m.real ** 2; acc2i += m.imag ** 2
        mu = acc / SHADOW
        mean[ti] = mu
        # var of a single shot, then /SHADOW for the variance of the average; clip tiny negatives
        vr = np.clip(acc2r / SHADOW - mu.real ** 2, 0, None) / SHADOW
        vi = np.clip(acc2i / SHADOW - mu.imag ** 2, 0, None) / SHADOW
        var_r[ti] = vr; var_i[ti] = vi
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
    """One entry per (element, real/imag): standardised series + fixed per-point noise (alpha)."""
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
                alpha = np.clip(vmean / sd ** 2, 1e-8, None)   # noise in standardised units
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
        gp = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False).fit(to.reshape(-1, 1), ys)
        m = gp.predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred, bl, bz


def fit_per_element(to, tn, ch, n, restarts=1, ls_bounds=(0.02, 1.0)):
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for (i, j), part, ys, mu, sd, _ in ch:
        ls0 = float(np.sqrt(ls_bounds[0] * ls_bounds[1]))
        k = C(1.0, "fixed") * RBF(ls0, ls_bounds) + WhiteKernel(0.05, (1e-3, 3e-1))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=False,
                                      n_restarts_optimizer=restarts).fit(to.reshape(-1, 1), ys)
        m = gp.predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def fit_per_element_grid(to, tn, ch, n):
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for (i, j), part, ys, mu, sd, _ in ch:
        best, bk = -np.inf, None
        for ls in LS_GRID:
            for nz in NOISE_GRID:
                k = C(1.0, "fixed") * RBF(ls, "fixed") + WhiteKernel(nz, "fixed")
                gp = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False).fit(to.reshape(-1, 1), ys)
                lml = gp.log_marginal_likelihood()
                if lml > best:
                    best, bk = lml, gp
        m = bk.predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def fit_grouped(to, tn, ch, n):
    """Two shared sets: diagonal channels (i==j) share one (ls, noise), off-diagonal share another."""
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for group in (lambda i, j: i == j, lambda i, j: i != j):
        sub = [c for c in ch if group(c[0][0], c[0][1])]
        if not sub:
            continue
        best, bl, bz = -np.inf, LS_GRID[0], NOISE_GRID[0]
        for ls in LS_GRID:
            for nz in NOISE_GRID:
                k = C(1.0, "fixed") * RBF(ls, "fixed") + WhiteKernel(nz, "fixed")
                tot = sum(GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False)
                          .fit(to.reshape(-1, 1), ys).log_marginal_likelihood()
                          for _, _, ys, _, _, _ in sub)
                if tot > best:
                    best, bl, bz = tot, ls, nz
        k = C(1.0, "fixed") * RBF(bl, "fixed") + WhiteKernel(bz, "fixed")
        for (i, j), part, ys, mu, sd, _ in sub:
            gp = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False).fit(to.reshape(-1, 1), ys)
            m = gp.predict(tn.reshape(-1, 1)) * sd + mu
            pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def fit_per_element_empnoise(to, tn, ch, n):
    """Fit only the lengthscale; the per-point noise is FIXED to the measured shadow variance."""
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    for (i, j), part, ys, mu, sd, alpha in ch:
        k = C(1.0, "fixed") * RBF(0.2, (0.02, 1.0))
        gp = GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=False,
                                      n_restarts_optimizer=1).fit(to.reshape(-1, 1), ys)
        m = gp.predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def frob(pred, true):
    return float(np.linalg.norm(pred - true, axis=(1, 2)).mean())


def run_variants(to, tn, ch, n):
    sh, bl, bz = fit_shared(to, tn, ch, n)
    out = {"shared": sh}
    out["per-elem"] = fit_per_element(to, tn, ch, n, restarts=1)
    out["per-elem-grid"] = fit_per_element_grid(to, tn, ch, n)
    out["per-elem-restart"] = fit_per_element(to, tn, ch, n, restarts=8)
    out["per-elem-bound"] = fit_per_element(to, tn, ch, n, restarts=1, ls_bounds=(bl / 2.5, bl * 2.5))
    out["grouped"] = fit_grouped(to, tn, ch, n)
    out["per-elem-empnoise"] = fit_per_element_empnoise(to, tn, ch, n)
    return out


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
    print(f"QUICK={QUICK}  2-qubit rho, {N_TIMES} time points, {SHADOW} shadows/time, "
          f"{len(SEEDS)} seeds, {len(VARIANTS)} variants\n", flush=True)

    scores = {v: [] for v in VARIANTS}
    for si, seed in enumerate(SEEDS):
        mean, var_r, var_i = avg_shadow_matrices(states, obs_idx, seed)
        ch = channels(mean, var_r, var_i)
        preds = run_variants(to, tn, ch, n)
        for v in VARIANTS:
            scores[v].append(frob(preds[v], truth))
        print(f"  seed {seed} done  ({time.perf_counter()-t0:.0f}s)  "
              + "  ".join(f"{v}={scores[v][-1]:.4f}" for v in VARIANTS), flush=True)

    ns = len(SEEDS)
    means = {v: float(np.mean(scores[v])) for v in VARIANTS}
    ses = {v: float(np.std(scores[v], ddof=1) / np.sqrt(ns)) for v in VARIANTS}
    base = means["shared"]

    # ---- bar chart: Frobenius per variant, error bars = SE, shared highlighted ----
    order = sorted(VARIANTS, key=lambda v: means[v])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue" if v == "shared" else ("tab:green" if means[v] < base else "tab:gray") for v in order]
    ax.bar(range(len(order)), [means[v] for v in order], yerr=[ses[v] for v in order],
           color=colors, capsize=4)
    ax.axhline(base, color="tab:blue", lw=0.9, ls="--", label="shared baseline")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Frobenius distance to exact rho(t)")
    ax.set_title(f"Making per-element MLL competitive, 2-qubit rho(t)  "
                 f"({N_TIMES} times, {SHADOW} shadows, {ns} seeds)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(_HERE, "mll_per_element_variants.png"), dpi=150)

    wall = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(os.path.join(_HERE, "mll_per_element_variants.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# 2-qubit TFIM rho(t); {N_TIMES} time points; {SHADOW} shadows/time; {ns} seeds; "
                    f"variants of per-element GP hyperparameters vs one shared set; "
                    f"%=improvement vs shared (+ = better); significant if |mean-shared|>2*sqrt(SE^2+SE_shared^2); "
                    f"wall={wall:.0f}s; peak_mem={peak_mb:.0f}MB"])
        w.writerow(["variant", "Frob_mean", "SE", "pct_vs_shared", "verdict_vs_shared"])
        for v in order:
            if v == "shared":
                verdict = "baseline"
            else:
                sig = abs(means[v] - base) > 2 * np.sqrt(ses[v] ** 2 + ses["shared"] ** 2)
                better = means[v] < base
                verdict = ((("beats" if better else "worse than") + " shared") +
                           (" (SIGNIFICANT)" if sig else " (not resolved)"))
            w.writerow([v, round(means[v], 4), round(ses[v], 4),
                        round((1 - means[v] / base) * 100, 1), verdict])
            print(f"  {v:20s} {means[v]:.4f} +/- {ses[v]:.4f}   "
                  f"{(1-means[v]/base)*100:+5.1f}% vs shared   {verdict}")
    print(f"\nwall={wall:.0f}s  peak_mem={peak_mb:.0f}MB  -> saved figure + CSV")


if __name__ == "__main__":
    main()
