#!/usr/bin/env python3
"""Separate observable-WEIGHT (k) dependence from total-SYSTEM-SIZE (n) dependence.

Classical-shadow locality predicts that reconstructing a k-qubit subsystem uses only those k qubits'
outcomes, so the measurement (shadow) noise should depend on k ONLY, not on the total system size n.
The shadow variance law is (3^k - <O>^2)/N, i.e. ~3^k per single shot. This script checks both:

  Sweep A (n-independence): fix the subsystem to qubits (0,1) [k=2], vary total n in {2,3,4,6}.
      -> measured single-shot shadow variance should be ~FLAT in n (locality); RMSE may drift because
         the 2-qubit subsystem DYNAMICS change with n (more bath -> more mixing).
  Sweep B (k-dependence): fix n=4, take subsystems of size k in {1,2,3}.
      -> measured single-shot shadow variance should grow ~geometrically (~3^k).

Also checks that empnoise (fix GP noise to the measured shadow variance) still beats shared / naive
per-element for subsystems of larger systems. 2-qubit-per-site TFIM chain. Reduced settings (this is
about trends, not house-standard numbers).

Run:  python mll_k_vs_n.py        (full)
      QUICK=1 python mll_k_vs_n.py (smoke test)
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
    TRUE_T, N_TIMES, TARGET_T, SHADOW, SEEDS = 80, 24, 80, 120, [10, 20]
else:
    TRUE_T, N_TIMES, TARGET_T, SHADOW, SEEDS = 300, 200, 120, 300, [10, 20, 30, 40, 50]
LS_GRID = np.geomspace(0.02, 1.0, 7)
NOISE_GRID = np.geomspace(1e-3, 3e-1, 5)
VARIANTS = ["shared", "per-elem", "empnoise"]

# (sweep tag, n_total, subsystem). (4,(0,1)) is shared by both sweeps -> computed once, listed once.
CONFIGS = [
    ("n-indep", 2, (0, 1)),
    ("n-indep", 3, (0, 1)),
    ("n-indep/k-dep", 4, (0, 1)),
    ("n-indep", 6, (0, 1)),
    ("k-dep", 4, (0,)),
    ("k-dep", 4, (0, 1, 2)),
]


def avg_shadow_matrices(states, obs_idx, seed, shadow, nq, subsys):
    """Averaged shadow matrix + per-element variance-of-the-mean, PLUS the mean single-shot variance
    of the k-body Z correlator <Z^{otimes k}> (the clean 3^k demonstrator)."""
    np.random.seed(seed); random.seed(seed)
    dim = 2 ** len(subsys)
    mean = np.zeros((len(obs_idx), dim, dim), dtype=np.complex128)
    var_r = np.zeros((len(obs_idx), dim, dim)); var_i = np.zeros((len(obs_idx), dim, dim))
    zvar_times = []
    for ti, idx in enumerate(obs_idx):
        st = states[int(idx)]
        rho_t = st if st.isoper else st * st.dag()
        acc = np.zeros((dim, dim), dtype=np.complex128)
        acc2r = np.zeros((dim, dim)); acc2i = np.zeros((dim, dim))
        zsum = 0.0; zsum2 = 0.0
        for _ in range(shadow):
            setting = cr.create_random_pauli_obs(nq)
            meas = cr.pauli_measurement(rho_t, setting, nq)
            mats = [cr.local_shadow_matrix(meas[q][0], meas[q][1]) for q in subsys]
            m = mats[0]
            for mm in mats[1:]:
                m = np.kron(m, mm)
            acc += m; acc2r += m.real ** 2; acc2i += m.imag ** 2
            zval = 1.0                                   # single-shot <Z^{otimes k}> = prod Tr[Z m_q]
            for mm in mats:
                zval *= (mm[0, 0].real - mm[1, 1].real)
            zsum += zval; zsum2 += zval * zval
        mu = acc / shadow
        mean[ti] = mu
        var_r[ti] = np.clip(acc2r / shadow - mu.real ** 2, 0, None) / shadow
        var_i[ti] = np.clip(acc2i / shadow - mu.imag ** 2, 0, None) / shadow
        zvar_times.append(zsum2 / shadow - (zsum / shadow) ** 2)   # single-shot var of <Z^k>
    return mean, var_r, var_i, float(np.mean(zvar_times))


def truth_rhos(states, tlist, target, subsys):
    d = 2 ** len(subsys)
    full = np.array([cr.reduced_density_matrix(s, subsys) for s in states])
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


def run_config(tag, nq, subsys, t0):
    k = len(subsys)
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target = np.linspace(TIME_MIN, TIME_MAX, TARGET_T)
    tn = (target - TIME_MIN) / (TIME_MAX - TIME_MIN)
    states = qt.mesolve(cr.build_ising_hamiltonian(nq),
                        cr.build_plus_initial_state(nq), tlist, []).states
    truth = truth_rhos(states, tlist, target, subsys)
    obs_idx = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    to = np.linspace(0, 1, N_TIMES)
    n = 2 ** k
    scores = {v: [] for v in VARIANTS}
    zvar = []   # single-shot variance of the k-body Z correlator (should be ~3^k - <O>^2)
    for seed in SEEDS:
        mean, var_r, var_i, zv = avg_shadow_matrices(states, obs_idx, seed, SHADOW, nq, subsys)
        zvar.append(zv)
        ch = channels(mean, var_r, var_i)
        preds = {"shared": fit_shared(to, tn, ch, n),
                 "per-elem": fit_per_element(to, tn, ch, n),
                 "empnoise": fit_empnoise(to, tn, ch, n)}
        for v in VARIANTS:
            scores[v].append(frob(preds[v], truth))
    ns = len(SEEDS)
    res = {v: (float(np.mean(scores[v])), float(np.std(scores[v], ddof=1) / np.sqrt(ns))) for v in VARIANTS}
    sv = float(np.mean(zvar))
    print(f"  [{tag}] n={nq} k={k} ({n}x{n})  <Z^k>var/shot={sv:.2f} (3^k={3**k})  "
          + "  ".join(f"{v}={res[v][0]:.4f}" for v in VARIANTS) + f"  ({time.perf_counter()-t0:.0f}s)",
          flush=True)
    return res, sv


def main():
    t0 = time.perf_counter()
    print(f"QUICK={QUICK}  reduced ρ from n-qubit TFIM chain; M={N_TIMES} times, {SHADOW} shadows, "
          f"{len(SEEDS)} seeds\n", flush=True)
    rows = []
    for tag, nq, subsys in CONFIGS:
        res, sv = run_config(tag, nq, subsys, t0)
        rows.append((tag, nq, len(subsys), 2 ** len(subsys), sv, res))

    # ---- figure: (left) shot variance vs config; (right) RMSE shared vs empnoise per config ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"n={nq},k={k}" for (tag, nq, k, mm, sv, res) in rows]
    a1.bar(range(len(rows)), [sv for (_, _, _, _, sv, _) in rows], color="tab:purple")
    for x, (tag, nq, k, mm, sv, res) in enumerate(rows):
        a1.text(x, sv, f"3^k={3**k}", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(range(len(rows))); a1.set_xticklabels(labels, rotation=30, ha="right")
    a1.set_ylabel("measured single-shot shadow variance")
    a1.set_title("Shadow noise: grows with k, flat in n"); a1.grid(axis="y", alpha=0.3)
    for v, col in zip(["shared", "empnoise"], ["tab:blue", "tab:green"]):
        a2.errorbar(range(len(rows)), [res[v][0] for (_, _, _, _, _, res) in rows],
                    yerr=[res[v][1] for (_, _, _, _, _, res) in rows], marker="o", lw=1.5,
                    capsize=3, color=col, label=v)
    a2.set_xticks(range(len(rows))); a2.set_xticklabels(labels, rotation=30, ha="right")
    a2.set_ylabel("Frobenius to exact reduced ρ(t)")
    a2.set_title("Reconstruction error (empnoise vs shared)"); a2.grid(alpha=0.3); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(_HERE, "mll_k_vs_n.png"), dpi=150)

    wall = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(os.path.join(_HERE, "mll_k_vs_n.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# reduced rho(t) from n-qubit TFIM chain; M={N_TIMES} times; {SHADOW} shadows; "
                    f"{len(SEEDS)} seeds; shot_var=measured single-shot shadow variance (compare 3^k); "
                    f"empnoise_vs_shared_pct = +better; wall={wall:.0f}s; peak_mem={peak_mb:.0f}MB"])
        w.writerow(["sweep", "n_total", "k_subsys", "matrix", "shot_var_per_shot", "3^k",
                    "shared", "shared_SE", "per_elem", "empnoise", "empnoise_SE", "empnoise_vs_shared_pct"])
        for tag, nq, k, mm, sv, res in rows:
            base = res["shared"][0]
            w.writerow([tag, nq, k, f"{mm}x{mm}", round(sv, 2), 3 ** k,
                        round(res["shared"][0], 4), round(res["shared"][1], 4),
                        round(res["per-elem"][0], 4), round(res["empnoise"][0], 4),
                        round(res["empnoise"][1], 4), round((1 - res["empnoise"][0] / base) * 100, 1)])
    print(f"\nwall={wall:.0f}s  peak_mem={peak_mb:.0f}MB  -> saved figure + CSV")


if __name__ == "__main__":
    main()
