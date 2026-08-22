"""
Nested-vs-independent seeding demonstration (Appendix A figure).

Shows that reading budget points off a single nested shadow stream (prefixes) understates the
across-budget error bars relative to drawing an independent dataset at each budget. Self-contained:
- true <X0>(t) from the 2-qubit TFIM (|++>, H = Z0Z1 + 0.5(X0+X1)) via qutip;
- single-qubit shadow estimate of <X0> simulated directly (single-shot = 3*outcome if the random
  basis is X, else 0 -- the k=1 shadow rule), so no reconstruction-pipeline coupling;
- reconstruction panel = a fixed-lengthscale GP regression of the estimate over time (the observable
  route, with the lengthscale FIXED so GP tuning is not a confound of the seeding effect).

Key point: at a single budget N both schemes give the same per-N variance; the artefact is the
CORRELATION across budgets. So we plot per-seed traces (nested = smooth/parallel, independent = jagged)
and quantify the SE of a budget contrast err(N_a) - err(N_b): nested << independent.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qutip as qt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
import os as _os
SP = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "out")  # repo-relative output dir
_os.makedirs(SP, exist_ok=True)

SEEDS = list(range(10, 201, 10))          # 20 seeds
BUDGETS = [50, 100, 200, 400, 800]        # shadows per observed time
NMAX = max(BUDGETS)
NUM_TRUE = 400                            # dense time grid for the true curve / RMSE
NUM_OBS = 40                              # observed time points
T0, T1 = 0.0, 2.0 * np.pi
CONTRAST = (100, 400)                     # the budget pair used for the SE-of-difference headline
GP_ELL = 0.45                             # fixed GP lengthscale (time units)


def true_x0(tgrid):
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = qt.tensor(plus, plus)
    H = (qt.tensor(qt.sigmaz(), qt.sigmaz())
         + 0.5 * (qt.tensor(qt.sigmax(), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.sigmax())))
    X0 = qt.tensor(qt.sigmax(), qt.qeye(2))
    return np.asarray(qt.sesolve(H, psi0, tgrid, e_ops=[X0]).expect[0])


def raw_stream(x_obs, nmax, rng):
    """Per observed time: basis (0=X,1=Y,2=Z) and outcome (+/-1) for nmax shadows."""
    streams = []
    for x in x_obs:
        basis = rng.integers(0, 3, size=nmax)
        out = np.where(rng.random(nmax) < (1.0 + x) / 2.0, 1.0, -1.0)
        streams.append((basis, out))
    return streams


def estimate_prefix(streams, n):
    """<X0> estimate at each observed time using the first n shadows of each stream."""
    est = np.empty(len(streams))
    for i, (basis, out) in enumerate(streams):
        b, o = basis[:n], out[:n]
        est[i] = np.where(b == 0, 3.0 * o, 0.0).mean()
    return est


def gp_reconstruct(obs_t, y, dense_t, ell, noise_var):
    d2 = (obs_t[:, None] - obs_t[None, :]) ** 2
    K = np.exp(-0.5 * d2 / ell ** 2) + noise_var * np.eye(len(obs_t))
    Ks = np.exp(-0.5 * (dense_t[:, None] - obs_t[None, :]) ** 2 / ell ** 2)
    return Ks @ np.linalg.solve(K, y)


def cell_seed(seed, budget):
    return int(np.random.SeedSequence([int(seed), int(budget)]).generate_state(1, np.uint32)[0])


def main():
    dense_t = np.linspace(T0, T1, NUM_TRUE)
    x_dense = true_x0(dense_t)
    obs_idx = np.linspace(0, NUM_TRUE - 1, NUM_OBS).astype(int)
    obs_t = dense_t[obs_idx]
    x_obs = x_dense[obs_idx]

    # error matrices [n_seeds, n_budgets] for each (arm, level)
    shapes = (len(SEEDS), len(BUDGETS))
    raw = {"nested": np.zeros(shapes), "independent": np.zeros(shapes)}
    rec = {"nested": np.zeros(shapes), "independent": np.zeros(shapes)}

    for si, seed in enumerate(SEEDS):
        # nested: ONE stream of NMAX, read prefixes
        nested_stream = raw_stream(x_obs, NMAX, np.random.default_rng(seed))
        for bi, N in enumerate(BUDGETS):
            est_nest = estimate_prefix(nested_stream, N)
            raw["nested"][si, bi] = np.sqrt(np.mean((est_nest - x_obs) ** 2))
            rec_nest = gp_reconstruct(obs_t, est_nest, dense_t, GP_ELL, 3.0 / N)
            rec["nested"][si, bi] = np.sqrt(np.mean((rec_nest - x_dense) ** 2))

            # independent: fresh stream of exactly N with an independent per-cell seed
            indep_stream = raw_stream(x_obs, N, np.random.default_rng(cell_seed(seed, N)))
            est_ind = estimate_prefix(indep_stream, N)
            raw["independent"][si, bi] = np.sqrt(np.mean((est_ind - x_obs) ** 2))
            rec_ind = gp_reconstruct(obs_t, est_ind, dense_t, GP_ELL, 3.0 / N)
            rec["independent"][si, bi] = np.sqrt(np.mean((rec_ind - x_dense) ** 2))
        print(f"seed {seed} done", flush=True)

    ca, cb = CONTRAST
    ia, ib = BUDGETS.index(ca), BUDGETS.index(cb)

    def contrast_stats(mat):
        d = mat[:, ia] - mat[:, ib]          # err(N_a) - err(N_b) per seed
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), se

    lines = ["level,arm,mean_diff,se_diff,naive_significance"]
    print("\n=== SE of the budget contrast  err(%d) - err(%d) ===" % (ca, cb))
    for level, store in (("raw_estimate", raw), ("gp_reconstruction", rec)):
        for arm in ("nested", "independent"):
            m, se = contrast_stats(store[arm])
            sig = m / se if se > 0 else float("nan")
            print(f"  {level:17s} {arm:11s}  diff={m:.4f}  SE={se:.4f}  significance={sig:.1f} sigma")
            lines.append(f"{level},{arm},{m:.5f},{se:.5f},{sig:.2f}")
    open(os.path.join(SP, "nested_vs_independent_summary.csv"), "w").write("\n".join(lines) + "\n")

    # ---- figure: two panels, faint per-seed traces + mean, for both arms ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    colors = {"nested": "#c0392b", "independent": "#2471a3"}
    for ax, (level, store, title) in zip(
        axes,
        (("raw", raw, "Raw shadow estimate"), ("rec", rec, "GP reconstruction")),
    ):
        for arm in ("nested", "independent"):
            M = store[arm]
            for si in range(len(SEEDS)):
                ax.plot(BUDGETS, M[si], color=colors[arm], alpha=0.13, lw=0.8)
            ax.plot(BUDGETS, M.mean(0), color=colors[arm], lw=2.4, label=arm)
        m_n, se_n = contrast_stats(store[arm if False else "nested"])
        m_i, se_i = contrast_stats(store["independent"])
        ax.set_title(title)
        ax.set_xlabel("measurement budget (shadows per observed time; 40 times per run)")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.grid(alpha=0.25)
        ax.text(0.97, 0.95,
                f"SE of err({ca})-err({cb}):\n nested   {se_n:.4f}\n indep.   {se_i:.4f}"
                f"\n ratio    {se_i/se_n:.1f}x",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    axes[0].set_ylabel("error  ( <X0> RMS / reconstruction RMSE )")
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Nested prefixes understate across-budget error bars "
                 "(nested traces move together; independent draws scatter)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for dst in (os.path.join(OUT_DIR, "nested_vs_independent.png"),
                os.path.join(SP, "nested_vs_independent.png")):
        fig.savefig(dst, dpi=150); fig.savefig(dst.replace(".png",".pdf"))
    print("\nsaved nested_vs_independent.png + summary.csv")


if __name__ == "__main__":
    main()
