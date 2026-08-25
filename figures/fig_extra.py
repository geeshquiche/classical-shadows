import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures","out") +"/"
_PER_ELEM = _os.path.join(_ROOT,"per_element_rho") +"/"
_EST = _os.path.join(_ROOT,"estimator_comparison") +"/"
_STUDIES = _os.path.join(_ROOT,"studies") +"/"
_os.makedirs(_OUT, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401  (sets shared rcParams)

SP = _OUT
OUT = _OUT


def fig_empnoise():
    # forest plot: per-element variants against the shared baseline (20 seeds)
    names = ["per-element,\nempirical noise", "per-element,\nbounded", "shared", "grouped",
             "per-element,\nrestarts", "per-element,\ngrid search", "per-element,\nfully fitted"]
    rmse = [0.0303, 0.0327, 0.0333, 0.0337, 0.0364, 0.0373, 0.0377]
    se = [0.0004, 0.0005, 0.0004, 0.0004, 0.0007, 0.0007, 0.0005]
    colors = ["#27ae60","#cd6155","#2c3e50","#7f8c8d","#cd6155","#cd6155","#cd6155"]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    ax.axvline(0.0333, ls="--", color="#2c3e50", lw=1.1, alpha=.8, zorder=0)
    ax.errorbar(rmse, y, xerr=se, fmt="none", ecolor="k", elinewidth=1.1, capsize=3, zorder=1)
    ax.scatter(rmse, y, s=70, c=colors, edgecolor="k", linewidth=.6, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel(r"Frobenius error of $\rho(t)$")
    ax.text(0.0333, y[0] + 0.42,"shared baseline", fontsize=9, color="#2c3e50", ha="center")
    ax.set_ylim(y[-1] - 0.8, y[0] + 0.9)
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(_OUT +"empnoise.png", dpi=150); fig.savefig(_OUT +"empnoise.pdf")
    print("empnoise fig done")


def fig_locality():
    # two panels: variance vs observable weight, and variance vs system size at fixed weight
    rows = [l.split(",") for l in open(_PER_ELEM +"mll_k_vs_n.csv").read().splitlines()
            if l and not l.startswith("#") and not l.startswith("sweep")]
    kd = sorted([(int(r[2]), float(r[4]), float(r[5])) for r in rows if"k-dep" in r[0]])
    nd = sorted([(int(r[1]), float(r[4]), float(r[5])) for r in rows if"n-indep" in r[0]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 3.3))
    ks = np.array([k for k, _, _ in kd]); vs = np.array([v for _, v, _ in kd])
    es = np.array([e for _, _, e in kd])
    a1.plot(ks, 3.0 ** ks,"k--", lw=1.3, label=r"$3^k$")
    a1.errorbar(ks, vs, yerr=es, fmt="o", ms=9, color="#e67e22", capsize=4, label="measured")
    a1.set_yscale("log"); a1.set_xticks(ks)
    a1.set_yticks([3, 9, 27]); a1.set_yticklabels(["3","9","27"])
    a1.set_xlabel("observable weight $k$ (at $n=4$)")
    a1.set_ylabel("per-shadow variance")
    a1.legend(fontsize=10); a1.grid(alpha=.25, which="both")
    ns = np.array([n for n, _, _ in nd]); vn = np.array([v for _, v, _ in nd])
    en = np.array([e for _, _, e in nd])
    a2.axhline(9, ls="--", color="k", lw=1.3, label=r"$3^k=9$")
    a2.errorbar(ns, vn, yerr=en, fmt="s", ms=9, color="#2471a3", capsize=4, label="measured")
    a2.set_ylim(8.0, 9.4); a2.set_xticks(ns)
    a2.set_xlabel("chain length $n$ (at $k=2$)")
    a2.set_ylabel("per-shadow variance")
    a2.legend(fontsize=10); a2.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(_OUT +"locality.png", dpi=150); fig.savefig(_OUT +"locality.pdf")
    print("locality fig done")


fig_empnoise()
fig_locality()
