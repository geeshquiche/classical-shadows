import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures", "out") + "/"
_PER_ELEM = _os.path.join(_ROOT, "per_element_rho") + "/"
_EST = _os.path.join(_ROOT, "estimator_comparison") + "/"
_STUDIES = _os.path.join(_ROOT, "studies") + "/"
_os.makedirs(_OUT, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = _OUT
OUT = _OUT


def fig_empnoise():
    # matched-Pauli program, review_ablations/rho_final_variants_summary.csv (20 seeds, 2026-08-21)
    names = ["per-elem\n(empnoise)", "per-elem\nbound", "shared", "grouped",
             "per-elem\nrestart", "per-elem\ngrid", "per-elem\n(fully fitted)"]
    rmse = [0.0303, 0.0327, 0.0333, 0.0337, 0.0364, 0.0373, 0.0377]
    se = [0.0004, 0.0005, 0.0004, 0.0004, 0.0007, 0.0007, 0.0005]
    colors = ["#27ae60", "#cd6155", "#2c3e50", "#7f8c8d", "#cd6155", "#cd6155", "#cd6155"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(7), rmse, yerr=se, capsize=3, color=colors, edgecolor="k", lw=.5)
    ax.axhline(0.0333, ls="--", color="#2c3e50", lw=1, alpha=.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"reconstruction RMSE (full $\rho$)")
    ax.set_ylim(0.028, 0.040)
    ax.set_title("Per-element beats a shared model only when the noise is fixed to the shadow variance\n"
                 "(20 seeds; dashed line = shared baseline)", fontsize=9.5)
    ax.annotate("fixing the noise\n($-9.1\\%$)", xy=(0, 0.0303), xytext=(0.55, 0.0305),
                fontsize=8, color="#1e8449", arrowprops=dict(arrowstyle="->", color="#1e8449"))
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT + "empnoise.png", dpi=150); fig.savefig(OUT + "empnoise.pdf")
    fig.savefig(SP + "/empnoise.png", dpi=150); fig.savefig(SP + "/empnoise.pdf")
    print("empnoise fig done")


def fig_locality():
    k = np.array([1, 2, 3])
    meas = np.array([2.98, 8.69, 26.76])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(k, 3.0 ** k, "k--", lw=1.2, label=r"$3^k$")
    ax.plot(k, meas, "o", ms=9, color="#e67e22", label="measured shadow variance")
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlabel("observable weight $k$")
    ax.set_ylabel("per-shadow variance")
    ax.set_title(r"Shadow variance grows as $3^k$ with weight and is independent of system size"
                 "\n(4-qubit chain; for $k=2$ it holds flat at $\\approx 8.6$ for $n=2$ to $6$)", fontsize=9.5)
    ax.legend()
    ax.grid(alpha=.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT + "locality.png", dpi=150); fig.savefig(OUT + "locality.pdf")
    fig.savefig(SP + "/locality.png", dpi=150); fig.savefig(SP + "/locality.pdf")
    print("locality fig done")


fig_empnoise()
fig_locality()
