import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures", "out") + "/"
_PER_ELEM = _os.path.join(_ROOT, "per_element_rho") + "/"
_EST = _os.path.join(_ROOT, "estimator_comparison") + "/"
_STUDIES = _os.path.join(_ROOT, "studies") + "/"
_os.makedirs(_OUT, exist_ok=True)
"""Budget-sweep curves and smoothers-ladder curves for the report (from chain D CSVs)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RA = _STUDIES
OUT = _OUT
def load(p):
    with open(p) as f:
        next(f); return list(csv.DictReader(f))
# ---- budget sweep: RMSE vs per-time budget, allocation x noise, two observable panels ----
rows = load(RA + "budget_sweep_summary.csv")
B = sorted({int(r["budget_per_time"]) for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
style = {("1x", "empirical"): ("#2471a3", "-", "o", r"$1\times B$, measured noise"),
         ("2x", "empirical"): ("#148f77", "-", "s", r"$2\times B/2$, measured noise"),
         ("5x", "empirical"): ("#c0392b", "-", "^", r"$5\times B/5$, measured noise"),
         ("1x", "fitted"): ("#2471a3", ":", "o", r"$1\times B$, fitted noise"),
         ("2x", "fitted"): ("#148f77", ":", "s", r"$2\times B/2$, fitted noise"),
         ("5x", "fitted"): ("#c0392b", ":", "^", r"$5\times B/5$, fitted noise")}
for ax, obs, tt in zip(axes, ["XI", "ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    for (alloc, noise), (c, ls, mk, lab) in style.items():
        m = [float(next(r for r in rows if r["observable"] == obs and int(r["budget_per_time"]) == b and r["allocation"] == alloc and r["noise"] == noise)["rmse_mean"]) for b in B]
        se = [float(next(r for r in rows if r["observable"] == obs and int(r["budget_per_time"]) == b and r["allocation"] == alloc and r["noise"] == noise)["rmse_se"]) for b in B]
        ax.errorbar(B, m, yerr=se, color=c, ls=ls, marker=mk, ms=4.5, lw=1.6, capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(B); ax.set_xticklabels(B); ax.minorticks_off()
    ax.set_xlabel("shadow budget per observed time, $B$"); ax.set_title(tt); ax.grid(alpha=.25, which="both")
axes[0].set_ylabel("RMSE"); axes[1].legend(fontsize=7.5, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "budget_sweep.pdf"); fig.savefig(OUT + "budget_sweep.png", dpi=150)
# ---- smoothers ladder: RMSE vs density, gp / spline / savgol, two observable panels ----
rows = load(RA + "smoothers_ladder_summary.csv")
dens = sorted({(int(r["times"]), int(r["shadows"])) for r in rows})
xs = [t * s for t, s in dens]; labels = [f"{t}$\\times${s}" for t, s in dens]
fig2, axes2 = plt.subplots(1, 2, figsize=(11.2, 4.2))
st = {"gp": ("#2471a3", "-", "o", "GP, measured noise (this work)"), "spline_gcv": ("#e67e22", "--", "s", "smoothing spline (GCV)"),
      "savgol": ("#7f8c8d", "-.", "^", "Savitzky--Golay")}
for ax, obs, tt in zip(axes2, ["XI", "ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    for m, (c, ls, mk, lab) in st.items():
        y = [float(next(r for r in rows if r["observable"] == obs and int(r["times"]) == t and int(r["shadows"]) == s and r["method"] == m)["rmse_mean"]) for t, s in dens]
        se = [float(next(r for r in rows if r["observable"] == obs and int(r["times"]) == t and int(r["shadows"]) == s and r["method"] == m)["rmse_se"]) for t, s in dens]
        ax.errorbar(xs, y, yerr=se, color=c, ls=ls, marker=mk, ms=4.5, lw=1.6, capsize=2, label=lab.replace("--", "–"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.minorticks_off()
    ax.set_xlabel(r"sampling density (times $\times$ shadows)"); ax.set_title(tt); ax.grid(alpha=.25, which="both")
axes2[0].set_ylabel("RMSE"); axes2[0].legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(OUT + "smoothers_ladder.pdf"); fig2.savefig(OUT + "smoothers_ladder.png", dpi=150)
print("budget_sweep + smoothers_ladder figures done")
