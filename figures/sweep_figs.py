import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures","out") +"/"
_PER_ELEM = _os.path.join(_ROOT,"per_element_rho") +"/"
_EST = _os.path.join(_ROOT,"estimator_comparison") +"/"
_STUDIES = _os.path.join(_ROOT,"studies") +"/"
_os.makedirs(_OUT, exist_ok=True)
"""Budget-sweep curves and smoothers-ladder curves for the report (from chain D CSVs)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401
RA = _STUDIES
OUT = _OUT
def load(p):
    with open(p) as f:
        next(f); return list(csv.DictReader(f))
# ---- budget sweep: RMSE vs per-time budget, allocation x noise, two observable panels ----
rows = load(_STUDIES + "budget_sweep_summary.csv")
B = sorted({int(r["budget_per_time"]) for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(6.7, 4.4), sharey=False)
style = {("1x", "empirical"): ("#2471a3", "-", "o", r"$1\times B$"),
         ("2x", "empirical"): ("#148f77", "-", "s", r"$2\times B/2$"),
         ("5x", "empirical"): ("#c0392b", "-", "^", r"$5\times B/5$"),
         ("1x", "fitted"): ("#2471a3", ":", "o", None),
         ("2x", "fitted"): ("#148f77", ":", "s", None),
         ("5x", "fitted"): ("#c0392b", ":", "^", None)}
for ax, obs, tt in zip(axes, ["XI", "ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    for (alloc, noise), (c, ls, mk, lab) in style.items():
        sel = [r for r in rows if r["observable"] == obs and r["allocation"] == alloc and r["noise"] == noise]
        m = [float(next(r for r in sel if int(r["budget_per_time"]) == b)["rmse_mean"]) for b in B]
        se = [float(next(r for r in sel if int(r["budget_per_time"]) == b)["rmse_se"]) for b in B]
        ax.errorbar(B, m, yerr=se, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2,
                    label=lab if lab else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(B); ax.set_xticklabels(B); ax.minorticks_off()
    # a log axis spanning less than a decade labels only one tick by default; label the
    # values actually covered so the reader can read an RMSE off the axis
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.006, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("budget per time, $B$")
    ax.set_title(tt); ax.grid(alpha=.25, which="both")
axes[0].set_ylabel("RMSE")
h, l = axes[0].get_legend_handles_labels()
leg1 = fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
from matplotlib.lines import Line2D
style_key = [Line2D([], [], color="0.35", ls="-", label="measured noise"),
             Line2D([], [], color="0.35", ls=":", label="fitted noise")]
fig.legend(handles=style_key, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.075))
fig.add_artist(leg1)
fig.subplots_adjust(bottom=0.30, wspace=0.28, top=0.90)
fig.savefig(_OUT + "budget_sweep.pdf"); fig.savefig(_OUT + "budget_sweep.png", dpi=150)

# ---- smoothers ladder: RMSE vs density, gp / spline / savgol, two observable panels ----
rows = load(RA +"smoothers_ladder_summary.csv")
dens = sorted({(int(r["times"]), int(r["shadows"])) for r in rows})
xs = [t * s for t, s in dens]; labels = [f"{t}$\\times${s}" for t, s in dens]
fig2, axes2 = plt.subplots(1, 2, figsize=(6.7, 3.4))
st = {"gp": ("#2471a3","-","o","GP, measured noise"),"spline_gcv": ("#e67e22","--","s","smoothing spline (GCV)"),
"savgol": ("#7f8c8d","-.","^","Savitzky--Golay")}
for ax, obs, tt in zip(axes2, ["XI","ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    for m, (c, ls, mk, lab) in st.items():
        y = [float(next(r for r in rows if r["observable"] == obs and int(r["times"]) == t and int(r["shadows"]) == s and r["method"] == m)["rmse_mean"]) for t, s in dens]
        se = [float(next(r for r in rows if r["observable"] == obs and int(r["times"]) == t and int(r["shadows"]) == s and r["method"] == m)["rmse_se"]) for t, s in dens]
        ax.errorbar(xs, y, yerr=se, color=c, ls=ls, marker=mk, ms=4.5, lw=1.6, capsize=2, label=lab.replace("--","–"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5); ax.minorticks_off()
    ax.set_title(tt); ax.grid(alpha=.25, which="both")
axes2[0].set_ylabel("RMSE")
for ax in axes2:
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5) if lo <= t <= hi]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
# the legend must be a FIGURE legend: an axes legend anchored outside its axes is pulled into the
# tight bounding box at save time, which stretches the canvas and shrinks the panels
h2, l2 = axes2[0].get_legend_handles_labels()
fig2.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.34, wspace=0.26)
fig2.legend(h2, l2, fontsize=8.8, ncol=3, loc="lower center", frameon=False,
            bbox_to_anchor=(0.5, 0.005), columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
fig2.supxlabel(r"sampling density (observed times $\times$ shadows)", fontsize=11, y=0.10)
report_style.save_exact(fig2, [_OUT +"smoothers_ladder.pdf", _OUT +"smoothers_ladder.png"])
print("budget_sweep + smoothers_ladder figures done")
