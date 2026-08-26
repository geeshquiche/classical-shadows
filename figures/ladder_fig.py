import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures","out") +"/"
_PER_ELEM = _os.path.join(_ROOT,"per_element_rho") +"/"
_EST = _os.path.join(_ROOT,"estimator_comparison") +"/"
_STUDIES = _os.path.join(_ROOT,"studies") +"/"
_os.makedirs(_OUT, exist_ok=True)
"""Sensitivity ladder figure: ZZ coverage + RMSE vs sampling density, 4 arms, empirical noise.
Data: review_ablations/final_config_coverage_{tag}_summary.csv (16-arm grid, 20 seeds each)."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401  (sets shared rcParams)

RA = _STUDIES
OUT = _OUT
SP = _OUT

TAGS = ["20x60","40x100","60x150","100x200","150x300","200x400"]
BUDGET = [20 * 60, 40 * 100, 60 * 150, 100 * 200, 150 * 300, 200 * 400]
ARMS = [("rerandomised","matched","#2471a3","-","o","matched count, fresh"),
        ("fixed","matched","#148f77","--","s","matched count, fixed"),
        ("rerandomised","expected","#e67e22","-.","^","expected rate, fresh"),
        ("fixed","expected","#c0392b",":","D","expected rate, fixed")]

data = {}  # (tag, protocol, norm) -> (cov, cov_se, rmse, rmse_se)
for tag in TAGS:
    with open(RA + f"final_config_coverage_{tag}_summary.csv") as f:
        next(f)
        for r in csv.DictReader(f):
            if r["observable"] =="ZZ" and r["noise"] =="empirical":
                data[(tag, r["protocol"], r["normalisation"])] = (
                    float(r["coverage_mean"]), float(r["coverage_se"]),
                    float(r["rmse_mean"]), float(r["rmse_se"]))

# The six configurations are discrete settings, not samples of a continuous variable, so they are
# plotted at even spacing.  On the log-budget axis used previously the two densest settings sat almost
# on top of each other and their tick labels collided.
X = np.arange(len(TAGS))
LABELS = [t.replace("x", "\n$\\times$") for t in TAGS]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 3.15))
for prot, norm, c, ls, mk, lab in ARMS:
    cov = [data[(t, prot, norm)][0] for t in TAGS]
    cse = [data[(t, prot, norm)][1] for t in TAGS]
    rm = [data[(t, prot, norm)][2] for t in TAGS]
    rse = [data[(t, prot, norm)][3] for t in TAGS]
    a1.errorbar(X, cov, yerr=cse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)
    a2.errorbar(X, rm, yerr=rse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)

a1.axhline(0.95, color="k", ls="--", lw=1.1)
a1.text(X[-1], 0.9525, "nominal 0.95", fontsize=9, ha="right", va="bottom")
a1.set_ylim(0.915, 1.012)
a1.set_ylabel("coverage of the 95% band")

a2.set_yscale("log")
lo, hi = a2.get_ylim()
ticks = [t for t in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3) if lo <= t <= hi]
a2.set_yticks(ticks); a2.set_yticklabels([f"{t:g}" for t in ticks])
a2.set_ylabel(r"RMSE of $\langle Z_0Z_1\rangle$")

for ax in (a1, a2):
    ax.set_xticks(X); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_xlim(-0.35, len(TAGS) - 0.65)
    ax.grid(alpha=.25, which="both")

# four long entries fit only as a figure legend across the full width
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, fontsize=8.8, ncol=2, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.34, wspace=0.30)
fig.supxlabel(r"sampling density (observed times $\times$ shadows per time)", fontsize=11, y=0.165)
for base in (OUT, SP):
    report_style.save_exact(fig, [base +"coverage_ladder.pdf", base +"coverage_ladder.png"])
print("coverage_ladder done")
