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
ARMS = [("rerandomised","matched","#2471a3","-","o","matched count, fresh bases"),
        ("fixed","matched","#148f77","--","s","matched count, fixed bases"),
        ("rerandomised","expected","#e67e22","-.","^","expected rate, fresh bases"),
        ("fixed","expected","#c0392b",":","D","expected rate, fixed bases")]

data = {}  # (tag, protocol, norm) -> (cov, cov_se, rmse, rmse_se)
for tag in TAGS:
    with open(RA + f"final_config_coverage_{tag}_summary.csv") as f:
        next(f)
        for r in csv.DictReader(f):
            if r["observable"] =="ZZ" and r["noise"] =="empirical":
                data[(tag, r["protocol"], r["normalisation"])] = (
                    float(r["coverage_mean"]), float(r["coverage_se"]),
                    float(r["rmse_mean"]), float(r["rmse_se"]))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 3.0))
for prot, norm, c, ls, mk, lab in ARMS:
    cov = [data[(t, prot, norm)][0] for t in TAGS]
    cse = [data[(t, prot, norm)][1] for t in TAGS]
    rm = [data[(t, prot, norm)][2] for t in TAGS]
    rse = [data[(t, prot, norm)][3] for t in TAGS]
    a1.errorbar(BUDGET, cov, yerr=cse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)
    a2.errorbar(BUDGET, rm, yerr=rse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)
a1.axhline(0.95, color="k", ls="--", lw=1.1)
a1.text(BUDGET[0], 0.952,"nominal 0.95", fontsize=9)
a1.set_xscale("log")
a1.set_xticks(BUDGET)
a1.set_xticklabels([t.replace("x", r"$\times$") for t in TAGS], rotation=45, ha="right", fontsize=9.5)
a1.minorticks_off()
a1.set_ylabel("coverage of the 95% band")

a1.set_ylim(0.88, 1.02)
a1.grid(alpha=.25)
a1.legend(fontsize=9, loc="lower right", framealpha=.92)
a2.set_xscale("log")
a2.set_yscale("log")
a2.set_xticks(BUDGET)
a2.set_xticklabels([t.replace("x", r"$\times$") for t in TAGS], rotation=45, ha="right", fontsize=9.5)
a2.minorticks_off()
a2.set_ylabel(r"RMSE of $\langle Z_0Z_1\rangle$")

lo, hi = a2.get_ylim()
ticks = [t for t in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3) if lo <= t <= hi]
a2.set_yticks(ticks)
a2.set_yticklabels([f"{t:g}" for t in ticks])
a2.grid(alpha=.25, which="both")
# place the layout explicitly: tight_layout over-reserves under rotated tick labels and
# leaves a dead band between the ticks and the shared x label
fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.28, wspace=0.30)
fig.supxlabel(r"sampling density (observed times $\times$ shadows per time)", fontsize=11, y=0.035)
for base in (OUT, SP):
    fig.savefig(base +"coverage_ladder.pdf")
    fig.savefig(base +"coverage_ladder.png", dpi=150)
print("coverage_ladder done")
