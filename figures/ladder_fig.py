import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures", "out") + "/"
_PER_ELEM = _os.path.join(_ROOT, "per_element_rho") + "/"
_EST = _os.path.join(_ROOT, "estimator_comparison") + "/"
_STUDIES = _os.path.join(_ROOT, "studies") + "/"
_os.makedirs(_OUT, exist_ok=True)
"""Sensitivity ladder figure: ZZ coverage + RMSE vs sampling density, 4 arms, empirical noise.
Data: review_ablations/final_config_coverage_{tag}_summary.csv (16-arm grid, 20 seeds each)."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA = _STUDIES
OUT = _OUT
SP = _OUT

TAGS = ["20x60", "40x100", "60x150", "100x200", "150x300", "200x400"]
BUDGET = [20 * 60, 40 * 100, 60 * 150, 100 * 200, 150 * 300, 200 * 400]
ARMS = [("rerandomised", "matched", "#2471a3", "-", "o", "realised count, fresh bases (this work)"),
        ("fixed", "matched", "#148f77", "--", "s", "realised count, fixed bases"),
        ("rerandomised", "expected", "#e67e22", "-.", "^", "expected rate, fresh bases"),
        ("fixed", "expected", "#c0392b", ":", "D", "expected rate, fixed bases")]

data = {}  # (tag, protocol, norm) -> (cov, cov_se, rmse, rmse_se)
for tag in TAGS:
    with open(RA + f"final_config_coverage_{tag}_summary.csv") as f:
        next(f)
        for r in csv.DictReader(f):
            if r["observable"] == "ZZ" and r["noise"] == "empirical":
                data[(tag, r["protocol"], r["normalisation"])] = (
                    float(r["coverage_mean"]), float(r["coverage_se"]),
                    float(r["rmse_mean"]), float(r["rmse_se"]))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for prot, norm, c, ls, mk, lab in ARMS:
    cov = [data[(t, prot, norm)][0] for t in TAGS]
    cse = [data[(t, prot, norm)][1] for t in TAGS]
    rm = [data[(t, prot, norm)][2] for t in TAGS]
    rse = [data[(t, prot, norm)][3] for t in TAGS]
    a1.errorbar(BUDGET, cov, yerr=cse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)
    a2.errorbar(BUDGET, rm, yerr=rse, color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, label=lab)
a1.axhline(0.95, color="k", ls="--", lw=1.1)
a1.text(BUDGET[0], 0.952, "nominal 0.95", fontsize=8)
a1.set_xscale("log")
a1.set_xticks(BUDGET)
a1.set_xticklabels([t.replace("x", r"$\times$") for t in TAGS], fontsize=8)
a1.minorticks_off()
a1.set_ylabel("empirical coverage of the nominal 95% band")
a1.set_xlabel(r"sampling density (times $\times$ shadows per time)")
a1.set_ylim(0.88, 1.02)
a1.grid(alpha=.25)
a1.legend(fontsize=7.5, loc="lower right")
a2.set_xscale("log")
a2.set_yscale("log")
a2.set_xticks(BUDGET)
a2.set_xticklabels([t.replace("x", r"$\times$") for t in TAGS], fontsize=8)
a2.minorticks_off()
a2.set_ylabel(r"RMSE of $\langle Z_0Z_1\rangle$")
a2.set_xlabel(r"sampling density (times $\times$ shadows per time)")
a2.grid(alpha=.25, which="both")
fig.tight_layout()
for base in (OUT, SP):
    fig.savefig(base + "coverage_ladder.pdf")
    fig.savefig(base + "coverage_ladder.png", dpi=150)
print("coverage_ladder done")
