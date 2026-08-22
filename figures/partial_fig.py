import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures", "out") + "/"
_PER_ELEM = _os.path.join(_ROOT, "per_element_rho") + "/"
_EST = _os.path.join(_ROOT, "estimator_comparison") + "/"
_STUDIES = _os.path.join(_ROOT, "studies") + "/"
_os.makedirs(_OUT, exist_ok=True)
"""partial_vs_full_rho figure from partial_final_summary.csv (matched-Pauli program)."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA = _STUDIES
OUT = _OUT
data = {}
with open(RA + "partial_final_summary.csv") as f:
    next(f)
    for r in csv.DictReader(f):
        data.setdefault(r["ordering"], []).append(
            (float(r["fraction"]), float(r["frob_mean"]), float(r["frob_se"])))
fig, ax = plt.subplots(figsize=(7.0, 4.3))
for ordering, c, mk in [("magnitude", "#2471a3", "o"), ("diagonal-first", "#c0392b", "s"),
                        ("random", "#27ae60", "^")]:
    pts = sorted(data[ordering])
    fr = [p[0] for p in pts]; m = [p[1] for p in pts]; se = [p[2] for p in pts]
    ax.errorbar(fr, m, yerr=se, marker=mk, ms=4, lw=1.5, capsize=2, color=c, label=ordering)
ax.set_xlabel("fraction of independent entries reconstructed")
ax.set_ylabel(r"Frobenius error of the reduced block")
ax.grid(alpha=.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT + "partial_vs_full_rho.pdf")
fig.savefig(OUT + "partial_vs_full_rho.png", dpi=150)
print("partial fig done")
