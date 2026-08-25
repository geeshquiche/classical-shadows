import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures","out") +"/"
_PER_ELEM = _os.path.join(_ROOT,"per_element_rho") +"/"
_EST = _os.path.join(_ROOT,"estimator_comparison") +"/"
_STUDIES = _os.path.join(_ROOT,"studies") +"/"
_os.makedirs(_OUT, exist_ok=True)
"""Regenerate the CSV-derived report figures, writing PNG + vector PDF into report_build."""
import csv, io, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401  (sets shared rcParams)
import matplotlib.patches as mp
import qutip as qt

RB = _OUT
FF = _PER_ELEM
EC = _EST
SP = _OUT


def load(p):
    lines = [l for l in open(p) if not l.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def save(fig, name):
    fig.savefig(RB + name +".png", dpi=140)
    fig.savefig(RB + name +".pdf")
    plt.close(fig)
    print(name,"done")


# ---- coverage bar ----
# this work (final_config_coverage_summary.csv: rerand+matched+empirical) vs the two naive
# alternatives: MLL-fitted noise in the GPyTorch route implementation (test_pipeline_coverage:
# 0.783/0.892) and expected-rate normalisation under fixed bases (grid: 0.980/0.915).
fig, ax = plt.subplots(figsize=(6.7, 3.7))
xs = [0, 0.34, 0.68, 1.5, 1.84, 2.18]
vals = [1.000, 0.783, 0.980, 0.996, 0.892, 0.915]
cols = ["#2471a3","#c0392b","#e59866"] * 2
ax.bar(xs, vals, 0.30, color=cols, edgecolor="k", lw=.5)
ax.axhline(0.95, ls="--", color="k", lw=1.2)
ax.text(2.42, 0.952,"nominal 0.95", fontsize=9)
ax.set_xticks([0.34, 1.84])
ax.set_xticklabels([r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"])
ax.set_ylabel("empirical coverage of the nominal 95% band")
ax.set_ylim(0.6, 1.05)
ax.legend(handles=[mp.Patch(fc="#2471a3", ec="k", label="this work"),
                   mp.Patch(fc="#c0392b", ec="k", label="noise fitted by marginal likelihood"),
                   mp.Patch(fc="#e59866", ec="k", label="expected-rate normalisation, fixed bases")],
          fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
save(fig,"coverage_bar")

# ---- route robustness: two panels (XI from routes3_robustness_XI.csv, ZZ from routes3_robustness.csv) ----
RA3 = _STUDIES
colors = {"raw":"#95a5a6","gp":"#2471a3","conditional":"#e59866"}
labels = {"raw":"raw","gp":"GP regression","conditional":"conditional"}
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.4))
for ax, (obs, fname, tt) in zip(axes, [("XI","routes3_robustness_XI.csv", r"$\langle X_0\rangle$"),
                                      ("ZZ","routes3_robustness.csv", r"$\langle Z_0Z_1\rangle$")]):
    rows = load(RA3 + fname)
    for rt in colors:
        xs, ms, ses = [], [], []
        for b in [40, 120, 240]:
            v = [float(r["rmse"]) for r in rows if r["observable"] == obs and r["route"] == rt and int(r["shadow"]) == b]
            if v:
                xs.append(b); ms.append(np.mean(v)); ses.append(np.std(v, ddof=1) / np.sqrt(len(v)))
        ax.errorbar(xs, ms, yerr=ses, marker="o", ms=4.5, lw=1.7, capsize=3, color=colors[rt], label=labels[rt])
    ax.set_title(tt); ax.grid(alpha=.25)
    ax.set_xticks([40, 120, 240])
axes[0].set_ylabel("RMSE")
axes[0].legend(loc="upper right")
fig.supxlabel("shadows per observed time (budget)", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig,"route_robustness")

# ---- n-qubit scaling (matched-Pauli program, 2026-08-21) ----
RA = _STUDIES
rows = load(RA +"rho_final_nqubit_summary.csv")
by = {(r["qubits"], r["arm"]): (float(r["frob_mean"]), float(r["frob_se"])) for r in rows}
fig, ax = plt.subplots(figsize=(6.7, 3.0))
x = np.arange(2); w = 0.27
for i, (k, lab, c) in enumerate([("shared","shared","#2c3e50"),
                                 ("per-elem","fully fitted per-element","#c0392b"),
                                 ("empnoise","per-element, empirical noise","#27ae60")]):
    ax.bar(x + (i - 1) * w, [by[(q, k)][0] for q in ("2","3")], w,
           yerr=[by[(q, k)][1] for q in ("2","3")], capsize=3, color=c, edgecolor="k", lw=.5, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([r"2 qubits ($4\times4$)", r"3 qubits ($8\times8$)"])
ax.set_ylabel(r"Frobenius error of $\rho(t)$")
ax.legend(fontsize=9.5)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
save(fig,"nqubit_scaling")

# ---- 2x2 de-confound: interaction plot (rho_final_program MODE=core, 20 seeds) ----
rows = load(_STUDIES +"rho_final_core_summary.csv")
by = {r["arm"]: (float(r["frob_mean"]), float(r["frob_se"])) for r in rows}
fig, ax = plt.subplots(figsize=(6.7, 3.1))
xpos = [0, 1]
for arms, lab, c, mk in [(("shared","shared-empnoise"), r"shared $\ell$","#2c3e50","o"),
                         (("per-elem","empnoise"), r"per-element $\ell$","#27ae60","s")]:
    m = [by[a][0] for a in arms]; e = [by[a][1] for a in arms]
    ax.errorbar(xpos, m, yerr=e, marker=mk, ms=9, lw=1.8, capsize=4, color=c, label=lab)
ax.set_xticks(xpos)
ax.set_xticklabels([r"noise fitted", r"noise supplied""\n" r"(measured variance)"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel(r"Frobenius error of $\rho(t)$")
ax.legend(fontsize=10)
ax.grid(alpha=.25)
fig.tight_layout()
save(fig,"twobytwo")

# ---- budget comparison (matched-count estimator rerun, 2026-08-21) ----
rows = load(RA +"budget_final_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.4))
for ax, obs, tt in zip(axes, ["XI","ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    sub = {r["noise"]: r for r in rows if r["observable"] == obs}
    x = np.arange(2); w = 0.36
    for i, (cfgl, lab, col) in enumerate([("1x500","$1\\times500$","#5dade2"),
                                          ("5x100","$5\\times100$","#e59866")]):
        m = [float(sub[nz][f"rmse_{cfgl}"]) for nz in ("fitted","empirical")]
        se = [float(sub[nz][f"se_{cfgl}"]) for nz in ("fitted","empirical")]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=se, capsize=3, color=col, edgecolor="k", lw=.5, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["fitted noise","empirical noise"], fontsize=9)
    ax.set_title(tt)
    ax.grid(axis="y", alpha=.25)
axes[0].set_ylabel("RMSE")
axes[0].legend(fontsize=9.5)
fig.tight_layout()
save(fig,"budget_comparison")

# ---- shot scaling (matched-Pauli program, 10 seeds, 2026-08-21) ----
rows = load(RA +"rho_final_shots_summary.csv")
N = np.array(sorted({float(r["N"]) for r in rows}))
by = {(float(r["N"]), r["arm"]): (float(r["frob_mean"]), float(r["frob_se"])) for r in rows}
fig, ax = plt.subplots(figsize=(6.7, 3.1))
styles = {"shared": ("#2c3e50","shared"),"per-elem": ("#c0392b","fully fitted per-element"),
"empnoise": ("#27ae60","per-element, empirical noise")}
for k, (c, lab) in styles.items():
    y = np.array([by[(n, k)][0] for n in N])
    se = np.array([by[(n, k)][1] for n in N])
    ax.errorbar(N, y, yerr=se, marker="o", ms=4, lw=1.6, color=c, capsize=2, label=lab)
ax.plot(N, by[(N[0],"shared")][0] * np.sqrt(N[0] / N),"k--", lw=1.1, alpha=0.7, label=r"$N^{-1/2}$ guide")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(N); ax.set_xticklabels([int(n) for n in N]); ax.minorticks_off()
_yt = [0.01, 0.02, 0.03, 0.05, 0.08]
ax.set_yticks(_yt); ax.set_yticklabels([f"{v:g}" for v in _yt])
ax.set_xlabel("shadows per observed time, $N$")
ax.set_ylabel(r"Frobenius error of $\rho(t)$")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=9.5)
fig.tight_layout()
save(fig,"shot_scaling")

# ---- 8-Hamiltonian route comparison: final-method GP arm (recompute) vs stored conditional ----
rows = load(RA +"ham_gp_recompute.csv")
names = ["ZZ_slow","ZZ_fast","ZZ_plus_XI","weak","entangling","TFIM","beating","dephasing"]
# printed labels; the keys above are the column values in the CSV and must not change
PRETTY = {"ZZ_slow": "slow ZZ", "ZZ_fast": "fast ZZ", "ZZ_plus_XI": "ZZ + XI"}
def pretty(n): return PRETTY.get(n, n)
data = {}
for n in names:
    for rt, col_name in (("matrix","gp_rmse"), ("conditional","conditional_rmse")):
        v = sorted({int(r["seed"]): float(r[col_name]) for r in rows
                    if r["hamiltonian"] == n}.items())
        data[(n, rt)] = np.array([x[1] for x in v])
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(6.7, 3.2))
for i, (rt, col, lab) in enumerate([("matrix","#5dade2","GP regression"),
                                    ("conditional","#e59866","conditional (autoregressive classifier)")]):
    m = [data[(n, rt)].mean() for n in names]
    se = [data[(n, rt)].std(ddof=1) / np.sqrt(len(data[(n, rt)])) for n in names]
    ax.bar(x + (i - 0.5) * w, m, w, yerr=se, capsize=3, color=col, edgecolor="k", lw=.5, label=lab)
ns = {n: len(data[(n,"matrix")]) for n in names}
ax.set_xticks(x)
ax.set_xticklabels([f"{pretty(n)}\n({ns[n]} seeds)" for n in names], fontsize=9)
ax.set_ylabel(r"RMSE of $\langle X_0\rangle(t)$")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
save(fig,"hamiltonian_route_comparison")

# ---- Hamiltonian dynamics preview ----
ZZ = qt.tensor(qt.sigmaz(), qt.sigmaz()); XI = qt.tensor(qt.sigmax(), qt.qeye(2))
IX = qt.tensor(qt.qeye(2), qt.sigmax()); ZI = qt.tensor(qt.sigmaz(), qt.qeye(2))
IZ = qt.tensor(qt.qeye(2), qt.sigmaz()); XX = qt.tensor(qt.sigmax(), qt.sigmax())
YY = qt.tensor(qt.sigmay(), qt.sigmay())
HAMS = [("ZZ_slow", 0.5 * ZZ, []), ("ZZ_fast", 2.5 * ZZ, []), ("ZZ_plus_XI", ZZ + 0.5 * XI, []),
        ("weak", 0.05 * ZZ, []), ("entangling", XX + YY, []), ("TFIM", ZZ + 0.5 * XI + 0.5 * IX, []),
        ("beating", 0.4 * ZI + 0.6 * IZ + 0.3 * ZZ, []), ("dephasing", 0.5 * ZZ, [np.sqrt(0.3) * ZI])]
plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit(); psi0 = qt.tensor(plus, plus)
tl = np.linspace(0, 2 * np.pi, 300)
fig, axes = plt.subplots(2, 4, figsize=(6.7, 3.1), sharex=True, sharey=True)
for ax, (nm, H, c) in zip(axes.ravel(), HAMS):
    y = np.real(np.asarray(qt.expect(XI, qt.mesolve(H, psi0, tl, c).states)))
    ax.plot(tl, y, color="#2471a3", lw=1.5)
    ax.set_title(pretty(nm), fontsize=9.5)
    ax.grid(alpha=.25)
for ax in axes.ravel():
    ax.tick_params(labelsize=8.5)
    ax.set_xticks([0, 3, 6])
fig.supxlabel(r"time $t$ (units of $1/J$)", fontsize=11)
fig.supylabel(r"$\langle X_0\rangle$", fontsize=11)
fig.tight_layout()
save(fig,"hamiltonian_dynamics")

print("[csvfigs] all done")
