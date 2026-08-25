import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT, "figures","out") +"/"
_PER_ELEM = _os.path.join(_ROOT,"per_element_rho") +"/"
_EST = _os.path.join(_ROOT,"estimator_comparison") +"/"
_STUDIES = _os.path.join(_ROOT,"studies") +"/"
_os.makedirs(_OUT, exist_ok=True)
"""Generate clean report-quality figures. Independent seeds throughout."""
import csv, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401  (sets shared rcParams)
import qutip as qt

SP = _OUT
OUT = _OUT


def fig_route():
    # Values = 20-seed means/SEs of Table tab:routes (routes3_table_summary.csv, 2026-08-21,
    # final matched-count estimator; marginal route binned to a cautionary note).
    xm=[0.082,0.037,0.158]; xe=[0.002,0.002,0.002]
    zm=[0.153,0.072,0.205]; ze=[0.005,0.004,0.003]
    labels=["raw","GP\nregression","conditional"]
    x=np.arange(3); w=0.38
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.bar(x-w/2,xm,w,yerr=xe,capsize=3,color="#5dade2",edgecolor="k",lw=.5,
           label=r"$\langle X_0\rangle$ (single-qubit)")
    ax.bar(x+w/2,zm,w,yerr=ze,capsize=3,color="#e59866",edgecolor="k",lw=.5,
           label=r"$\langle Z_0 Z_1\rangle$ (correlator)")
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
    ax.set_ylabel("reconstruction RMSE")
    ax.legend(); ax.grid(axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT+"route_comparison.png",dpi=150); fig.savefig(OUT+"route_comparison.pdf")
    print("route fig done")


def fig_recon():
    tg = np.linspace(0, 2 * np.pi, 300)
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = qt.tensor(plus, plus)
    H = (qt.tensor(qt.sigmaz(), qt.sigmaz())
         + 0.5 * (qt.tensor(qt.sigmax(), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.sigmax())))
    X0 = qt.tensor(qt.sigmax(), qt.qeye(2))
    truth = np.real(qt.sesolve(H, psi0, tg, e_ops=[X0]).expect[0])
    # A-standard illustration (100 times x 200 shadows), FINAL estimator: fresh bases per
    # snapshot, matched-count (realised-m) normalisation, empirical per-point noise.
    obs_idx = np.linspace(0, 299, 100).astype(int)
    ot, xt = tg[obs_idx], truth[obs_idx]
    rng = np.random.default_rng(2024)
    N = 200
    est = np.empty(len(ot))
    nv_pt = np.empty(len(ot))
    for i, x in enumerate(xt):
        b = rng.integers(0, 3, N)
        o = np.where(rng.random(N) < (1 + x) / 2, 1.0, -1.0)
        matched = o[b == 0]
        m = len(matched)
        est[i] = matched.mean()
        p_s = (np.sum(matched > 0) + 1.0) / (m + 2.0)
        nv_pt[i] = 4.0 * p_s * (1.0 - p_s) / m        # Laplace-smoothed SE^2 of the matched mean
    ell = 0.5
    sf2 = max(np.var(est) - nv_pt.mean(), 0.05)       # signal (output) scale
    d2o = (ot[:, None] - ot[None, :]) ** 2
    K = sf2 * np.exp(-0.5 * d2o / ell ** 2) + np.diag(nv_pt)
    Ks = sf2 * np.exp(-0.5 * (tg[:, None] - ot[None, :]) ** 2 / ell ** 2)
    mean = Ks @ np.linalg.solve(K, est)
    var = sf2 - np.sum(Ks * np.linalg.solve(K, Ks.T).T, axis=1)
    band = 2 * np.sqrt(np.clip(var, 0, None) + np.interp(tg, ot, nv_pt))
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    ax.plot(tg, truth,"k-", lw=1.6, label="truth")
    ax.plot(ot, est,"o", ms=3.5, color="#e59866", alpha=0.8, label="shadow estimates")
    ax.plot(tg, mean,"-", color="#2471a3", lw=1.8, label="GP reconstruction")
    ax.fill_between(tg, mean - band, mean + band, color="#2471a3", alpha=0.18, label="95% band")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\langle X_0\rangle$")
    ax.set_title("Reconstructing a single-observable trajectory with uncertainty (2-qubit TFIM)", fontsize=10)
    ax.legend(fontsize=9.5, ncol=2, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_OUT +"reconstruction_band.png", dpi=150); fig.savefig(_OUT +"reconstruction_band.pdf")
    fig.savefig(_OUT +"/reconstruction_band.png", dpi=150); fig.savefig(_OUT +"/reconstruction_band.pdf")
    print("recon fig done")


fig_route()
fig_recon()
print("saved route_comparison.png + reconstruction_band.png")
