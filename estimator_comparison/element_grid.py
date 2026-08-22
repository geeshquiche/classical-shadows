#!/usr/bin/env python3
"""Matrix-element evolution grid + fitted-lengthscale map (report figures).

Panel grid: the 4x4 elements of the two-qubit TFIM rho(t). Diagonal and lower-triangle panels show the
real part, upper-triangle panels the imaginary part of the corresponding element. Each panel: exact
trajectory, raw shadow estimates, and the per-element empirical-noise GP reconstruction (noise fixed to the
measured per-element snapshot variance, lengthscale fitted per element on a grid).

Second figure: heat map of the fitted lengthscale per element (real channel), showing slow populations
selecting long lengthscales and fast coherences short ones.

One seed; configuration stated in the captions. Estimation uses only shot data (bases + outcomes); the
exact trajectory appears only as the plotted truth.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qutip as qt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.append(_PARENT)
sys.path.append(os.path.join(_PARENT, "rho_reconstruction"))
import conditional_rho as cr

import os as _os
SP = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "out")  # repo-relative output dir
_os.makedirs(SP, exist_ok=True)
SEED, N_TRUE, N_OBS, SHADOW = 10, 300, 80, 300
LS_GRID = np.geomspace(0.05, 1.2, 12)

tl = np.linspace(0, 2 * np.pi, N_TRUE)
H = cr.build_ising_hamiltonian(2)
states = qt.mesolve(H, cr.build_plus_initial_state(2), tl, []).states
obs_idx = np.linspace(0, N_TRUE - 1, N_OBS, dtype=int)
obs_t = tl[obs_idx]
truth = np.array([s.full() if s.isoper else (s * s.dag()).full() for s in states])

# ---- shadow estimates per element (empirical variance measured from the snapshots) ----
rng_seed = SEED
np.random.seed(rng_seed)
import random as _random
_random.seed(rng_seed)
mean = np.zeros((N_OBS, 4, 4), complex)
var_r = np.zeros((N_OBS, 4, 4))
var_i = np.zeros((N_OBS, 4, 4))
for ti, idx in enumerate(obs_idx):
    st = states[int(idx)]
    rho_t = st if st.isoper else st * st.dag()
    acc = np.zeros((4, 4), complex)
    a2r = np.zeros((4, 4)); a2i = np.zeros((4, 4))
    for _ in range(SHADOW):
        setting = cr.create_random_pauli_obs(2)
        meas = cr.pauli_measurement(rho_t, setting, 2)
        m = np.kron(cr.local_shadow_matrix(meas[0][0], meas[0][1]),
                    cr.local_shadow_matrix(meas[1][0], meas[1][1]))
        acc += m; a2r += m.real ** 2; a2i += m.imag ** 2
    mean[ti] = acc / SHADOW
    var_r[ti] = np.maximum(a2r / SHADOW - mean[ti].real ** 2, 1e-9) / SHADOW
    var_i[ti] = np.maximum(a2i / SHADOW - mean[ti].imag ** 2, 1e-9) / SHADOW


def gp_fit(y, noise_var):
    """Fixed-noise GP: pick lengthscale on LS_GRID by log marginal likelihood; return recon + best ls."""
    t = (obs_t - obs_t[0]) / (obs_t[-1] - obs_t[0])
    d2 = (t[:, None] - t[None, :]) ** 2
    best, best_ll = None, -np.inf
    sf2 = max(np.var(y) - noise_var.mean(), 1e-4)
    for ls in LS_GRID:
        K = sf2 * np.exp(-0.5 * d2 / ls ** 2) + np.diag(noise_var)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        ll = -0.5 * y @ a - np.log(np.diag(L)).sum()
        if ll > best_ll:
            best_ll, best = ll, ls
    K = sf2 * np.exp(-0.5 * d2 / best ** 2) + np.diag(noise_var)
    recon = (sf2 * np.exp(-0.5 * d2 / best ** 2)) @ np.linalg.solve(K, y)
    return recon, best


recon = np.zeros((N_OBS, 4, 4), complex)
ls_map = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        rr, lr = gp_fit(mean[:, i, j].real, var_r[:, i, j])
        ri, li = gp_fit(mean[:, i, j].imag, var_i[:, i, j]) if i != j else (np.zeros(N_OBS), lr)
        recon[:, i, j] = rr + 1j * ri
        ls_map[i, j] = lr

# ---- figure 1: 4x4 element grid ----
fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.5), sharex=True)
for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        if i <= j:
            tr = truth[:, i, j].real; me = mean[:, i, j].real; rc = recon[:, i, j].real
            tag = rf"$\mathrm{{Re}}\,\rho_{{{i}{j}}}$"
        else:
            tr = truth[:, i, j].imag; me = mean[:, i, j].imag; rc = recon[:, i, j].imag
            tag = rf"$\mathrm{{Im}}\,\rho_{{{i}{j}}}$"
        ax.plot(tl, tr, "k-", lw=1.1)
        ax.plot(obs_t, me, ".", ms=2.2, color="#e59866", alpha=0.75)
        ax.plot(obs_t, rc, "-", lw=1.4, color="#2471a3")
        ax.set_title(tag, fontsize=14, pad=3)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(alpha=0.2)
for ax in axes[3]:
    ax.set_xlabel("time", fontsize=12)
fig.legend(handles=[plt.Line2D([], [], color="k", lw=1.1, label="truth"),
                    plt.Line2D([], [], marker=".", ls="", color="#e59866", label="raw shadow estimate"),
                    plt.Line2D([], [], color="#2471a3", lw=1.4, label="per-element GP (empirical noise)")],
           loc="lower center", ncol=3, fontsize=13, frameon=False)
fig.tight_layout(rect=[0, 0.045, 1, 1])
for d in (_HERE, SP):
    fig.savefig(os.path.join(d, "element_grid.png"), dpi=130); fig.savefig(os.path.join(d, "element_grid.pdf"))

# ---- figure 2: fitted lengthscale map ----
fig2, ax = plt.subplots(figsize=(4.6, 3.9))
im = ax.imshow(ls_map, cmap="viridis")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{ls_map[i,j]:.2f}", ha="center", va="center",
                color="w" if ls_map[i, j] < ls_map.max() * 0.7 else "k", fontsize=8)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xlabel("column $b$"); ax.set_ylabel("row $a$")
fig2.colorbar(im, ax=ax, label="fitted lengthscale (fraction of window)")
fig2.tight_layout()
for d in (_HERE, SP):
    fig2.savefig(os.path.join(d, "lengthscale_map.png"), dpi=140); fig2.savefig(os.path.join(d, "lengthscale_map.pdf"))
print("saved element_grid.png + lengthscale_map.png")
