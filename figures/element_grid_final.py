"""Element-wise rho(t) illustration and per-element lengthscale map, final estimator.

One dataset (2-qubit TFIM, 500 observed times x 500 shadows, seed 10): matched-Pauli assembled
inputs (studies/rho_final_program.py), per-element Gaussian processes with the noise fixed to the
measured per-element variance. Left figure: four representative elements (a population, a slow and a
fast coherence, an imaginary part) with truth, raw estimates and reconstruction, large fonts.
Right figure: the fitted lengthscale of every element as a 4x4 map.
Outputs: figures/out/element_grid.{pdf,png}, figures/out/lengthscale_map.{pdf,png}.
"""
import os as _os, sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT,"figures","out"); _os.makedirs(_OUT, exist_ok=True)
for p in (_ROOT, _os.path.join(_ROOT,"studies"), _os.path.join(_ROOT,"per_element_rho"),
          _os.path.join(_ROOT,"rho_reconstruction")):
    sys.path.append(p)
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(_os.environ.get("TMPDIR","/tmp"),"mplcfg"))
import numpy as np, qutip as qt, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from Synthetic_Error_Uncertainty_Check import make_cell_seed, build_hamiltonian, build_initial_state
from rho_final_program import rho_inputs, truth_rhos
from mll_shot_scaling import channels

TRUE_T, N_TIMES, TARGET_T, N, SEED = 500, 500, 300, 500, 10
tlist = np.linspace(0, 2 * np.pi, TRUE_T); target = np.linspace(0, 2 * np.pi, TARGET_T)
obs_idx = np.linspace(0, TRUE_T - 1, N_TIMES, dtype=int); to = np.linspace(0, 1, N_TIMES)
tn = (target - 0) / (2 * np.pi)
states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tlist, []).states
truth = truth_rhos(states, tlist, target, 4)
mean, vr, vi = rho_inputs(states, obs_idx, tlist, make_cell_seed(SEED,"ZZ", N, N_TIMES, 0), N, 2)
ch = channels(mean, vr, vi)

pred = np.zeros((TARGET_T, 4, 4), dtype=complex); ls = {}
for (i, j), part, ys, mu, sd, alpha in ch:
    k = C(1.0,"fixed") * RBF(0.2, (0.02, 1.0))
    gp = GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=False, n_restarts_optimizer=1
                                  ).fit(to.reshape(-1, 1), ys)
    m = gp.predict(tn.reshape(-1, 1)) * sd + mu
    pred[:, i, j] += m if part =="real" else 1j * m
    ls[(i, j, part)] = float(gp.kernel_.k2.length_scale)

plt.rcParams.update({"font.size": 13,"axes.labelsize": 13,"legend.fontsize": 12})
sel = [((0, 0),"real", r"$\rho_{00}$ (population)"), ((0, 1),"real", r"$\mathrm{Re}\,\rho_{01}$"),
       ((0, 1),"imag", r"$\mathrm{Im}\,\rho_{01}$"), ((0, 3),"real", r"$\mathrm{Re}\,\rho_{03}$")]
fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.0), sharex=True)
for ax, ((i, j), part, lab) in zip(axes.ravel(), sel):
    f = (lambda z: z.real) if part =="real" else (lambda z: z.imag)
    ax.plot(tlist[obs_idx], f(mean[:, i, j]),".", ms=3, color="#e59866", alpha=.6, label="raw shadow estimate")
    ax.plot(target, f(truth[:, i, j]),"k-", lw=2, label="exact")
    ax.plot(target, f(pred[:, i, j]),"-", color="#2471a3", lw=2, label="GP reconstruction")
    ax.set_title(lab, fontsize=14); ax.grid(alpha=.25)
for ax in axes[1]: ax.set_xlabel("time $t$")
axes[0, 0].legend(loc="upper right", framealpha=.9)
fig.tight_layout()
for ext in ("pdf","png"): fig.savefig(_os.path.join(_OUT, f"element_grid.{ext}"), dpi=150)

M = np.zeros((4, 4))
for (i, j, part), v in ls.items():
    if part =="real" and i <= j: M[i, j] = v
    if part =="imag" and i < j: M[j, i] = v
    if part =="real" and i == j: M[i, i] = v
fig2, ax2 = plt.subplots(figsize=(5.6, 4.6))
im = ax2.imshow(M, cmap="viridis"); fig2.colorbar(im, ax=ax2, label="fitted lengthscale (fraction of window)")
for i in range(4):
    for j in range(4):
        ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < M.max() * .6 else"k", fontsize=12)
ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
ax2.set_title("per-element fitted lengthscale\n(upper: real parts, lower: imaginary parts)", fontsize=12)
fig2.tight_layout()
for ext in ("pdf","png"): fig2.savefig(_os.path.join(_OUT, f"lengthscale_map.{ext}"), dpi=150)
print("element_grid + lengthscale_map done; frob of reconstruction:",
      round(float(np.linalg.norm(pred - truth, axis=(1, 2)).mean()), 4))
