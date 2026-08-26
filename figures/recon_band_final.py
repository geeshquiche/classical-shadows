"""Figure 1: single-observable reconstruction with its predictive band, from the final pipeline.

One dataset at the report's standard configuration (100 observation times over t in [0, 2pi],
200 shadows per time, seed 10): fresh Pauli bases per time, matched-count per-time estimates with
Laplace-smoothed standard errors, Gaussian-process regression (Matern-3/2) whose observation noise
is supplied per point from those standard errors. The band is 1.96*sqrt(posterior variance +
shot-noise variance), the same construction the coverage study evaluates.
Output: figures/out/reconstruction_band.{pdf,png}
"""
import os as _os, sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.path.join(_ROOT,"figures","out"); _os.makedirs(_OUT, exist_ok=True)
for p in (_ROOT, _os.path.join(_ROOT,"studies"), _os.path.join(_ROOT,"rho_reconstruction"),
          _os.path.join(_ROOT,"calibration_and_adaptive")):
    sys.path.append(p)
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(_os.environ.get("TMPDIR","/tmp"),"mplcfg"))
import numpy as np, qutip as qt, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os2
_sys.path.append(_os2.path.dirname(_os2.path.abspath(__file__)))
import report_style  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import conditional_rho as cr
from test_pipeline_coverage import exact_curve, OPS, Z95
from coverage_basis_ablation import generate_rerandomised_measurement_df
from final_config_coverage import per_time_series, gp_fit, SUPPORTS
from Synthetic_Error_Uncertainty_Check import make_cell_seed

TRUE_T, PRED_T, N_TIMES, SHADOWS, SEED, OBS = 400, 300, 100, 200, 10,"XI"
tlist = np.linspace(0, 2 * np.pi, TRUE_T)
target = np.linspace(0, 2 * np.pi, PRED_T)
states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tlist, []).states
observed = np.linspace(0, TRUE_T - 1, N_TIMES, dtype=int)
truth = exact_curve(states, tlist, OPS[OBS], target)

cell = make_cell_seed(SEED, "ZZ", SHADOWS, N_TIMES, 7)
np.random.seed(cell)
mdf = generate_rerandomised_measurement_df(states, tlist, observed, 2, SHADOWS,
                                           shots_per_setting=1, seed=cell)

fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharex=True)
for ax, obs, lab in zip(axes, ["XI", "ZZ"], [r"$\langle X_0\rangle$", r"$\langle Z_0Z_1\rangle$"]):
    truth = exact_curve(states, tlist, OPS[obs], target)
    obs_times, series = per_time_series(mdf, SUPPORTS[obs])
    y, se = series["matched"]
    mean, std, se_tgt = gp_fit(obs_times, y, se, target, "empirical")
    band = Z95 * np.sqrt(std ** 2 + se_tgt ** 2)
    keep = ~np.isnan(y)
    ax.plot(obs_times[keep], y[keep], "o", ms=2.6, color="#e59866", alpha=.7, label="shadow estimates")
    ax.plot(target, truth, "k-", lw=1.6, label="exact")
    ax.plot(target, mean, "-", color="#2471a3", lw=1.8, label="reconstruction")
    ax.fill_between(target, mean - band, mean + band, color="#2471a3", alpha=.18, label="95% band")
    ax.set_xlabel(r"time $t$ (units of $1/J$)"); ax.set_ylabel(lab); ax.grid(alpha=.25)
    rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
    print(f"  {obs}: RMSE {rmse:.4f}, coverage {float(np.mean(np.abs(mean-truth) <= band)):.3f}")
# a four-entry legend does not fit inside either panel without covering data or spilling
# past the axes, so it goes underneath as a figure legend
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8.6, ncol=4, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.7, handletextpad=0.5)
fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.30, wspace=0.38)
report_style.save_exact(fig, [_os.path.join(_OUT, f"reconstruction_band.{ext}") for ext in ("pdf", "png")])
print("reconstruction_band done (both observables, pipeline)")
