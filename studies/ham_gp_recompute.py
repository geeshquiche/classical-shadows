#!/usr/bin/env python3
"""Recompute the GP arm of the 8-Hamiltonian sweep under the final (matched-count, empirical
noise) estimator, PAIRED against the existing conditional-route results.

The original sweep (estimator_comparison/hamiltonian_route_comparison.py) generated data per
(hamiltonian, seed) cell with make_cell_seed(seed, OBS+name, SHADOW, NUM_OBS, 0) and stored
both arms in hamiltonian_route_comparison_merged.csv. Data generation is deterministic given
the cell seed, so this script regenerates IDENTICAL measurement data per cell, computes the
final-method GP route on it, and pairs against the stored conditional numbers -- the expensive
conditional fits are reused, and the pairing is exact.

<X0> across the June Hamiltonian library; 100 times x 200 shadows; seeds as stored per cell.

Run:  python ham_gp_recompute.py     (QUICK=1 smokes one Hamiltonian, one seed)
"""
import os
import sys
import csv
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_EC = os.path.join(_PARENT, "estimator_comparison")
for _p in (_PARENT, _EC):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import (
    make_cell_seed, pauli_string_operator, observable_support, generate_measurement_df)
from final_config_coverage import gp_fit
from routes3_final import matched_series

QUICK = os.environ.get("QUICK", "0") == "1"
OBS, QUBIT = "XI", 2
NUM_TRUE, NUM_PRED, NUM_OBS, SHADOW = 400, 100, 100, 200
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi

ZZ = qt.tensor(qt.sigmaz(), qt.sigmaz())
XI_ = qt.tensor(qt.sigmax(), qt.qeye(2))
IX_ = qt.tensor(qt.qeye(2), qt.sigmax())
ZI_ = qt.tensor(qt.sigmaz(), qt.qeye(2))
IZ_ = qt.tensor(qt.qeye(2), qt.sigmaz())
XX = qt.tensor(qt.sigmax(), qt.sigmax())
YY = qt.tensor(qt.sigmay(), qt.sigmay())
HAMS = {
    "ZZ_slow": (0.5 * ZZ, []),
    "ZZ_fast": (2.5 * ZZ, []),
    "ZZ_plus_XI": (ZZ + 0.5 * XI_, []),
    "weak": (0.05 * ZZ, []),
    "entangling": (XX + YY, []),
    "TFIM": (ZZ + 0.5 * XI_ + 0.5 * IX_, []),
    "beating": (0.4 * ZI_ + 0.6 * IZ_ + 0.3 * ZZ, []),
    "dephasing": (0.5 * ZZ, [np.sqrt(0.3) * ZI_]),
}


def main():
    t0 = _time.perf_counter()
    merged = os.path.join(_EC, "hamiltonian_route_comparison_merged.csv")
    cond = {}
    with open(merged) as f:
        for r in csv.DictReader(f):
            if r["route"] == "conditional":
                cond.setdefault(r["hamiltonian"], {})[int(r["seed"])] = float(r["rmse"])

    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = qt.tensor(plus, plus)
    tlist = np.linspace(TIME_MIN, TIME_MAX, NUM_TRUE)
    pred_times = np.linspace(TIME_MIN, TIME_MAX, NUM_PRED)
    obs_idx = np.linspace(0, NUM_TRUE - 1, NUM_OBS, dtype=int)
    operator = pauli_string_operator(OBS, QUBIT)
    support = observable_support(OBS)

    ham_names = list(HAMS)[:1] if QUICK else list(HAMS)
    rows, sum_rows = [], []
    print(f"QUICK={QUICK}  <{OBS}> final-method GP arm, paired vs stored conditional\n",
          flush=True)
    for name in ham_names:
        H, c_ops = HAMS[name]
        states = qt.mesolve(H, psi0, tlist, c_ops).states
        true_full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth = np.interp(pred_times, tlist, true_full)
        truth_rms = float(np.sqrt(np.mean(truth ** 2))) or 1.0
        seeds = sorted(cond[name])[:1] if QUICK else sorted(cond[name])
        g, c = [], []
        for seed in seeds:
            data_seed = make_cell_seed(seed, OBS + name, SHADOW, NUM_OBS, 0)
            mdf = generate_measurement_df(states, tlist, obs_idx, QUBIT, SHADOW,
                                          seed=data_seed)
            obs_times, y, se = matched_series(mdf, support)
            mean, _std, _se_tgt = gp_fit(obs_times, y, se, pred_times, "empirical")
            rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
            g.append(rmse)
            c.append(cond[name][seed])
            rows.append({"hamiltonian": name, "seed": seed, "gp_rmse": round(rmse, 5),
                         "conditional_rmse": round(cond[name][seed], 5),
                         "gp_rrmse": round(rmse / truth_rms, 5)})
        g, c = np.array(g), np.array(c)
        d = g - c
        ns = len(g)
        sum_rows.append({
            "hamiltonian": name, "n_seeds": ns,
            "gp_mean": round(float(g.mean()), 5),
            "gp_se": round(float(g.std(ddof=1) / np.sqrt(ns)), 5),
            "cond_mean": round(float(c.mean()), 5),
            "cond_se": round(float(c.std(ddof=1) / np.sqrt(ns)), 5),
            "paired_diff": round(float(d.mean()), 5),
            "paired_se": round(float(d.std(ddof=1) / np.sqrt(ns)), 5),
            "gp_wins": int(np.sum(d < 0))})
        print(f"  {name:11s} gp={g.mean():.4f}+/-{g.std(ddof=1)/np.sqrt(ns):.4f}  "
              f"cond={c.mean():.4f}  diff={d.mean():+.4f}+/-{d.std(ddof=1)/np.sqrt(ns):.4f}  "
              f"wins {int(np.sum(d<0))}/{ns}  ({_time.perf_counter()-t0:.0f}s)", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# 8-Hamiltonian sweep, GP arm recomputed under final method (matched-count, "
              f"empirical noise, Matern-3/2), paired vs stored conditional arm (identical data "
              f"per cell via make_cell_seed); <XI>, {NUM_OBS}x{SHADOW}; wall={wall:.0f}s")
    for fname, data in [("ham_gp_recompute.csv", rows),
                        ("ham_gp_recompute_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved ham_gp_recompute{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
