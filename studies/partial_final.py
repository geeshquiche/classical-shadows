#!/usr/bin/env python3
"""Partial vs full element-wise reconstruction of a reduced block, final estimator.

Mirrors the report's Section 5.8 design: the two-qubit reduced block of the four-qubit TFIM
chain, 40 observed times x 200 shadows; a selected subset of the block's ten independent
entries (4 populations + 6 coherences) is GP-reconstructed (empirical-noise, per-element)
while the remainder keep their raw estimates; three selection orderings are compared
(decreasing mean magnitude, diagonal-first, random), Frobenius error against the exact reduced
block, evaluated at the observed times. Inputs use the matched-Pauli assembled estimator with
the fast pattern-count sampler. Seeds raised 8 -> 15.

Run:  python partial_final.py     (QUICK=1 smoke)
"""
import os
import sys
import csv
import itertools
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_FF = os.path.join(_PARENT, "per_element_rho")
for _p in (_PARENT, _FF, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import (make_cell_seed, build_hamiltonian,
                                               build_initial_state)
from rho_final_program import (sample_counts, matched_pauli_estimates, outcome_signs,
                               pauli_matrix)
from mll_shot_scaling import channels, fit_empnoise

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
NQ_TOTAL, BLOCK = 4, (0, 1)
if QUICK:
    TRUE_T, N_TIMES, SHADOWS, SEEDS = 120, 20, 60, [10, 11]
else:
    TRUE_T, N_TIMES, SHADOWS, SEEDS = 400, 40, 200, list(range(10, 25))
ORDERINGS = ["magnitude", "diagonal-first", "random"]
# ten independent entries of a 4x4 hermitian block: 4 diagonal + 6 upper-triangle
ENTRIES = [(i, i) for i in range(4)] + [(i, j) for i in range(4) for j in range(i + 1, 4)]


def block_inputs(states, obs_idx, tlist, seed, n_shots):
    """Matched-Pauli assembled 2q reduced block of the 4q chain, per observed time."""
    rng = np.random.default_rng(seed)
    patterns = list(itertools.product("XYZ", repeat=NQ_TOTAL))
    block_strings = [p + ("I",) * (NQ_TOTAL - 2) for p in itertools.product("IXYZ", repeat=2)
                     if any(c != "I" for c in p)]
    pmats2 = np.array([pauli_matrix(p[:2]) for p in block_strings])   # 2q matrices
    signs = outcome_signs(NQ_TOTAL)
    T = len(obs_idx)
    mean = np.zeros((T, 4, 4), dtype=np.complex128)
    var_r = np.zeros((T, 4, 4))
    var_i = np.zeros((T, 4, 4))
    for ti, idx in enumerate(obs_idx):
        psi = np.asarray(states[int(idx)].full()).ravel()
        counts = sample_counts(rng, psi, patterns, n_shots, NQ_TOTAL)
        est, se = matched_pauli_estimates(counts, patterns, NQ_TOTAL, block_strings, signs)
        rho = np.eye(4, dtype=complex)
        for e, pm in zip(est, pmats2):
            rho = rho + e * pm
        mean[ti] = rho / 4.0
        v = (se ** 2)[:, None, None]
        var_r[ti] = np.sum(v * (pmats2.real ** 2), axis=0) / 16.0
        var_i[ti] = np.sum(v * (pmats2.imag ** 2), axis=0) / 16.0
    return mean, var_r, var_i


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    obs_idx = np.linspace(0, TRUE_T - 1, N_TIMES, dtype=int)
    to = np.linspace(0, 1, N_TIMES)
    states = qt.mesolve(build_hamiltonian("tfim", NQ_TOTAL), build_initial_state(NQ_TOTAL),
                        tlist, []).states
    truth = np.array([np.asarray(states[int(i)].ptrace(list(BLOCK)).full()) for i in obs_idx])

    print(f"QUICK={QUICK}  4q chain, 2q block, {N_TIMES} times x {SHADOWS} shadows, "
          f"{len(SEEDS)} seeds\n", flush=True)

    rows, per_seed = [], {}
    for seed in SEEDS:
        cell = make_cell_seed(seed, "ZZII", SHADOWS, N_TIMES, 4)
        mean, vr, vi = block_inputs(states, obs_idx, tlist, cell, SHADOWS)
        ch = channels(mean, vr, vi)
        pred = fit_empnoise(to, to, ch, 4)          # per-element empnoise GP at observed times
        raw_frob = float(np.linalg.norm(mean - truth, axis=(1, 2)).mean())
        rng = np.random.default_rng(cell + 1)
        mag = {e: float(np.abs(mean[:, e[0], e[1]]).mean()) for e in ENTRIES}
        orders = {
            "magnitude": sorted(ENTRIES, key=lambda e: -mag[e]),
            "diagonal-first": ([e for e in ENTRIES if e[0] == e[1]]
                               + sorted([e for e in ENTRIES if e[0] != e[1]],
                                        key=lambda e: -mag[e])),
            "random": list(rng.permutation(len(ENTRIES))),
        }
        orders["random"] = [ENTRIES[i] for i in orders["random"]]
        for ordering in ORDERINGS:
            seq = orders[ordering]
            for k in range(len(ENTRIES) + 1):
                hybrid = mean.copy()
                for (i, j) in seq[:k]:
                    hybrid[:, i, j] = pred[:, i, j]
                    if i != j:
                        hybrid[:, j, i] = np.conj(pred[:, i, j])
                f = float(np.linalg.norm(hybrid - truth, axis=(1, 2)).mean())
                per_seed.setdefault((ordering, k), {})[seed] = f
                rows.append({"ordering": ordering, "n_entries": k,
                             "fraction": round(k / len(ENTRIES), 2), "seed": seed,
                             "frob": round(f, 5), "raw_frob": round(raw_frob, 5)})
        print(f"  seed {seed} done ({_time.perf_counter()-t0:.0f}s)  raw={raw_frob:.4f}  "
              f"full={per_seed[('magnitude', 10)][seed]:.4f}", flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (old-pipeline refs: raw ~0.35, full ~0.22, ~linear gain) ====",
          flush=True)
    sum_rows = []
    for ordering in ORDERINGS:
        for k in range(len(ENTRIES) + 1):
            arr = np.array([per_seed[(ordering, k)][s] for s in SEEDS])
            sum_rows.append({"ordering": ordering, "n_entries": k,
                             "fraction": round(k / len(ENTRIES), 2),
                             "frob_mean": round(float(arr.mean()), 5),
                             "frob_se": round(float(arr.std(ddof=1) / np.sqrt(ns)), 5)})
    for ordering in ORDERINGS:
        line = "  ".join(f"{k}:{np.mean([per_seed[(ordering, k)][s] for s in SEEDS]):.3f}"
                         for k in (0, 3, 5, 10))
        print(f"  {ordering:15s} {line}", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# partial vs full block reconstruction, final matched-Pauli estimator; 4q TFIM "
              f"chain 2q block, {N_TIMES}x{SHADOWS}, {ns} seeds, empnoise per-element GP; "
              f"wall={wall:.0f}s")
    for fname, data in [("partial_final.csv", rows), ("partial_final_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved partial_final{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
