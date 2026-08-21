#!/usr/bin/env python3
"""Per-element rho(t) program under the consistent matched-count estimator.

Estimator: every nontrivial Pauli expectation <P>(t) is estimated by the matched-count (Hajek)
form of Eq. (matched-estimator) -- mean of outcome products over snapshots whose bases align
with P's support, Laplace-smoothed SE -- and rho_hat(t) is assembled as
    rho_hat(t) = 2^-n [ I + sum_P  p_hat_P(t) P ].
Per-element noise variances are propagated from the Pauli SEs,
Var(rho_ab) ~ 4^-n sum_P P_ab^2 se_P^2 (real/imag parts separately; independence approx).
Bases are drawn fresh per snapshot (iid protocol). Sampling uses exact per-time multinomials
over the 3^n basis patterns and 2^n Born outcomes -- statistically identical to per-snapshot
simulation, dramatically faster. The truth enters ONLY measurement sampling + final metrics.

Arms (2x2 de-confound): {noise: fitted, fixed-empirical} x {lengthscale: shared, per-element},
RBF kernel, mirroring the house mll pipeline (fits imported from mll_shot_scaling).

Errors: mean Frobenius distance to exact rho(t), absolute AND relative to the truth's mean
Frobenius norm (consistent-fractions convention, 2026-08-21).

Modes (env MODE):
  core   (default): 2q TFIM full rho, 500 times x 500 shadows, 20 seeds, all 4 arms
  nqubit : 2q and 3q full rho, 500x500, 12 seeds, arms shared/per-elem/empnoise
  shots  : 2q, 500 times, N in {125,250,500,1000,2000}, 10 seeds, shared/per-elem/empnoise

Run:  MODE=core python rho_final_program.py     (QUICK=1 for smoke)
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_FF = os.path.join(_PARENT, "for_fred_latest")
for _p in (_PARENT, _FF):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import make_cell_seed, build_hamiltonian, build_initial_state
from mll_shot_scaling import channels, fit_shared, fit_per_element, fit_empnoise, frob, LS_GRID

QUICK = os.environ.get("QUICK", "0") == "1"
MODE = os.environ.get("MODE", "core")
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi

if MODE == "core":
    QUBITS_LIST, N_GRID = [2], [500]
    SEEDS = [10, 11] if QUICK else list(range(10, 30))
    ARMS = ["shared", "per-elem", "shared-empnoise", "empnoise"]
    REPEAT_ID = 0
elif MODE == "nqubit":
    QUBITS_LIST, N_GRID = [2, 3], [500]
    SEEDS = [10, 11] if QUICK else list(range(10, 22))
    ARMS = ["shared", "per-elem", "empnoise"]
    REPEAT_ID = 1
elif MODE == "shots":
    QUBITS_LIST, N_GRID = [2], [125, 250, 500, 1000, 2000]
    SEEDS = [10, 11] if QUICK else list(range(10, 20))
    ARMS = ["shared", "per-elem", "empnoise"]
    REPEAT_ID = 2
elif MODE == "variants":
    QUBITS_LIST, N_GRID = [2], [500]
    SEEDS = [10, 11] if QUICK else list(range(10, 30))
    ARMS = ["shared", "per-elem", "grouped", "bounded", "restart", "grid", "empnoise"]
    REPEAT_ID = 0   # same cells as MODE=core -> identical data, paired against core arms
else:
    raise SystemExit(f"unknown MODE={MODE}")

if QUICK:
    TRUE_T, N_TIMES, TARGET_T = 200, 60, 80
    N_GRID = [N_GRID[0]] if MODE != "shots" else [125, 500]
else:
    TRUE_T, N_TIMES, TARGET_T = 500, 500, 200

_s = {"I": np.eye(2, dtype=complex),
      "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
      "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
# eigenbasis unitaries: columns are +1/-1 eigenvectors of each Pauli
_U = {"X": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
      "Y": np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2),
      "Z": np.eye(2, dtype=complex)}


def pauli_matrix(letters):
    m = _s[letters[0]]
    for c in letters[1:]:
        m = np.kron(m, _s[c])
    return m


def born_probs(psi, pattern):
    """Probabilities of the 2^n outcome strings when measuring pattern (tuple of X/Y/Z)."""
    u = _U[pattern[0]]
    for c in pattern[1:]:
        u = np.kron(u, _U[c])
    amps = u.conj().T @ psi
    return np.clip(np.abs(amps) ** 2, 0, None)


def sample_counts(rng, psi, patterns, n_shots, n_qubits):
    """counts[pattern_index, outcome_index] for one time; bases fresh per snapshot."""
    pat_counts = rng.multinomial(n_shots, np.full(len(patterns), 1.0 / len(patterns)))
    out = np.zeros((len(patterns), 2 ** n_qubits), dtype=np.int64)
    for pi, (pattern, m) in enumerate(zip(patterns, pat_counts)):
        if m == 0:
            continue
        p = born_probs(psi, pattern)
        p = p / p.sum()
        out[pi] = rng.multinomial(m, p)
    return out


def outcome_signs(n_qubits):
    """signs[outcome_index, qubit] = +/-1.

    Kron ordering is big-endian: qubit 0 is the FIRST tensor factor, i.e. the HIGHEST bit of
    the outcome index. (A little-endian version silently reads the wrong qubit's outcome and
    contaminates every single-qubit Pauli estimate with 1/3 of the wrong-basis mean.)"""
    idx = np.arange(2 ** n_qubits)
    return np.array([1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)]).T


def matched_pauli_estimates(counts, patterns, n_qubits, pauli_strings, signs):
    """Matched-count estimate + smoothed SE for every nontrivial Pauli string, one time."""
    est = np.zeros(len(pauli_strings))
    se = np.zeros(len(pauli_strings))
    for si, letters in enumerate(pauli_strings):
        support = [q for q, c in enumerate(letters) if c != "I"]
        match = [pi for pi, pat in enumerate(patterns)
                 if all(pat[q] == letters[q] for q in support)]
        prod = np.prod(signs[:, support], axis=1)          # per outcome string
        m = int(counts[match].sum())
        if m == 0:
            est[si], se[si] = 0.0, 1.0
            continue
        tot = int((counts[match] * (prod > 0)[None, :]).sum())   # count of +1 products
        est[si] = (2.0 * tot - m) / m
        p_s = (tot + 1.0) / (m + 2.0)
        se[si] = np.sqrt(4.0 * p_s * (1.0 - p_s) / m)
    return est, se


def rho_inputs(states, obs_idx, tlist, seed, n_shots, n_qubits):
    """Matched-Pauli assembled rho_hat(t) + per-element variance inputs, all observed times."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    patterns = list(itertools.product("XYZ", repeat=n_qubits))
    pauli_strings = [p for p in itertools.product("IXYZ", repeat=n_qubits)
                     if any(c != "I" for c in p)]
    pmats = np.array([pauli_matrix(p) for p in pauli_strings])
    signs = outcome_signs(n_qubits)
    mean = np.zeros((len(obs_idx), dim, dim), dtype=np.complex128)
    var_r = np.zeros((len(obs_idx), dim, dim))
    var_i = np.zeros((len(obs_idx), dim, dim))
    for ti, idx in enumerate(obs_idx):
        st = states[int(idx)]
        psi = np.asarray(st.full()).ravel()
        counts = sample_counts(rng, psi, patterns, n_shots, n_qubits)
        est, se = matched_pauli_estimates(counts, patterns, n_qubits, pauli_strings, signs)
        rho = np.eye(dim, dtype=complex)
        for e, pm in zip(est, pmats):
            rho = rho + e * pm
        mean[ti] = rho / dim
        v = (se ** 2)[:, None, None]
        var_r[ti] = np.sum(v * (pmats.real ** 2), axis=0) / dim ** 2
        var_i[ti] = np.sum(v * (pmats.imag ** 2), axis=0) / dim ** 2
    return mean, var_r, var_i


def fit_shared_empnoise(to, tn, ch, n):
    """Shared lengthscale chosen by LML with the noise FIXED per point (empirical alpha)."""
    best, bl = -np.inf, LS_GRID[0]
    for ls in LS_GRID:
        k = C(1.0, "fixed") * RBF(ls, "fixed")
        tot = 0.0
        for _, _, ys, _, _, alpha in ch:
            tot += GaussianProcessRegressor(kernel=k, alpha=alpha, optimizer=None,
                                            normalize_y=False
                                            ).fit(to.reshape(-1, 1), ys).log_marginal_likelihood()
        if tot > best:
            best, bl = tot, ls
    pred = np.zeros((len(tn), n, n), dtype=np.complex128)
    k = C(1.0, "fixed") * RBF(bl, "fixed")
    for (i, j), part, ys, mu, sd, alpha in ch:
        m = GaussianProcessRegressor(kernel=k, alpha=alpha, optimizer=None,
                                     normalize_y=False).fit(
            to.reshape(-1, 1), ys).predict(tn.reshape(-1, 1)) * sd + mu
        pred[:, i, j] += m if part == "real" else 1j * m
    return pred


def _variants_fitters():
    """Variant-ladder arms, mirroring for_fred_latest/mll_per_element_variants.py definitions."""
    sys.path.append(_FF)
    from mll_per_element_variants import (fit_per_element as v_pe, fit_per_element_grid,
                                          fit_grouped)
    return {"bounded": lambda to, tn, ch, n: v_pe(to, tn, ch, n, ls_bounds=(0.05, 0.5)),
            "restart": lambda to, tn, ch, n: v_pe(to, tn, ch, n, restarts=8),
            "grid": fit_per_element_grid, "grouped": fit_grouped}


FITTERS = {"shared": fit_shared, "per-elem": fit_per_element,
           "shared-empnoise": fit_shared_empnoise, "empnoise": fit_empnoise}
if MODE == "variants":
    FITTERS.update(_variants_fitters())


def truth_rhos(states, tlist, target, dim):
    full = np.array([np.asarray((s * s.dag()).full()) if not s.isoper else np.asarray(s.full())
                     for s in states])
    out = np.empty((len(target), dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            out[:, i, j] = (np.interp(target, tlist, full[:, i, j].real)
                            + 1j * np.interp(target, tlist, full[:, i, j].imag))
    return out


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target = np.linspace(TIME_MIN, TIME_MAX, TARGET_T)
    tn = (target - TIME_MIN) / (TIME_MAX - TIME_MIN)
    obs_idx = np.linspace(0, TRUE_T - 1, N_TIMES, dtype=int)
    to = np.linspace(0, 1, N_TIMES)

    print(f"MODE={MODE} QUICK={QUICK}  qubits={QUBITS_LIST} N={N_GRID} "
          f"{len(SEEDS)} seeds arms={ARMS}\n", flush=True)

    rows, per_seed = [], {}
    for nq in QUBITS_LIST:
        dim = 2 ** nq
        states = qt.mesolve(build_hamiltonian("tfim", nq), build_initial_state(nq),
                            tlist, []).states
        truth = truth_rhos(states, tlist, target, dim)
        truth_scale = float(np.mean(np.linalg.norm(truth, axis=(1, 2))))
        for N in N_GRID:
            for seed in SEEDS:
                cell = make_cell_seed(seed, "Z" * nq, N, N_TIMES, REPEAT_ID)
                mean, var_r, var_i = rho_inputs(states, obs_idx, tlist, cell, N, nq)
                ch = channels(mean, var_r, var_i)
                for arm in ARMS:
                    pred = FITTERS[arm](to, tn, ch, dim)
                    f = frob(pred, truth)
                    per_seed.setdefault((nq, N, arm), {})[seed] = f
                    rows.append({"qubits": nq, "N": N, "arm": arm, "seed": seed,
                                 "frob": round(f, 5),
                                 "rel_frob": round(f / truth_scale, 5)})
                print(f"  nq={nq} N={N} seed={seed} done ({_time.perf_counter()-t0:.0f}s)  "
                      + "  ".join(f"{a}={per_seed[(nq, N, a)][seed]:.4f}" for a in ARMS),
                      flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (rel_frob = Frobenius / mean ||rho_true||_F; published old-pipeline "
          "core refs: shared .0366, per-elem .0399, empnoise .0331) ====", flush=True)
    sum_rows = []
    for (nq, N, arm), d in sorted(per_seed.items()):
        arr = np.array([d[s] for s in SEEDS])
        se_ = arr.std(ddof=1) / np.sqrt(ns)
        base = np.array([per_seed[(nq, N, "shared")][s] for s in SEEDS])
        diff = arr - base
        d_se = diff.std(ddof=1) / np.sqrt(ns)
        sum_rows.append({"qubits": nq, "N": N, "arm": arm,
                         "frob_mean": round(float(arr.mean()), 5),
                         "frob_se": round(float(se_), 5),
                         "pct_vs_shared": round(float(100 * diff.mean() / base.mean()), 2),
                         "paired_diff_se_pct": round(float(100 * d_se / base.mean()), 2)})
        print(f"  nq={nq} N={N:5d} {arm:15s} frob={arr.mean():.5f}+/-{se_:.5f}  "
              f"vs shared {100*diff.mean()/base.mean():+.1f}% +/- {100*d_se/base.mean():.1f}%",
              flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# per-element rho program, matched-Pauli assembled estimator, fresh bases; "
              f"MODE={MODE}; {N_TIMES} times, N={N_GRID}, qubits={QUBITS_LIST}, {ns} seeds, "
              f"RBF, arms={ARMS}; pattern-count sampler; wall={wall:.0f}s")
    for fname, data in [(f"rho_final_{MODE}.csv", rows),
                        (f"rho_final_{MODE}_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved rho_final_{MODE}{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
