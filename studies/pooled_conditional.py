#!/usr/bin/env python3
"""Does the conditional route lose because of the method, or because of how it is fitted?

The implementation used in the report trains one GP classifier per (qubit, shadow_id).  Under the
fixed-basis protocol each shadow has ONE basis, so each classifier sees exactly one binary outcome per
observed time -- it must infer a smooth p(t) from a sequence of single coin flips, and there are
2 x N_shadows of them.

Shadows that share a basis assignment are exchangeable, so they can be pooled: one classifier per
(qubit, basis pattern), each seeing ~N_shadows/3^n outcomes per time.  Same model class, same
autoregressive structure, same sampling -- only the training data is pooled.

This runs both variants on identical data, against raw and GP baselines.
"""
import os, sys, csv, random, time as _time, warnings, collections
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR","/tmp"),"mplcfg"))
import numpy as np, qutip as qt, torch
warnings.filterwarnings("ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_HERE)
for p in (_P,_HERE):
    if p not in sys.path: sys.path.append(p)
from Synthetic_Error_Uncertainty_Check import (build_hamiltonian, build_initial_state,
    pauli_string_operator, observable_support, generate_measurement_df, make_cell_seed,
    matched_pauli_string_estimate_from_shots)
from Bayesian_part import (build_conditional_training_data, train_gp_classifier,
    predict_conditional_gp_curve)
from final_config_coverage import gp_fit
from routes3_final import matched_series

QUICK = os.environ.get("QUICK","0")=="1"
NT, NS = (30, 60) if QUICK else (100, 200)
SEEDS = [10,11] if QUICK else list(range(10, 10+int(os.environ.get("NSEED","6"))))
TRUE_T, PRED_T = 400, 100
INDUCING, ITERS, RESAMPLES = (20, 40, 4) if QUICK else (40, 80, 20)
OBSERVABLES = os.environ.get("OBS","XI,ZZ").split(",")


def basis_pattern(mdf, n):
    tab = (mdf[["shadow_id","qubit","pauli"]].drop_duplicates(subset=["shadow_id","qubit"])
           .pivot(index="shadow_id", columns="qubit", values="pauli"))
    return {int(s): tuple(tab.loc[s, q] for q in range(n)) for s in tab.index}


def pooled_training_data(mdf, qubit, shadow_ids):
    """Stack the per-shadow training tensors for every shadow sharing a basis pattern."""
    xs, ys = [], []
    for s in shadow_ids:
        x, y = build_conditional_training_data(mdf, qubit, s)
        xs.append(x); ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def sample_estimates(bank_lookup, mdf, n, shadow_ids, target, obs_string, resamples, rng):
    """Sample outcomes per shadow from its classifier and form the matched-count estimate.

    Qubit q is conditioned on the outcomes already drawn for qubits <q AT THE SAME TIME, as the
    chain-rule factorisation requires.  Conditioning on a single time's draw for the whole trajectory
    destroys the correlation the route exists to capture and collapses correlator estimates toward zero.
    """
    tab = (mdf[["shadow_id","qubit","pauli"]].drop_duplicates(subset=["shadow_id","qubit"])
           .pivot(index="shadow_id", columns="qubit", values="pauli"))
    T = len(target)
    cache = {}
    def curve(q, s, pattern):
        key = (q, bank_lookup(q, s), pattern)
        if key not in cache:
            cache[key] = np.asarray(predict_conditional_gp_curve(
                bank_lookup(q, s, entry=True), t_values=target, previous_outcomes=list(pattern)),
                dtype=float)
        return cache[key]

    out = []
    for _r in range(resamples):
        shots_by_t = [[] for _ in range(T)]
        for s in shadow_ids:
            # draws[q][i] = outcome of qubit q at time i, conditioned on qubits <q at time i
            draws = np.zeros((n, T), dtype=int)
            for q in range(n):
                if q == 0:
                    c = curve(0, s, ())
                    draws[0] = np.where(np.array([rng.random() for _ in range(T)]) < c, 1, -1)
                else:
                    # the conditioning pattern varies with time, so evaluate every pattern the
                    # previous qubits actually took and select per time point
                    patterns = {}
                    for i in range(T):
                        pat = tuple(int(draws[j][i]) for j in range(q))
                        if pat not in patterns:
                            patterns[pat] = curve(q, s, pat)
                    u = np.array([rng.random() for _ in range(T)])
                    for i in range(T):
                        pat = tuple(int(draws[j][i]) for j in range(q))
                        draws[q][i] = 1 if u[i] < patterns[pat][i] else -1
            for i in range(T):
                shots_by_t[i].append([[tab.loc[s, q], int(draws[q][i])] for q in range(n)])
        out.append([matched_pauli_string_estimate_from_shots(sh, obs_string) for sh in shots_by_t])
    return np.asarray(out, dtype=float)


def main():
    t0=_time.perf_counter()
    tl = np.linspace(0, 2*np.pi, TRUE_T); target = np.linspace(0, 2*np.pi, PRED_T)
    states = qt.mesolve(build_hamiltonian("tfim",2), build_initial_state(2), tl, []).states
    observed = np.linspace(0, TRUE_T-1, NT, dtype=int)
    rows = []
    print(f"{NT} times x {NS} shadows, {len(SEEDS)} seeds, inducing={INDUCING} iters={ITERS} "
          f"resamples={RESAMPLES}\n", flush=True)
    for obs in OBSERVABLES:
        op = pauli_string_operator(obs, 2); support = observable_support(obs)
        truth = np.interp(target, tl, np.real(np.asarray(qt.expect(op, states), dtype=float)))
        for seed in SEEDS:
            ds = make_cell_seed(seed, obs, NS, NT, 0)
            torch.manual_seed(ds); rng = random.Random(ds)
            mdf = generate_measurement_df(states=states, t_grid=tl, observed_indices=observed,
                                          n=2, shadow_size=NS, seed=ds)
            sids = sorted(mdf["shadow_id"].unique().tolist())
            ot, y, se = matched_series(mdf, support); keep = ~np.isnan(y)
            raw = np.interp(target, ot[keep], y[keep])
            gpm, _, _ = gp_fit(ot[keep], y[keep], se[keep], target, "empirical")
            res = {"raw": raw, "gp": gpm}

            # --- variant A: one classifier per (qubit, shadow_id), as in the report ---
            bankA = {q: {} for q in range(2)}
            for q in range(2):
                for s in sids:
                    x, yy = build_conditional_training_data(mdf, q, s)
                    m, lk = train_gp_classifier(x, yy, num_inducing=min(INDUCING, len(yy)),
                                                training_iter=ITERS, lr=0.05, kernel="matern32")
                    m.eval(); lk.eval(); bankA[q][s] = {"model": m, "likelihood": lk}
            lookA = lambda q, s, entry=False: bankA[q][s] if entry else s
            res["per-shadow"] = np.nanmean(
                sample_estimates(lookA, mdf, 2, sids, target, obs, RESAMPLES, rng), axis=0)

            # --- variant B: one classifier per (qubit, basis pattern), pooled ---
            pat = basis_pattern(mdf, 2)
            groups = collections.defaultdict(list)
            for s in sids: groups[pat[s]].append(s)
            bankB = {q: {} for q in range(2)}
            for q in range(2):
                for g, members in groups.items():
                    x, yy = pooled_training_data(mdf, q, members)
                    m, lk = train_gp_classifier(x, yy, num_inducing=min(INDUCING, len(yy)),
                                                training_iter=ITERS, lr=0.05, kernel="matern32")
                    m.eval(); lk.eval(); bankB[q][g] = {"model": m, "likelihood": lk}
            lookB = lambda q, s, entry=False: bankB[q][pat[s]] if entry else pat[s]
            res["pooled"] = np.nanmean(
                sample_estimates(lookB, mdf, 2, sids, target, obs, RESAMPLES, rng), axis=0)

            for k, curve in res.items():
                rmse = float(np.sqrt(np.nanmean((curve-truth)**2)))
                rows.append({"observable": obs, "seed": seed, "arm": k, "rmse": round(rmse,5)})
            print(f"  {obs} seed={seed} " + "  ".join(
                f"{k}={float(np.sqrt(np.nanmean((v-truth)**2))):.4f}" for k,v in res.items())
                + f"   ({_time.perf_counter()-t0:.0f}s)  [{len(groups)} basis groups]", flush=True)

    print("\n==== POOLED vs PER-SHADOW CONDITIONAL ====", flush=True)
    summ=[]
    for obs in OBSERVABLES:
        for arm in ("raw","gp","per-shadow","pooled"):
            v=[r["rmse"] for r in rows if r["observable"]==obs and r["arm"]==arm]
            if not v: continue
            m=float(np.mean(v)); e=float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0
            summ.append({"observable":obs,"arm":arm,"n_seeds":len(v),
                         "rmse_mean":round(m,5),"rmse_se":round(e,5)})
            print(f"  {obs:2s} {arm:11s} rmse={m:.4f} +/- {e:.4f}  (n={len(v)})", flush=True)
    wall=_time.perf_counter()-t0
    hdr=(f"# conditional route: per-shadow vs basis-pooled classifiers, identical data; "
         f"{NT} times x {NS} shadows, {len(SEEDS)} seeds, inducing={INDUCING}, iters={ITERS}, "
         f"resamples={RESAMPLES}; wall={wall:.0f}s")
    for fn,data in [("pooled_conditional.csv",rows),("pooled_conditional_summary.csv",summ)]:
        with open(os.path.join(_HERE,fn),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); f.write(hdr+"\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved pooled_conditional{{,_summary}}.csv", flush=True)

if __name__=="__main__":
    main()
