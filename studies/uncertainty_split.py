#!/usr/bin/env python3
"""Aleatoric/epistemic decomposition of the reconstruction uncertainty, by resampling.

Section 3 defines the split by the law of total variance and says it is estimated by refitting many
independent datasets; the Introduction lists it as a contribution.  No ledger for it survived, so this
measures it under the final method.

For S independent shadow datasets of the same size, each fitted independently:
    epistemic(t) = Var_s[ mean_s(t) ]           spread of the posterior mean across datasets
    aleatoric(t) = E_s[ Var_s(t) ]              mean within-dataset predictive variance
    total(t)     = aleatoric(t) + epistemic(t)  (law of total variance)
Reported as the time-averaged shares, at two sampling densities so the trade can be seen to move.
"""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR","/tmp"),"mplcfg"))
import numpy as np, qutip as qt
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_HERE)
for p in (_P,_HERE,os.path.join(_P,"rho_reconstruction"),os.path.join(_P,"calibration_and_adaptive")):
    if p not in sys.path: sys.path.append(p)
import conditional_rho as cr
from test_pipeline_coverage import exact_curve, OPS
from coverage_basis_ablation import generate_rerandomised_measurement_df
from final_config_coverage import per_time_series, gp_fit, SUPPORTS
from Synthetic_Error_Uncertainty_Check import make_cell_seed

QUICK = os.environ.get("QUICK","0")=="1"
SEEDS = list(range(10, 10+int(os.environ.get("NSEED","2" if QUICK else "20"))))
CONFIGS = [(20,60)] if QUICK else [(20,60),(100,200)]
TRUE_T, PRED_T = 400, 100

def main():
    t0=_time.perf_counter()
    tl=np.linspace(0,2*np.pi,TRUE_T); tgt=np.linspace(0,2*np.pi,PRED_T)
    states=qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tl, []).states
    truth={n: exact_curve(states,tl,op,tgt) for n,op in OPS.items()}
    rows=[]
    for (nt,ns_) in CONFIGS:
        observed=np.linspace(0,TRUE_T-1,nt,dtype=int)
        means={n: [] for n in SUPPORTS}; varis={n: [] for n in SUPPORTS}
        for seed in SEEDS:
            cell=make_cell_seed(seed,"SPLIT",ns_,nt,11); np.random.seed(cell)
            mdf=generate_rerandomised_measurement_df(states,tl,observed,2,ns_,
                                                    shots_per_setting=1,seed=cell)
            for name in SUPPORTS:
                ot,series=per_time_series(mdf,SUPPORTS[name])
                y,se=series["matched"]; keep=~np.isnan(y)
                mean,std,se_t=gp_fit(ot[keep],y[keep],se[keep],tgt,"empirical")
                means[name].append(mean)
                varis[name].append(std**2 + se_t**2)     # within-dataset predictive variance
        for name in SUPPORTS:
            M=np.array(means[name]); V=np.array(varis[name])
            epi=M.var(axis=0, ddof=1)          # spread of the posterior mean across datasets
            ale=V.mean(axis=0)                 # mean within-dataset predictive variance
            tot=ale+epi
            bias=(M.mean(axis=0)-truth[name])**2
            rows.append({"times":nt,"shadows":ns_,"observable":name,"n_seeds":len(SEEDS),
                         "aleatoric_mean":round(float(ale.mean()),6),
                         "epistemic_mean":round(float(epi.mean()),6),
                         "total_mean":round(float(tot.mean()),6),
                         "aleatoric_share_pct":round(float(100*ale.mean()/tot.mean()),1),
                         "epistemic_share_pct":round(float(100*epi.mean()/tot.mean()),1),
                         "sq_bias_mean":round(float(bias.mean()),6)})
            print(f"  {name:2s} {nt:3d}x{ns_:3d}  aleatoric {100*ale.mean()/tot.mean():5.1f}%   "
                  f"epistemic {100*epi.mean()/tot.mean():5.1f}%   total var {tot.mean():.5f}   "
                  f"({_time.perf_counter()-t0:.0f}s)", flush=True)
    wall=_time.perf_counter()-t0
    hdr=(f"# aleatoric/epistemic decomposition by resampling, final method (fresh bases, matched-count, "
         f"empirical noise, Matern-3/2); {len(SEEDS)} datasets per configuration; wall={wall:.0f}s")
    with open(os.path.join(_HERE,"uncertainty_split.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); f.write(hdr+"\n"); w.writeheader(); w.writerows(rows)
    print(f"\nwall={wall:.0f}s -> saved uncertainty_split.csv", flush=True)

if __name__=="__main__":
    main()
