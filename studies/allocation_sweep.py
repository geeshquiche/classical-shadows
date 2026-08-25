#!/usr/bin/env python3
"""Matched-budget allocation: observation times against shadows per time, under the FINAL estimator.

Section 5.2 of the report claims a fixed-estimator sweep over (times x shots) placed the optimum at an
intermediate allocation, but no ledger for that claim survives, and the nearest surviving dataset
(synthetic_XI_hpc) shows the opposite -- breadth winning at nearly every matched budget.  This re-runs the
question under the method the report actually uses: fresh bases per time, matched-count per-time estimates,
Laplace-smoothed per-point SE, empirical-noise Matern-3/2 GP.

Total budget T = times x shadows is held fixed within each group.  Errors are RMSE against the exact
trajectory on a common dense grid, so allocations are compared on the same target.
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
from test_pipeline_coverage import exact_curve, OPS, Z95
from coverage_basis_ablation import generate_rerandomised_measurement_df
from final_config_coverage import per_time_series, gp_fit, SUPPORTS
from Synthetic_Error_Uncertainty_Check import make_cell_seed

QUICK = os.environ.get("QUICK","0")=="1"
SEEDS = [10,11] if QUICK else list(range(10,10+int(os.environ.get("NSEED","10"))))
TRUE_T, PRED_T = 400, 100
# (total budget, [(times, shadows), ...]) -- breadth to depth at each matched budget
GROUPS = ([(8000, [(40,200),(80,100)])] if QUICK else
          [(20000, [(50,400),(100,200),(200,100)]),
           (40000, [(100,400),(200,200),(400,100)])])
OBS = os.environ.get("OBS","XI,ZZ").split(",")

def main():
    t0=_time.perf_counter()
    tl=np.linspace(0,2*np.pi,TRUE_T); tgt=np.linspace(0,2*np.pi,PRED_T)
    states=qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tl, []).states
    truth={n: exact_curve(states,tl,op,tgt) for n,op in OPS.items()}
    rows=[]; per={}
    for budget, allocs in GROUPS:
        for (nt, ns_) in allocs:
            observed=np.linspace(0,TRUE_T-1,nt,dtype=int)
            for seed in SEEDS:
                cell=make_cell_seed(seed,"ALLOC",ns_,nt,3); np.random.seed(cell)
                mdf=generate_rerandomised_measurement_df(states,tl,observed,2,ns_,
                                                        shots_per_setting=1,seed=cell)
                for name in OBS:
                    ot,series=per_time_series(mdf,SUPPORTS[name])
                    y,se=series["matched"]
                    keep=~np.isnan(y)
                    mean,std,se_t=gp_fit(ot[keep],y[keep],se[keep],tgt,"empirical")
                    rmse=float(np.sqrt(np.mean((mean-truth[name])**2)))
                    cov=float(np.mean(np.abs(mean-truth[name])<=Z95*np.sqrt(std**2+se_t**2)))
                    per.setdefault((budget,nt,ns_,name),{})[seed]=(rmse,cov)
                    rows.append({"budget":budget,"times":nt,"shadows":ns_,"observable":name,
                                 "seed":seed,"rmse":round(rmse,5),"coverage_95":round(cov,4)})
            print(f"  budget={budget} {nt}x{ns_} done ({_time.perf_counter()-t0:.0f}s)",flush=True)
    n=len(SEEDS); summ=[]
    print("\n==== MATCHED-BUDGET ALLOCATION (final estimator) ====",flush=True)
    for (budget,nt,ns_,name),d in sorted(per.items()):
        a=np.array([d[s] for s in SEEDS])
        m=a[:,0].mean(); e=a[:,0].std(ddof=1)/np.sqrt(n)
        summ.append({"budget":budget,"times":nt,"shadows":ns_,"observable":name,"n_seeds":n,
                     "rmse_mean":round(float(m),5),"rmse_se":round(float(e),5),
                     "coverage_mean":round(float(a[:,1].mean()),4)})
        print(f"  {name:2s} budget={budget:6d} {nt:3d} times x {ns_:3d} shadows  "
              f"rmse={m:.4f}+/-{e:.4f}  cov={a[:,1].mean():.3f}",flush=True)
    wall=_time.perf_counter()-t0
    hdr=(f"# matched-budget allocation, times vs shadows per time, FINAL estimator (fresh bases, "
         f"matched-count, smoothed SE, empirical-noise Matern-3/2); {n} seeds; wall={wall:.0f}s")
    for fn,data in [("allocation_sweep.csv",rows),("allocation_sweep_summary.csv",summ)]:
        with open(os.path.join(_HERE,fn),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); f.write(hdr+"\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved allocation_sweep{{,_summary}}.csv",flush=True)

if __name__=="__main__":
    main()
