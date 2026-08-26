#!/usr/bin/env python3
"""Independent verification that the code computes what the report says it does.

Every number in the report has been checked against the result CSVs.  This checks the layer underneath:
that the CSVs mean what they claim.  Each test rebuilds a quantity a second, independent way and compares.
"""
import os, sys, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR","/tmp"),"mplcfg"))
import numpy as np, qutip as qt
warnings.filterwarnings("ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_HERE)
for p in (_P,_HERE,os.path.join(_P,"rho_reconstruction"),os.path.join(_P,"calibration_and_adaptive")):
    if p not in sys.path: sys.path.append(p)
from Synthetic_Error_Uncertainty_Check import (build_hamiltonian, build_initial_state,
    pauli_string_operator, observable_support, generate_measurement_df, make_cell_seed)
from coverage_basis_ablation import generate_rerandomised_measurement_df
from routes3_final import matched_series
import conditional_rho as cr
from final_config_coverage import per_time_series, gp_fit, SUPPORTS

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1;  print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:    fail += 1; print(f"  FAIL  {name}   [{detail}]")

print("== 1. Hamiltonian and initial state match the report's equations ==")
n = 2
H = build_hamiltonian("tfim", n)
I, X, Z = qt.qeye(2), qt.sigmax(), qt.sigmaz()
H_ref = qt.tensor(Z, Z) + 0.5*(qt.tensor(X, I) + qt.tensor(I, X))   # J=1, g=0.5
check("2-qubit TFIM equals J*ZZ + g*(X0+X1), J=1 g=0.5",
      np.allclose(H.full(), H_ref.full()), f"max|diff|={np.abs(H.full()-H_ref.full()).max():.2e}")
H4 = build_hamiltonian("tfim", 4)
H4_ref = sum(qt.tensor(*[Z if k in (q,q+1) else I for k in range(4)]) for q in range(3)) \
       + 0.5*sum(qt.tensor(*[X if k==q else I for k in range(4)]) for q in range(4))
check("4-qubit chain is the nearest-neighbour sum in the report",
      np.allclose(H4.full(), H4_ref.full()), f"max|diff|={np.abs(H4.full()-H4_ref.full()).max():.2e}")
psi0 = build_initial_state(2)
plus = (qt.basis(2,0)+qt.basis(2,1)).unit()
check("initial state is |+>^{otimes n}", np.allclose(psi0.full().ravel(), qt.tensor(plus,plus).full().ravel()))
check("conditional_rho builds the same Hamiltonian",
      np.allclose(cr.build_ising_hamiltonian(2).full(), H_ref.full()))

print("\n== 2. Qubit ordering: does 'qubit 0' mean the same thing to the operator and the sampler? ==")
# X0 on |+>|+> has expectation 1 at t=0 and Z0 has 0; an ordering swap would be invisible here, so use
# an asymmetric state to distinguish qubit 0 from qubit 1
psi_asym = qt.tensor(qt.basis(2,0), plus)          # qubit 0 in |0>, qubit 1 in |+>
Z0 = pauli_string_operator("ZI", 2); Z1 = pauli_string_operator("IZ", 2)
check("Z0 reads qubit 0 (expect +1), Z1 reads qubit 1 (expect 0)",
      abs(qt.expect(Z0, psi_asym)-1) < 1e-9 and abs(qt.expect(Z1, psi_asym)) < 1e-9,
      f"<Z0>={qt.expect(Z0,psi_asym):.3f} <Z1>={qt.expect(Z1,psi_asym):.3f}")
check("observable_support('ZI') is qubit 0", observable_support("ZI") == [(0,"Z")],
      str(observable_support("ZI")))
check("observable_support('IZ') is qubit 1", observable_support("IZ") == [(1,"Z")],
      str(observable_support("IZ")))

print("\n== 3. Both matched-count implementations are unbiased and agree ==")
tl = np.linspace(0, 2*np.pi, 40)
states = qt.mesolve(H, psi0, tl, []).states
observed = np.arange(0, 40, 8)
truth_xi = np.array([qt.expect(pauli_string_operator("XI",2), states[i]) for i in observed])
truth_zz = np.array([qt.expect(pauli_string_operator("ZZ",2), states[i]) for i in observed])
est_xi, est_zz, alt_xi = [], [], []
for r in range(30):
    seed = make_cell_seed(1000+r, "VERIFY", 400, len(observed), 0)
    # the final-method path: fresh bases per time, read by per_time_series
    np.random.seed(seed)
    mdf = generate_rerandomised_measurement_df(states, tl, observed, 2, 400,
                                               shots_per_setting=1, seed=seed)
    _, s_xi = per_time_series(mdf, SUPPORTS["XI"]); _, s_zz = per_time_series(mdf, SUPPORTS["ZZ"])
    est_xi.append(s_xi["matched"][0]); est_zz.append(s_zz["matched"][0])
    # the route-comparison path: fixed bases, read by routes3_final.matched_series
    mdf2 = generate_measurement_df(states=states, t_grid=tl, observed_indices=observed,
                                   n=2, shadow_size=400, seed=seed)
    _, y2, _ = matched_series(mdf2, observable_support("XI"))
    alt_xi.append(y2)
mx, mz = np.nanmean(est_xi, 0), np.nanmean(est_zz, 0)
sx = np.nanstd(est_xi, 0, ddof=1)/np.sqrt(30); sz = np.nanstd(est_zz, 0, ddof=1)/np.sqrt(30)
zx = np.abs(mx-truth_xi)/np.maximum(sx,1e-12); zz_ = np.abs(mz-truth_zz)/np.maximum(sz,1e-12)
check("matched-count <X0> unbiased over 30 datasets (all |z|<3)", zx.max() < 3, f"max|z|={zx.max():.2f}")
check("matched-count <Z0Z1> unbiased over 30 datasets (all |z|<3)", zz_.max() < 3, f"max|z|={zz_.max():.2f}")
ma = np.nanmean(alt_xi, 0); sa = np.nanstd(alt_xi, 0, ddof=1)/np.sqrt(30)
za = np.abs(ma-truth_xi)/np.maximum(sa,1e-12)
check("the OTHER implementation (routes3 matched_series, fixed bases) is also unbiased",
      za.max() < 3, f"max|z|={za.max():.2f}")
zab = np.abs(mx-ma)/np.maximum(np.hypot(sx,sa),1e-12)
check("the two implementations agree on the same observable", zab.max() < 3, f"max|z|={zab.max():.2f}")

print("\n== 4. Single-shot shadow variance equals 3^k - <O>^2 ==")
seed = make_cell_seed(7, "VERIFY", 4000, 3, 0)
obs3 = observed[:3]
np.random.seed(seed)
mdf = generate_rerandomised_measurement_df(states, tl, obs3, 2, 4000, shots_per_setting=1, seed=seed)
for name, k in (("XI",1), ("ZZ",2)):
    support = SUPPORTS[name]
    t0 = np.sort(mdf["time"].unique())[0]
    block = mdf[mdf["time"] == t0]
    singles = []
    for _sid, shot in block.groupby(["shadow_id","shot_repeat"]):
        row = {int(r.qubit): (str(r.pauli), int(r.outcome)) for r in shot.itertuples(index=False)}
        if all(row[q][0] == lab for q, lab in support):
            pr = 1.0
            for q, _l in support: pr *= row[q][1]
            singles.append((3.0**k)*pr)
        else:
            singles.append(0.0)
    singles = np.asarray(singles)
    exp_true = float(qt.expect(pauli_string_operator(name,2), states[obs3[0]]))
    predicted = 3.0**k - exp_true**2
    measured = singles.var(ddof=1)
    rel = abs(measured-predicted)/predicted
    check(f"{name}: single-shot variance ~ 3^{k} - <O>^2", rel < 0.08,
          f"measured {measured:.3f} vs predicted {predicted:.3f} ({100*rel:.1f}%)")

print("\n== 5. gp_fit supplies per-point noise and scores against the latent function ==")
ot, series = per_time_series(mdf, SUPPORTS["XI"])
y, se = series["matched"]
keep = ~np.isnan(y)
tgt = np.linspace(0, 2*np.pi, 25)
mean_a, std_a, set_a = gp_fit(ot[keep], y[keep], se[keep], tgt, "empirical")
mean_b, std_b, set_b = gp_fit(ot[keep], y[keep], se[keep]*4.0, tgt, "empirical")
check("inflating the supplied SE widens the returned observation noise",
      np.mean(set_b) > 1.8*np.mean(set_a), f"se_tgt {np.mean(set_a):.4f} -> {np.mean(set_b):.4f}")
check("inflating the supplied SE also smooths the mean (less noise-chasing)",
      np.std(np.diff(mean_b)) < np.std(np.diff(mean_a)),
      f"roughness {np.std(np.diff(mean_a)):.4f} -> {np.std(np.diff(mean_b)):.4f}")
check("gp_fit returns a finite posterior sd everywhere", np.all(np.isfinite(std_a)) and np.all(std_a > 0))

print(f"\n==== {ok} passed, {fail} failed ====")
sys.exit(1 if fail else 0)
