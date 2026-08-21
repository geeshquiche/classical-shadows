#!/usr/bin/env python3
"""Conditional / autoregressive density-matrix reconstruction (extracted from the
Multi_Qubit_Ising_Subsystem notebook) as an importable module, so it can be compared
head-to-head with the matrix-element GP-regression route on identical data.

Nothing here changes the method — the functions are lifted from the notebook (learnable-alpha
Bernoulli-GP classifiers, autoregressive P(m_q | t, m_<q, shadow), shadow-matrix reconstruction).
"""
from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import qutip as qt
import torch
import gpytorch
from torch.distributions import Bernoulli

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from quantum_part import create_random_pauli_obs, pauli_measurement
from classical_shadow_matrix import pauli_eigen
from Bayesian_part import GPBinaryVI


@dataclass
class CondConfig:
    shadow_size: int = 200
    num_inducing: int = 40
    training_iter: int = 80
    gp_lr: float = 0.05
    gp_kernel: str = "matern32"
    alpha_initial: float = 5.0
    alpha_range: tuple = (1.0, 20.0)
    prediction_resamples: int = 20


# ───────────────── learnable-alpha Bernoulli GP classifier (from notebook) ─────────────────
class ScaledSigmoidBernoulliLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, alpha_init=1.0, alpha_bounds=(0.05, 20.0)):
        super().__init__()
        lower, upper = map(float, alpha_bounds)
        if not lower < float(alpha_init) < upper:
            raise ValueError("alpha_init must lie strictly inside alpha_bounds.")
        self.register_parameter("raw_alpha", torch.nn.Parameter(torch.zeros(())))
        self.register_constraint("raw_alpha", gpytorch.constraints.Interval(lower, upper))
        self.alpha = torch.as_tensor(float(alpha_init))

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        value = torch.as_tensor(value, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, function_samples, **kwargs):
        return Bernoulli(logits=self.alpha * function_samples)


def train_gp_classifier_with_alpha(train_x, train_y, num_inducing=50, training_iter=200,
                                   lr=0.05, kernel="matern32", alpha_init=1.0,
                                   alpha_bounds=(0.05, 20.0)):
    train_y = train_y.clone().float()
    unique_y = set(train_y.unique().tolist())
    if unique_y.issubset({-1.0, 1.0}):
        train_y = ((train_y + 1.0) / 2.0).float()
    elif not unique_y.issubset({0.0, 1.0}):
        raise ValueError("train_y must be binary in {0,1} or {-1,+1}.")
    unique_x = torch.unique(train_x, dim=0)
    unique_x = unique_x[torch.argsort(unique_x[:, 0])]
    inducing_count = min(int(num_inducing), unique_x.size(0))
    inducing_index = torch.linspace(0, unique_x.size(0) - 1, inducing_count).long()
    inducing_points = unique_x[inducing_index].clone()
    model = GPBinaryVI(inducing_points, kernel=kernel)
    likelihood = ScaledSigmoidBernoulliLikelihood(alpha_init=alpha_init, alpha_bounds=alpha_bounds)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    losses = []
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, likelihood, losses


def predict_gp_classifier_with_alpha(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_dist = model(test_x)
        predictive_dist = likelihood(latent_dist)
        prob = predictive_dist.mean
        latent_mean = latent_dist.mean
        latent_var = latent_dist.variance
    while prob.dim() > latent_mean.dim():
        prob = prob.mean(dim=0)
    return prob, latent_mean, latent_var


# ─────────────────────── dynamics + shadow-matrix helpers (from notebook) ──────────────────
def one_site_op(op, site, n):
    return qt.tensor([op if q == site else qt.qeye(2) for q in range(n)])


def two_site_op(op_a, site_a, op_b, site_b, n):
    return qt.tensor([op_a if q == site_a else op_b if q == site_b else qt.qeye(2)
                      for q in range(n)])


def build_plus_initial_state(n):
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return qt.tensor([plus for _ in range(n)])


def build_ising_hamiltonian(n, j=1.0, transverse_h=0.5, longitudinal_h=0.0):
    hamiltonian = 0
    for q in range(n - 1):
        hamiltonian = hamiltonian + j * two_site_op(qt.sigmaz(), q, qt.sigmaz(), q + 1, n)
    for q in range(n):
        hamiltonian = hamiltonian + transverse_h * one_site_op(qt.sigmax(), q, n)
        if longitudinal_h != 0.0:
            hamiltonian = hamiltonian + longitudinal_h * one_site_op(qt.sigmaz(), q, n)
    return hamiltonian


def reduced_density_matrix(state, subsystem_qubits):
    rho = state if state.isoper else state * state.dag()
    return rho.ptrace(list(subsystem_qubits)).full()


def local_shadow_matrix(pauli_label, outcome):
    ket = pauli_eigen[str(pauli_label)][int(outcome)]
    return (3.0 * (ket * ket.dag()) - qt.qeye(2)).full()


def subsystem_shadow_matrix(shot_rows, subsystem_qubits):
    ordered = shot_rows.set_index("qubit").loc[list(subsystem_qubits)]
    factors = [local_shadow_matrix(row.pauli, row.outcome) for row in ordered.itertuples()]
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def generate_repeated_measurement_df(states, t_grid, observed_indices, total_qubits,
                                     shadow_size, shots_per_setting=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    pauli_settings = [create_random_pauli_obs(total_qubits) for _ in range(shadow_size)]
    records = []
    for idx in observed_indices:
        state_t = states[int(idx)]
        rho_t = state_t if state_t.isoper else state_t * state_t.dag()
        time = float(t_grid[int(idx)])
        for shadow_id, pauli_setting in enumerate(pauli_settings):
            for shot_repeat in range(int(shots_per_setting)):
                measurement = pauli_measurement(rho_t, pauli_setting, total_qubits)
                for qubit, (pauli_label, outcome) in enumerate(measurement):
                    records.append({"time": time, "shadow_id": int(shadow_id),
                                    "shot_repeat": int(shot_repeat), "qubit": int(qubit),
                                    "pauli": str(pauli_label), "outcome": int(outcome)})
    return pd.DataFrame(records)


# ─────────────────────── autoregressive training + reconstruction ──────────────────────────
def build_repeated_conditional_training_data(measurement_df, target_qubit, previous_qubits, shadow_id):
    needed_qubits = list(previous_qubits) + [int(target_qubit)]
    block = measurement_df.loc[
        (measurement_df["shadow_id"] == int(shadow_id))
        & measurement_df["qubit"].isin(needed_qubits),
        ["time", "shot_repeat", "qubit", "outcome"],
    ]
    if block.empty:
        raise ValueError(f"No data for shadow_id={shadow_id}.")
    pivot = block.pivot_table(index=["time", "shot_repeat"], columns="qubit",
                              values="outcome", aggfunc="first")
    missing = [q for q in needed_qubits if q not in pivot.columns]
    if missing:
        raise ValueError(f"Missing qubits {missing} for shadow_id={shadow_id}.")
    data = pivot[needed_qubits].dropna().sort_index()
    times = data.index.get_level_values("time").to_numpy(dtype=np.float32).reshape(-1, 1)
    if previous_qubits:
        prev = data[list(previous_qubits)].to_numpy(dtype=np.float32)
        x = np.concatenate([times, prev], axis=1)
    else:
        x = times
    y = data[int(target_qubit)].to_numpy(dtype=np.float32)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_subsystem_autoregressive_models(measurement_df, subsystem_qubits, cfg):
    model_bank = {int(q): {} for q in subsystem_qubits}
    alpha_records = []
    for q_pos, q in enumerate(subsystem_qubits):
        previous_qubits = tuple(subsystem_qubits[:q_pos])
        for shadow_id in range(cfg.shadow_size):
            train_x, train_y = build_repeated_conditional_training_data(
                measurement_df, target_qubit=q, previous_qubits=previous_qubits, shadow_id=shadow_id)
            model, likelihood, _ = train_gp_classifier_with_alpha(
                train_x, train_y,
                num_inducing=min(cfg.num_inducing, torch.unique(train_x, dim=0).size(0)),
                training_iter=cfg.training_iter, lr=cfg.gp_lr, kernel=cfg.gp_kernel,
                alpha_init=cfg.alpha_initial, alpha_bounds=cfg.alpha_range)
            model_bank[int(q)][int(shadow_id)] = {
                "model": model, "likelihood": likelihood, "previous_qubits": previous_qubits}
            alpha_records.append({"qubit": int(q), "shadow_id": int(shadow_id),
                                  "alpha": float(likelihood.alpha.detach().cpu())})
    return model_bank, pd.DataFrame(alpha_records)


def pauli_table_for_subsystem(measurement_df, subsystem_qubits, shadow_size):
    table = (measurement_df.loc[measurement_df["qubit"].isin(subsystem_qubits),
                                ["shadow_id", "qubit", "pauli"]]
             .drop_duplicates(subset=["shadow_id", "qubit"])
             .pivot(index="shadow_id", columns="qubit", values="pauli")
             .reindex(index=range(shadow_size), columns=list(subsystem_qubits)))
    if table.isna().any().any():
        raise ValueError("Missing Pauli setting for at least one subsystem qubit/shadow.")
    return table


def sample_subsystem_shadow_matrix(model_bank, pauli_table, subsystem_qubits, shadow_id, time_value):
    sampled = {}
    local_factors = []
    for q_pos, q in enumerate(subsystem_qubits):
        previous_qubits = tuple(subsystem_qubits[:q_pos])
        features = [float(time_value)] + [float(sampled[pq]) for pq in previous_qubits]
        test_x = torch.tensor([features], dtype=torch.float32)
        entry = model_bank[int(q)][int(shadow_id)]
        prob, _, _ = predict_gp_classifier_with_alpha(entry["model"], entry["likelihood"], test_x)
        p_plus = float(torch.clamp(prob.reshape(-1)[0], 1e-6, 1.0 - 1e-6).cpu())
        outcome = 1 if np.random.random() < p_plus else -1
        sampled[int(q)] = outcome
        local_factors.append(local_shadow_matrix(pauli_table.loc[int(shadow_id), int(q)], outcome))
    shadow = local_factors[0]
    for factor in local_factors[1:]:
        shadow = np.kron(shadow, factor)
    return shadow


def reconstruct_subsystem_density_dynamics(model_bank, measurement_df, subsystem_qubits, target_times, cfg):
    pauli_table = pauli_table_for_subsystem(measurement_df, subsystem_qubits, cfg.shadow_size)
    dim = 2 ** len(subsystem_qubits)
    samples = np.zeros((cfg.prediction_resamples, len(target_times), dim, dim), dtype=np.complex128)
    for r in range(cfg.prediction_resamples):
        for t_idx, time_value in enumerate(target_times):
            rho_hat = np.zeros((dim, dim), dtype=np.complex128)
            for shadow_id in range(cfg.shadow_size):
                rho_hat += sample_subsystem_shadow_matrix(
                    model_bank, pauli_table, subsystem_qubits, shadow_id, float(time_value))
            samples[r, t_idx] = rho_hat / float(cfg.shadow_size)
    return samples, samples.mean(axis=0), samples.std(axis=0)


def averaged_subsystem_shadow_matrices(measurement_df, subsystem_qubits):
    """Direct (non-Bayesian) subsystem shadow matrices per observed time — the INPUT the
    matrix-element GP route regresses. Averages subsystem_shadow_matrix over shadows/shots."""
    dim = 2 ** len(subsystem_qubits)
    times = np.sort(measurement_df["time"].unique())
    out = np.zeros((len(times), dim, dim), dtype=np.complex128)
    for ti, t in enumerate(times):
        block = measurement_df[measurement_df["time"] == t]
        acc = np.zeros((dim, dim), dtype=np.complex128)
        n = 0
        for _, shot_rows in block.groupby(["shadow_id", "shot_repeat"]):
            acc += subsystem_shadow_matrix(shot_rows, subsystem_qubits)
            n += 1
        out[ti] = acc / max(n, 1)
    return times, out
