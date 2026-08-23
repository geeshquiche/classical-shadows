#!/usr/bin/env python3
"""Run the synthetic error/uncertainty check and save all notebook plots.

This is the script version of Synthetic_Error_Uncertainty_Check.ipynb for HPC
batch runs.  Defaults match the current notebook configuration, but most values
can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
import torch

from Bayesian_part import (
    predict_gp_classifier,
    train_conditional_gp_classifiers,
    train_gp_classifier,
)
from quantum_part import (
    create_classical_shadow,
    create_random_pauli_obs,
    estimation_mean,
    pauli_measurement,
    reconstruct_operator_dynamics_from_conditional_models,
    simulate_classical_shadow,
)


PAULI_OPS = {"I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
COVERAGE_Z_VALUES = {"1sigma": 1.0, "95pct": 1.959963984540054}


@dataclass
class Config:
    base_seed: int = 10
    qubit_num: int = 2
    observable_string: str = "XI"
    dynamics_name: str = "tfim"
    num_true_time_points: int = 1000
    prediction_time_count: int = 200
    time_min: float = 0.0
    time_max: float = 2.0 * np.pi
    shadow_size_grid: tuple[int, ...] = (500, 600, 700, 800, 900, 1000)
    num_time_index_grid: tuple[int, ...] = (500, 600, 700, 800, 900, 1000)
    num_dataset_repeats: int = 2
    prediction_resamples: int = 50
    gp_kernel: str = "matern32"
    normal_training_iter: int = 120
    conditional_training_iter: int = 80
    num_inducing: int = 40
    gp_lr: float = 0.05
    use_matched_basis_estimator_for_pauli_strings: bool = False
    independent_seeds: bool = True   # independent draws per cell (False gives nested-prefix draws)
    inspect_shadow_size: int | None = None
    inspect_num_time_indices: int | None = None
    inspect_repeat_id: int = 0
    output_dir: Path = Path("Synthetic_Error_Uncertainty_Check_outputs")
    plot_formats: tuple[str, ...] = ("png",)


def parse_int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one integer")
    return values


def parse_plot_formats(text: str) -> tuple[str, ...]:
    values = tuple(v.strip().lstrip(".").lower() for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("plot format list cannot be empty")
    return values


def pauli_string_operator(pauli_string: str, qubit_num: int) -> qt.Qobj:
    labels = list(pauli_string.upper())
    if len(labels) != qubit_num:
        raise ValueError(f"observable_string must have length {qubit_num}.")
    if any(label not in PAULI_OPS for label in labels):
        raise ValueError("Only I, X, Y, Z labels are supported.")
    return qt.tensor([PAULI_OPS[label] for label in labels])


def observable_support(pauli_string: str) -> list[tuple[int, str]]:
    return [(q, label) for q, label in enumerate(pauli_string.upper()) if label != "I"]


_PAULI_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _observable_to_int(observable_string: str) -> int:
    """Stable (PYTHONHASHSEED-independent) integer encoding of a Pauli string."""
    value = 0
    for label in observable_string.upper():
        value = value * 4 + _PAULI_TO_INT.get(label, 0)
    return value


def make_cell_seed(
    base_seed: int,
    observable_string: str,
    shadow_size: int,
    num_time_indices: int,
    repeat_id: int,
) -> int:
    """Distinct, reproducible, collision-free seed for one experiment cell.

    Hashes the (base_seed, observable, shadow_size, num_time_indices, repeat_id)
    tuple with numpy's SeedSequence.  This avoids the collisions an additive
    formula (base + 100000*repeat + 1000*shadow + time) suffers when its linear
    terms overlap in magnitude -- e.g. (repeat=1, shadow=500) and
    (repeat=0, shadow=600) both map to base+600000+time.  Every distinct cell
    tuple yields an independent, well-mixed 32-bit seed.
    """
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            _observable_to_int(observable_string),
            int(shadow_size),
            int(num_time_indices),
            int(repeat_id),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def one_site_op(op: qt.Qobj, site: int, n: int) -> qt.Qobj:
    return qt.tensor([op if q == site else qt.qeye(2) for q in range(n)])


def two_site_op(op_a: qt.Qobj, site_a: int, op_b: qt.Qobj, site_b: int, n: int) -> qt.Qobj:
    return qt.tensor(
        [op_a if q == site_a else op_b if q == site_b else qt.qeye(2) for q in range(n)]
    )


def build_initial_state(n: int) -> qt.Qobj:
    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)
    plus = (ket0 + ket1).unit()
    return qt.tensor([plus for _ in range(n)])


def build_hamiltonian(name: str, n: int) -> qt.Qobj:
    if name == "ising_zz":
        hamiltonian = 0
        for q in range(n - 1):
            hamiltonian = hamiltonian + two_site_op(qt.sigmaz(), q, qt.sigmaz(), q + 1, n)
        return hamiltonian
    if name == "tfim":
        hamiltonian = 0
        for q in range(n - 1):
            hamiltonian = hamiltonian + two_site_op(qt.sigmaz(), q, qt.sigmaz(), q + 1, n)
        for q in range(n):
            hamiltonian = hamiltonian + 0.5 * one_site_op(qt.sigmax(), q, n)
        return hamiltonian
    if name == "local_x":
        hamiltonian = 0
        for q in range(n):
            hamiltonian = hamiltonian + one_site_op(qt.sigmax(), q, n)
        return hamiltonian
    if name == "custom":
        return two_site_op(qt.sigmaz(), 0, qt.sigmaz(), 1, n)
    raise ValueError("Unknown dynamics_name.")


def generate_measurement_df(states, t_grid, observed_indices, n, shadow_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    obs_list = [create_random_pauli_obs(n) for _ in range(shadow_size)]
    records = []
    for idx in observed_indices:
        state_t = states[int(idx)]
        rho_t = state_t if state_t.isoper else state_t * state_t.dag()
        time = float(t_grid[int(idx)])
        for shadow_id, pauli_setting in enumerate(obs_list):
            measurement = pauli_measurement(rho_t, pauli_setting, n)
            for qubit, (pauli_label, outcome) in enumerate(measurement):
                records.append(
                    {
                        "time": time,
                        "shadow_id": int(shadow_id),
                        "qubit": int(qubit),
                        "pauli": str(pauli_label),
                        "outcome": int(outcome),
                    }
                )
    return pd.DataFrame(records)


def matched_pauli_string_estimate_from_shots(shots, pauli_string):
    support = observable_support(pauli_string)
    values = []
    for shot in shots:
        if all(shot[q][0] == label for q, label in support):
            prod = 1
            for q, _label in support:
                prod *= int(shot[q][1])
            values.append(prod)
    return np.nan if not values else float(np.mean(values))


def estimate_from_shots(shots, observable, pauli_string, use_matched_basis):
    if use_matched_basis:
        return matched_pauli_string_estimate_from_shots(shots, pauli_string)
    shadows = create_classical_shadow(shots)
    return estimation_mean(shadows, observable)


def summarize_prediction_samples(sample_matrix, true_curve):
    samples = np.asarray(sample_matrix, dtype=float)
    pred_mean = np.nanmean(samples, axis=0)
    pred_std = np.nanstd(samples, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros(samples.shape[1])
    err = pred_mean - true_curve
    row = {
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err**2))),
        "max_abs_error": float(np.nanmax(np.abs(err))),
        "mean_uncertainty_std": float(np.nanmean(pred_std)),
        "median_uncertainty_std": float(np.nanmedian(pred_std)),
    }
    for label, z_value in COVERAGE_Z_VALUES.items():
        lower = pred_mean - z_value * pred_std
        upper = pred_mean + z_value * pred_std
        row[f"coverage_{label}"] = float(np.nanmean((true_curve >= lower) & (true_curve <= upper)))
        row[f"mean_width_{label}"] = float(np.nanmean(2.0 * z_value * pred_std))
    return row, pred_mean, pred_std


def fit_single_qubit_per_shadow_probabilities(
    measurement_df,
    target_times,
    cfg: Config,
    shadow_size_current: int,
    support: list[tuple[int, str]],
):
    """Fit only the qubits needed by the requested Pauli observable."""
    all_probs = torch.zeros(len(support), shadow_size_current, len(target_times), dtype=torch.float32)
    test_x = torch.tensor(np.asarray(target_times, dtype=np.float32), dtype=torch.float32).unsqueeze(-1)

    for support_idx, (q, _label) in enumerate(support):
        qubit_data = measurement_df[measurement_df["qubit"] == q]
        for s in sorted(qubit_data["shadow_id"].unique()):
            shadow_data = qubit_data[qubit_data["shadow_id"] == s].sort_values("time")
            train_x = torch.tensor(shadow_data["time"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(-1)
            train_y = torch.tensor(shadow_data["outcome"].to_numpy(dtype=np.float32), dtype=torch.float32)
            model, likelihood = train_gp_classifier(
                train_x,
                train_y,
                num_inducing=min(cfg.num_inducing, len(train_x)),
                training_iter=cfg.normal_training_iter,
                lr=cfg.gp_lr,
                kernel=cfg.gp_kernel,
            )
            prob_star, _, _ = predict_gp_classifier(model, likelihood, test_x)
            all_probs[support_idx, int(s), :] = torch.clamp(prob_star, 1e-6, 1.0 - 1e-6)
    return all_probs


def sample_normal_probability_estimates(
    all_probs: torch.Tensor,
    measurement_df: pd.DataFrame,
    support: list[tuple[int, str]],
    cfg: Config,
) -> np.ndarray:
    """Vectorized classical-shadow resampling for a Pauli-string observable."""
    shadow_size_current = all_probs.shape[1]
    settings = (
        measurement_df.loc[
            (measurement_df["shadow_id"] < shadow_size_current)
            & measurement_df["qubit"].isin([q for q, _label in support]),
            ["shadow_id", "qubit", "pauli"],
        ]
        .drop_duplicates(subset=["shadow_id", "qubit"])
        .pivot(index="shadow_id", columns="qubit", values="pauli")
        .reindex(index=range(shadow_size_current), columns=[q for q, _label in support])
        .to_numpy()
    )
    if pd.isna(settings).any():
        raise ValueError("Missing Pauli settings needed for vectorized resampling.")

    probabilities = all_probs.detach().cpu().numpy().transpose(2, 1, 0)
    outcomes = np.where(
        np.random.random((cfg.prediction_resamples, *probabilities.shape)) < probabilities,
        1.0,
        -1.0,
    )
    matching_basis = np.all(
        settings == np.asarray([label for _q, label in support]), axis=1
    )
    outcome_product = np.prod(outcomes, axis=-1)

    if cfg.use_matched_basis_estimator_for_pauli_strings:
        numerators = (outcome_product * matching_basis[None, None, :]).sum(axis=-1)
        denominators = matching_basis.sum()
        return np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan, dtype=float),
            where=denominators > 0,
        )

    return (3.0 ** len(support)) * np.mean(
        outcome_product * matching_basis[None, None, :], axis=-1
    )


def run_conditional_model(measurement_df, target_times, cfg: Config, shadow_size_current: int, operator):
    model_bank = train_conditional_gp_classifiers(
        measurement_df=measurement_df,
        qubit_num=cfg.qubit_num,
        shadow_ids=range(shadow_size_current),
        num_inducing=cfg.num_inducing,
        training_iter=cfg.conditional_training_iter,
        lr=cfg.gp_lr,
        kernel=cfg.gp_kernel,
    )
    sample_matrix = []
    for _ in range(cfg.prediction_resamples):
        rec = reconstruct_operator_dynamics_from_conditional_models(
            model_bank=model_bank,
            measurement_df=measurement_df,
            qubit_num=cfg.qubit_num,
            operator=operator,
            time_points=target_times,
            shadow_ids=range(shadow_size_current),
        )
        if cfg.use_matched_basis_estimator_for_pauli_strings:
            est = [
                matched_pauli_string_estimate_from_shots(shots, cfg.observable_string)
                for shots in rec["shots"]
            ]
        else:
            est = rec["estimation"]
        sample_matrix.append(est)
    return np.asarray(sample_matrix, dtype=float)


def plot_metric_heatmap(results_df, metric, title=None):
    pivot = results_df.groupby(["num_time_indices", "shadow_size"])[metric].mean().unstack("shadow_size")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("shadow size")
    ax.set_ylabel("number of observed time indices")
    ax.set_title(title or metric)
    fig.colorbar(im, ax=ax, label=metric)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            ax.text(x, y, f"{val:.3g}", ha="center", va="center", color="white")
    fig.tight_layout()
    return fig


def save_figure(fig, output_dir: Path, stem: str, formats: tuple[str, ...]):
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(
    cfg: Config,
    output_dir: Path,
    results_df: pd.DataFrame,
    curve_records: dict,
    tlist: np.ndarray,
    true_values: np.ndarray,
    prediction_times: np.ndarray,
    true_on_prediction_grid: np.ndarray,
    operator,
    support,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tlist, true_values, linewidth=2)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\langle O \rangle$")
    ax.set_title("Synthetic theoretical observable dynamics")
    fig.tight_layout()
    save_figure(fig, output_dir, "01_theoretical_observable_dynamics", cfg.plot_formats)

    heatmap_specs = [
        ("mae", "MAE across synthetic sweep", "02_heatmap_mae"),
        ("rmse", "RMSE across synthetic sweep", "03_heatmap_rmse"),
        ("mean_uncertainty_std", "Mean prediction uncertainty std", "04_heatmap_mean_uncertainty_std"),
        ("coverage_95pct", "95% uncertainty coverage", "05_heatmap_coverage_95pct"),
    ]
    for metric, title, stem in heatmap_specs:
        fig = plot_metric_heatmap(results_df, metric, title)
        save_figure(fig, output_dir, stem, cfg.plot_formats)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for num_t, block in results_df.groupby("num_time_indices"):
        trend = (
            block.groupby("shadow_size")[["mae", "mean_uncertainty_std", "coverage_95pct"]]
            .mean()
            .sort_index()
        )
        x = trend.index.to_numpy(dtype=float)
        axes[0].plot(x, trend["mae"].to_numpy(dtype=float), marker="o", label=f"T={num_t}")
        axes[1].plot(
            x,
            trend["mean_uncertainty_std"].to_numpy(dtype=float),
            marker="o",
            label=f"T={num_t}",
        )
        axes[2].plot(x, trend["coverage_95pct"].to_numpy(dtype=float), marker="o", label=f"T={num_t}")
    axes[0].set_xlabel("shadow size")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Error")
    axes[1].set_xlabel("shadow size")
    axes[1].set_ylabel("mean uncertainty std")
    axes[1].set_title("Uncertainty")
    axes[2].set_xlabel("shadow size")
    axes[2].set_ylabel("95% coverage")
    axes[2].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Calibration")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "06_error_uncertainty_calibration_trends", cfg.plot_formats)

    inspect_shadow_size = cfg.inspect_shadow_size or cfg.shadow_size_grid[-1]
    inspect_num_time_indices = cfg.inspect_num_time_indices or cfg.num_time_index_grid[-1]
    inspect_key = (inspect_shadow_size, inspect_num_time_indices, cfg.inspect_repeat_id)
    if inspect_key not in curve_records:
        print(f"Skipping inspect plot because {inspect_key} is not present in curve records.")
        return

    record = curve_records[inspect_key]
    pred_mean = record["pred_mean"]
    pred_std = record["pred_std"]
    measurement_df_inspect = record["measurement_df"]

    shadow_estimate_times = []
    standard_shadow_estimates = []
    matched_shadow_estimates = []
    basis_match_fractions = []

    for time, time_df in measurement_df_inspect.groupby("time", sort=True):
        shots = []
        for _shadow_id, shot_df in time_df.groupby("shadow_id", sort=True):
            shot = [None for _ in range(cfg.qubit_num)]
            for row in shot_df.itertuples(index=False):
                shot[int(row.qubit)] = (str(row.pauli), int(row.outcome))
            if all(item is not None for item in shot):
                shots.append(shot)

        matching_shots = [shot for shot in shots if all(shot[q][0] == label for q, label in support)]
        basis_match_fractions.append(len(matching_shots) / len(shots))
        shadow_estimate_times.append(float(time))
        standard_shadow_estimates.append(estimation_mean(create_classical_shadow(shots), operator))
        matched_shadow_estimates.append(
            matched_pauli_string_estimate_from_shots(shots, cfg.observable_string)
        )

    shadow_estimate_times = np.asarray(shadow_estimate_times, dtype=float)
    standard_shadow_estimates = np.asarray(standard_shadow_estimates, dtype=float)
    matched_shadow_estimates = np.asarray(matched_shadow_estimates, dtype=float)
    basis_match_fractions = np.asarray(basis_match_fractions, dtype=float)
    shadow_basis_scale = (3.0 ** len(support)) * float(np.nanmean(basis_match_fractions))
    print(
        "Standard shadow conditional basis scale "
        f"= 3^{len(support)} * match_fraction "
        f"= {shadow_basis_scale:.3f} "
        f"(mean match_fraction={np.nanmean(basis_match_fractions):.3f})"
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(tlist, true_values, "--", linewidth=2, label="theory")
    axes[0].plot(prediction_times, pred_mean, linewidth=2, label="Bayesian mean")
    axes[0].scatter(
        shadow_estimate_times,
        standard_shadow_estimates,
        s=28,
        color="tab:orange",
        alpha=0.85,
        label="standard shadow estimate from measurement_df",
    )
    axes[0].scatter(
        shadow_estimate_times,
        matched_shadow_estimates,
        s=22,
        color="tab:green",
        alpha=0.85,
        label="matched-basis estimate from measurement_df",
    )
    axes[0].fill_between(
        prediction_times,
        pred_mean - COVERAGE_Z_VALUES["95pct"] * pred_std,
        pred_mean + COVERAGE_Z_VALUES["95pct"] * pred_std,
        alpha=0.22,
        label="95% prediction-resampling band",
    )
    axes[0].set_ylabel(r"$\langle O \rangle$")
    axes[0].set_title(
        f"shadow={inspect_shadow_size}, time indices={inspect_num_time_indices}, "
        f"repeat={cfg.inspect_repeat_id}"
    )
    axes[0].legend()

    axes[1].plot(
        prediction_times,
        np.abs(pred_mean - true_on_prediction_grid),
        color="tab:red",
        label="Bayesian absolute error",
    )
    axes[1].plot(prediction_times, pred_std, color="tab:blue", label="prediction std")
    axes[1].scatter(
        shadow_estimate_times,
        np.abs(standard_shadow_estimates - np.interp(shadow_estimate_times, tlist, true_values)),
        s=24,
        color="tab:orange",
        alpha=0.85,
        label="standard shadow absolute error",
    )
    axes[1].scatter(
        shadow_estimate_times,
        np.abs(matched_shadow_estimates - np.interp(shadow_estimate_times, tlist, true_values)),
        s=20,
        color="tab:green",
        alpha=0.85,
        label="matched-basis absolute error",
    )
    axes[1].set_xlabel("time")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "07_inspect_fitted_curve", cfg.plot_formats)


def run(cfg: Config):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.grid": True})

    if max(cfg.num_time_index_grid) > cfg.num_true_time_points:
        raise ValueError(
            "num_true_time_points must be at least the largest value in "
            "num_time_index_grid so each observed time is unique."
        )

    np.random.seed(cfg.base_seed)
    random.seed(cfg.base_seed)
    torch.manual_seed(cfg.base_seed)

    tlist = np.linspace(cfg.time_min, cfg.time_max, cfg.num_true_time_points)
    prediction_times = np.linspace(cfg.time_min, cfg.time_max, cfg.prediction_time_count)
    operator = pauli_string_operator(cfg.observable_string, cfg.qubit_num)
    support = observable_support(cfg.observable_string)
    method = "normal_probability" if len(support) == 1 else "conditional_probability"

    psi0 = build_initial_state(cfg.qubit_num)
    hamiltonian = build_hamiltonian(cfg.dynamics_name, cfg.qubit_num)
    result = qt.mesolve(hamiltonian, psi0, tlist, [])
    true_values = np.real(np.asarray(qt.expect(operator, result.states), dtype=float))
    true_on_prediction_grid = np.interp(prediction_times, tlist, true_values)

    print(f"Observable: {cfg.observable_string}")
    print(f"Method: {method}")
    print(f"Dynamics: {cfg.dynamics_name}")
    print(f"Output directory: {cfg.output_dir.resolve()}")

    rows = []
    curve_records = {}
    for num_time_indices_current in cfg.num_time_index_grid:
        for repeat_id in range(cfg.num_dataset_repeats):
            observed_indices = np.linspace(
                0,
                len(tlist) - 1,
                num_time_indices_current,
                dtype=int,
            )
            if method == "normal_probability":
                if cfg.independent_seeds:
                    # Fully-independent draws: each shadow_size gets its own
                    # measurement set AND its own torch seed (no nested prefixes).
                    for shadow_size_current in cfg.shadow_size_grid:
                        seed = make_cell_seed(
                            cfg.base_seed,
                            cfg.observable_string,
                            shadow_size_current,
                            num_time_indices_current,
                            repeat_id,
                        )
                        torch.manual_seed(seed)
                        measurement_df = generate_measurement_df(
                            states=result.states,
                            t_grid=tlist,
                            observed_indices=observed_indices,
                            n=cfg.qubit_num,
                            shadow_size=shadow_size_current,
                            seed=seed,
                        )
                        all_probs = fit_single_qubit_per_shadow_probabilities(
                            measurement_df,
                            prediction_times,
                            cfg,
                            shadow_size_current,
                            support,
                        )
                        sample_matrix = sample_normal_probability_estimates(
                            all_probs,
                            measurement_df,
                            support,
                            cfg,
                        )
                        metrics, pred_mean, pred_std = summarize_prediction_samples(
                            sample_matrix,
                            true_on_prediction_grid,
                        )
                        metrics.update(
                            {
                                "shadow_size": shadow_size_current,
                                "num_time_indices": num_time_indices_current,
                                "repeat_id": repeat_id,
                                "prediction_resamples": cfg.prediction_resamples,
                            }
                        )
                        rows.append(metrics)
                        curve_records[(shadow_size_current, num_time_indices_current, repeat_id)] = {
                            "pred_mean": pred_mean,
                            "pred_std": pred_std,
                            "samples": sample_matrix,
                            "measurement_df": measurement_df.copy(),
                        }
                    continue
                max_shadow_size = max(cfg.shadow_size_grid)
                seed = cfg.base_seed + 100000 * repeat_id + num_time_indices_current
                print(
                    f"time_indices={num_time_indices_current}, repeat={repeat_id}, "
                    f"fitting {max_shadow_size} nested shadow models"
                )
                measurement_df = generate_measurement_df(
                    states=result.states,
                    t_grid=tlist,
                    observed_indices=observed_indices,
                    n=cfg.qubit_num,
                    shadow_size=max_shadow_size,
                    seed=seed,
                )
                start = perf_counter()
                all_probs = fit_single_qubit_per_shadow_probabilities(
                    measurement_df,
                    prediction_times,
                    cfg,
                    max_shadow_size,
                    support,
                )
                print(f"Model fitting completed in {perf_counter() - start:.1f} s")

                for shadow_size_current in cfg.shadow_size_grid:
                    start = perf_counter()
                    sample_matrix = sample_normal_probability_estimates(
                        all_probs[:, :shadow_size_current, :],
                        measurement_df,
                        support,
                        cfg,
                    )
                    metrics, pred_mean, pred_std = summarize_prediction_samples(
                        sample_matrix,
                        true_on_prediction_grid,
                    )
                    metrics.update(
                        {
                            "shadow_size": shadow_size_current,
                            "num_time_indices": num_time_indices_current,
                            "repeat_id": repeat_id,
                            "prediction_resamples": cfg.prediction_resamples,
                        }
                    )
                    rows.append(metrics)
                    curve_records[(shadow_size_current, num_time_indices_current, repeat_id)] = {
                        "pred_mean": pred_mean,
                        "pred_std": pred_std,
                        "samples": sample_matrix,
                        "measurement_df": measurement_df[
                            measurement_df["shadow_id"] < shadow_size_current
                        ].copy(),
                    }
                    print(
                        f"shadow_size={shadow_size_current} reconstruction completed in "
                        f"{perf_counter() - start:.1f} s"
                    )
            else:
                for shadow_size_current in cfg.shadow_size_grid:
                    if cfg.independent_seeds:
                        seed = make_cell_seed(
                            cfg.base_seed,
                            cfg.observable_string,
                            shadow_size_current,
                            num_time_indices_current,
                            repeat_id,
                        )
                        torch.manual_seed(seed)
                    else:
                        seed = (
                            cfg.base_seed
                            + 100000 * repeat_id
                            + 1000 * shadow_size_current
                            + num_time_indices_current
                        )
                    print(
                        f"shadow_size={shadow_size_current}, "
                        f"time_indices={num_time_indices_current}, repeat={repeat_id}"
                    )
                    measurement_df = generate_measurement_df(
                        states=result.states,
                        t_grid=tlist,
                        observed_indices=observed_indices,
                        n=cfg.qubit_num,
                        shadow_size=shadow_size_current,
                        seed=seed,
                    )
                    sample_matrix = run_conditional_model(
                        measurement_df,
                        prediction_times,
                        cfg,
                        shadow_size_current,
                        operator,
                    )
                    metrics, pred_mean, pred_std = summarize_prediction_samples(
                        sample_matrix,
                        true_on_prediction_grid,
                    )
                    metrics.update(
                        {
                            "shadow_size": shadow_size_current,
                            "num_time_indices": num_time_indices_current,
                            "repeat_id": repeat_id,
                            "prediction_resamples": cfg.prediction_resamples,
                        }
                    )
                    rows.append(metrics)
                    curve_records[(shadow_size_current, num_time_indices_current, repeat_id)] = {
                        "pred_mean": pred_mean,
                        "pred_std": pred_std,
                        "samples": sample_matrix,
                        "measurement_df": measurement_df.copy(),
                    }

    results_df = pd.DataFrame(rows)
    metric_cols = [
        "mae",
        "rmse",
        "max_abs_error",
        "mean_uncertainty_std",
        "coverage_1sigma",
        "coverage_95pct",
        "mean_width_1sigma",
        "mean_width_95pct",
    ]
    summary_df = (
        results_df.groupby(["shadow_size", "num_time_indices"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    results_df.to_csv(cfg.output_dir / "synthetic_error_results.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "synthetic_error_summary.csv", index=False)

    serializable_cfg = asdict(cfg)
    serializable_cfg["output_dir"] = str(cfg.output_dir)
    with (cfg.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_cfg, f, indent=2)

    for key, record in curve_records.items():
        shadow_size, num_time_indices, repeat_id = key
        np.savez_compressed(
            cfg.output_dir / f"curve_shadow{shadow_size}_time{num_time_indices}_repeat{repeat_id}.npz",
            prediction_times=prediction_times,
            true_on_prediction_grid=true_on_prediction_grid,
            pred_mean=record["pred_mean"],
            pred_std=record["pred_std"],
            samples=record["samples"],
        )

    save_all_plots(
        cfg,
        cfg.output_dir,
        results_df,
        curve_records,
        tlist,
        true_values,
        prediction_times,
        true_on_prediction_grid,
        operator,
        support,
    )
    print("Finished synthetic error/uncertainty check.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=Config.base_seed)
    parser.add_argument("--qubit-num", type=int, default=Config.qubit_num)
    parser.add_argument("--observable-string", default=Config.observable_string)
    parser.add_argument("--dynamics-name", default=Config.dynamics_name)
    parser.add_argument("--num-true-time-points", type=int, default=Config.num_true_time_points)
    parser.add_argument("--prediction-time-count", type=int, default=Config.prediction_time_count)
    parser.add_argument("--time-min", type=float, default=Config.time_min)
    parser.add_argument("--time-max", type=float, default=Config.time_max)
    parser.add_argument("--shadow-size-grid", type=parse_int_grid, default=Config.shadow_size_grid)
    parser.add_argument("--num-time-index-grid", type=parse_int_grid, default=Config.num_time_index_grid)
    parser.add_argument("--num-dataset-repeats", type=int, default=Config.num_dataset_repeats)
    parser.add_argument("--prediction-resamples", type=int, default=Config.prediction_resamples)
    parser.add_argument("--gp-kernel", choices=["matern32", "rbf"], default=Config.gp_kernel)
    parser.add_argument("--normal-training-iter", type=int, default=Config.normal_training_iter)
    parser.add_argument("--conditional-training-iter", type=int, default=Config.conditional_training_iter)
    parser.add_argument("--num-inducing", type=int, default=Config.num_inducing)
    parser.add_argument("--gp-lr", type=float, default=Config.gp_lr)
    parser.add_argument("--matched-basis-estimator", action="store_true")
    parser.add_argument(
        "--independent-seeds",
        action="store_true",
        help="Draw an independent measurement set + torch seed per "
        "(observable, shadow_size, num_time_indices, repeat) cell instead of the "
        "nested-prefix design (default off preserves existing behaviour).",
    )
    parser.add_argument("--inspect-shadow-size", type=int, default=None)
    parser.add_argument("--inspect-num-time-indices", type=int, default=None)
    parser.add_argument("--inspect-repeat-id", type=int, default=Config.inspect_repeat_id)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--plot-formats", type=parse_plot_formats, default=Config.plot_formats)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    cfg = Config(
        base_seed=args.base_seed,
        qubit_num=args.qubit_num,
        observable_string=args.observable_string,
        dynamics_name=args.dynamics_name,
        num_true_time_points=args.num_true_time_points,
        prediction_time_count=args.prediction_time_count,
        time_min=args.time_min,
        time_max=args.time_max,
        shadow_size_grid=args.shadow_size_grid,
        num_time_index_grid=args.num_time_index_grid,
        num_dataset_repeats=args.num_dataset_repeats,
        prediction_resamples=args.prediction_resamples,
        gp_kernel=args.gp_kernel,
        normal_training_iter=args.normal_training_iter,
        conditional_training_iter=args.conditional_training_iter,
        num_inducing=args.num_inducing,
        gp_lr=args.gp_lr,
        use_matched_basis_estimator_for_pauli_strings=args.matched_basis_estimator,
        independent_seeds=args.independent_seeds,
        inspect_shadow_size=args.inspect_shadow_size,
        inspect_num_time_indices=args.inspect_num_time_indices,
        inspect_repeat_id=args.inspect_repeat_id,
        output_dir=args.output_dir,
        plot_formats=args.plot_formats,
    )
    run(cfg)


if __name__ == "__main__":
    main()
