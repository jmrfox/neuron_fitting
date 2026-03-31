# AdEx LIF model fit to imc/ipc projecting OT neurons
# Fitted to FI curve from Goddard 2014 using Nevergrad

import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adexlif_model import *
from typing import Dict, Optional, List, Callable
import numpy as np
import nevergrad as ng

neuron_type = "imc"
parameter_bank = default_parameter_bank(neuron_type, array_mode=True)
data = get_goddard_ficurve_data(n_thin=2)
data_currents = data[neuron_type]["current"]
data_frequencies = data[neuron_type]["frequency"]


def compute_fi_and_metrics(
    params: ParameterSet,
    current_amplitudes: np.ndarray,
    metrics: Optional[Dict[str, bool]] = None,
) -> Dict[str, np.ndarray | float | bool]:
    """Run F-I simulation and compute metrics aligned with emcee objectives.

    Returns a dict containing:
    - rates_inside_stim: (n,)
    - rates_outside_stim: (n,)
    - spikes_outside_stim: (n,) per-current counts
    - spikes_outside_stim_total: float
    - potential_means: (n,)
    - first_spike_delays: (n,) delay w.r.t. stimulus start (>=0), 0 if no spike or before delay
    - latency_mean: float mean of positive delays (NaN if none)
    - success: bool
    """
    if metrics is None:
        metrics = {}

    exp = AdExExperiment(params)
    res = exp.f_i_curve(current_amplitudes)

    # First spike delay per current (following emcee_* pattern)
    delay = params["delay"]
    first_spike_delays = np.zeros(len(current_amplitudes), dtype=float)
    for i, spike_times in enumerate(res["spike_times"]):
        if len(spike_times) > 0:
            first_spike = spike_times[0]
            if first_spike > delay:
                first_spike_delays[i] = first_spike - delay

    # Aggregate outputs
    out: Dict[str, np.ndarray | float | bool] = {
        "rates_inside_stim": res["rates_inside_stim"],
        "rates_outside_stim": res["rates_outside_stim"],
        "spikes_outside_stim": res["spikes_outside_stim"],
        "spikes_outside_stim_total": float(np.sum(res["spikes_outside_stim"])),
        "potential_means": res["potential_means"],
        "first_spike_delays": first_spike_delays,
        "success": bool(res["success"]),
    }

    # Latency: mean over positive entries
    if metrics.get("latency", True):
        pos = first_spike_delays[first_spike_delays > 0]
        out["latency_mean"] = float(np.mean(pos)) if len(pos) > 0 else float("nan")
    else:
        out["latency_mean"] = float("nan")

    return out


DEFAULT_OBJECTIVE_ORDER: List[str] = [
    "fi_mse",
    "spikes_outside_penalty",
    "pot_mean_mse",
    "delay_penalty",
]


def _obj_fi_mse(
    fi_metrics: Dict[str, np.ndarray | float | bool],
    params: ParameterSet,
    target_fi: Optional[np.ndarray],
) -> float:
    if target_fi is None:
        raise ValueError("target_fi must be provided for fi_mse objective")
    sim = np.asarray(fi_metrics["rates_inside_stim"])  # type: ignore[index]
    if sim.shape != target_fi.shape:
        raise ValueError(f"target_fi shape {target_fi.shape} != sim shape {sim.shape}")
    return float(np.mean((sim - target_fi) ** 2))


def _obj_spikes_outside_penalty(
    fi_metrics: Dict[str, np.ndarray | float | bool],
    params: ParameterSet,
    target_fi: Optional[np.ndarray],
) -> float:
    spikes_vec = np.asarray(fi_metrics["spikes_outside_stim"])  # type: ignore[index]
    return 100.0 * float(np.sum(spikes_vec**2))


def _obj_pot_mean_mse(
    fi_metrics: Dict[str, np.ndarray | float | bool],
    params: ParameterSet,
    target_fi: Optional[np.ndarray],
) -> float:
    vrest = float(params["Vrest"])  # parameter-dependent target
    pot_means = np.asarray(fi_metrics["potential_means"])  # type: ignore[index]
    return float(np.sum((pot_means - vrest) ** 2))


def _obj_delay_penalty(
    fi_metrics: Dict[str, np.ndarray | float | bool],
    params: ParameterSet,
    target_fi: Optional[np.ndarray],
) -> float:
    delays = np.asarray(fi_metrics["first_spike_delays"])  # type: ignore[index]
    return 0.2 * float(np.sum(delays**2))


OBJECTIVE_REGISTRY: Dict[
    str,
    Callable[
        [Dict[str, np.ndarray | float | bool], ParameterSet, Optional[np.ndarray]],
        float,
    ],
] = {
    "fi_mse": _obj_fi_mse,
    "spikes_outside_penalty": _obj_spikes_outside_penalty,
    "pot_mean_mse": _obj_pot_mean_mse,
    "delay_penalty": _obj_delay_penalty,
}


def make_objective_vector(
    fi_metrics: Dict[str, np.ndarray | float | bool],
    params: ParameterSet,
    target_fi: Optional[np.ndarray] = None,
    include: Optional[Dict[str, bool]] = None,
    weights: Optional[Dict[str, float]] = None,
    order: Optional[List[str]] = None,
) -> np.ndarray:
    """Compose objectives from a registry with toggles and weights (all minimized).

    - `include`: dict[str,bool] enabling/disabling objectives.
    - `weights`: optional scaling per objective name.
    - `order`: list specifying the order of objectives in the output vector.
    Extend by registering new functions in `OBJECTIVE_REGISTRY` and toggling here.
    """
    if include is None:
        include = {name: True for name in DEFAULT_OBJECTIVE_ORDER}
    if weights is None:
        weights = {}
    if order is None:
        order = [name for name in DEFAULT_OBJECTIVE_ORDER if include.get(name, False)]

    objs: List[float] = []
    for name in order:
        if not include.get(name, False):
            continue
        if name not in OBJECTIVE_REGISTRY:
            raise KeyError(f"Objective '{name}' not registered.")
        val = OBJECTIVE_REGISTRY[name](fi_metrics, params, target_fi)
        objs.append(weights.get(name, 1.0) * float(val))

    return np.array(objs, dtype=float)


def objective_function(parameter_array):
    param_instance = parameter_bank.array_to_instance(parameter_array)
    fi_metrics = compute_fi_and_metrics(param_instance, data_currents)
    return make_objective_vector(fi_metrics, param_instance, data_frequencies)


def get_parameter_count(parameter_bank):
    """Number of sampled parameters, for sanity check in optimizer"""
    return len(parameter_bank.get_default_values(return_array=True))


def create_nevergrad_parameter(bank: ParameterBank) -> ng.p.Array:
    """Create a Nevergrad parameter from ParameterBank bounds.

    Returns:
        Nevergrad Array parameter with bounds from the bank.
    """
    lower = bank.lower_bounds
    upper = bank.upper_bounds

    # Handle both scalar and vector parameters
    if isinstance(lower, np.ndarray):
        # Vector parameters
        lower_bounds = lower
        upper_bounds = upper
    else:
        # Scalar parameters: convert to arrays
        lower_bounds = np.array(lower)
        upper_bounds = np.array(upper)

    # Create Nevergrad Array parameter with bounds
    return ng.p.Array(init=0.5 * (lower_bounds + upper_bounds)).set_bounds(
        lower=lower_bounds, upper=upper_bounds
    )


def multiobjective_function(parameter_array: np.ndarray) -> list:
    """Multi-objective function for Nevergrad.

    Returns a list of objective values (one per objective).
    Returns large penalties if simulation fails.
    """
    try:
        param_instance = parameter_bank.array_to_instance(parameter_array)
        fi_metrics = compute_fi_and_metrics(param_instance, data_currents)

        # Check if simulation succeeded
        if not fi_metrics["success"]:
            # Return large penalties for all objectives
            return [1e10] * len(DEFAULT_OBJECTIVE_ORDER)

        # Get objective vector (returns array of individual objectives)
        obj_vector = make_objective_vector(fi_metrics, param_instance, data_frequencies)
        return obj_vector.tolist()  # Convert to list for Nevergrad

    except Exception as e:
        print(f"Error in objective function: {e}")
        return [1e10] * len(DEFAULT_OBJECTIVE_ORDER)  # Large penalty for errors


def run_optimization(
    budget: int = 1000,
    num_workers: int = 1,
    seed: Optional[int] = None,
    optimizer_name: str = "NgIohTuned",
    **optimizer_kwargs,
):
    """Run Nevergrad optimization with parallel evaluation.

    Args:
        budget: Total number of function evaluations.
        num_workers: Number of parallel workers for concurrent evaluation.
        seed: Random seed for reproducibility.
        optimizer_name: Name of Nevergrad optimizer to use. Options include:
            - 'NgIohTuned': IOH tuned optimizer (default)
            - 'NGOpt': Meta-optimizer that selects best algorithm
            - 'TwoPointsDE': Two-point differential evolution
            - 'CMA': Covariance Matrix Adaptation
            - 'PSO': Particle Swarm Optimization
            - 'OnePlusOne': Simple (1+1) evolution strategy
            - 'DE': Classic differential evolution
        **optimizer_kwargs: Additional keyword arguments for the optimizer.

    Returns:
        Result object with .value (best params) and ._losses (best objectives).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Create parameter space
    param_space = create_nevergrad_parameter(parameter_bank)
    n_params = len(parameter_bank.get_default_values(return_array=True))

    print(f"Starting Nevergrad optimization with {optimizer_name}")
    print(f"Number of parameters: {n_params}")
    print(f"Budget (function evaluations): {budget}")
    print(f"Workers: {num_workers}")
    print(f"Lower bounds: {parameter_bank.lower_bounds}")
    print(f"Upper bounds: {parameter_bank.upper_bounds}")

    # Create optimizer
    optimizer = ng.optimizers.registry[optimizer_name](
        parametrization=param_space, budget=budget, num_workers=num_workers
    )

    if seed is not None:
        np.random.seed(seed)

    # Provide reference upper bounds for multi-objective optimization
    n_objectives = len(DEFAULT_OBJECTIVE_ORDER)
    optimizer.tell(ng.p.MultiobjectiveReference(), [1e4] * n_objectives)

    # Run optimization with ask/tell interface for parallelization
    print(f"\nRunning multi-objective optimization...")
    print(f"Number of objectives: {n_objectives}")
    print(f"Objectives: {DEFAULT_OBJECTIVE_ORDER}")
    print(f"Using parallel evaluation with {num_workers} workers\n")

    if num_workers > 1:
        # Parallel execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            n_evaluated = 0
            while n_evaluated < budget:
                # Ask for candidates
                n_ask = min(num_workers, budget - n_evaluated)
                candidates = [optimizer.ask() for _ in range(n_ask)]

                # Submit jobs to executor
                futures = {
                    executor.submit(multiobjective_function, cand.value): cand
                    for cand in candidates
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    cand = futures[future]
                    try:
                        losses = future.result()
                        optimizer.tell(cand, losses)
                        n_evaluated += 1

                        # Progress update every 10 evaluations
                        if n_evaluated % 10 == 0:
                            print(f"Evaluated {n_evaluated}/{budget} solutions")
                    except Exception as e:
                        print(f"Error evaluating candidate: {e}")
                        optimizer.tell(cand, [1e10] * n_objectives)
                        n_evaluated += 1
    else:
        # Sequential execution (num_workers=1)
        for i in range(budget):
            candidate = optimizer.ask()
            losses = multiobjective_function(candidate.value)
            optimizer.tell(candidate, losses)

            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{budget} solutions")

    print(f"\nOptimization complete!")

    # Get Pareto front
    pareto_front = optimizer.pareto_front()
    print(f"\nPareto front size: {len(pareto_front)}")

    if len(pareto_front) == 0:
        raise RuntimeError(
            "No valid solutions found in Pareto front. All evaluations may have failed."
        )

    # Use best compromise solution from Pareto front (min sum of objectives)
    # This is more reliable than recommendation.value which can be None
    best_solution = min(pareto_front, key=lambda p: sum(p.losses))
    best_value = best_solution.value
    best_losses = list(best_solution.losses)

    print(f"\nBest compromise solution (min sum of objectives):")
    print(f"Objective losses: {best_losses}")
    for i, obj_name in enumerate(DEFAULT_OBJECTIVE_ORDER):
        print(f"  {obj_name}: {best_losses[i]:.6f}")

    # Convert best parameters back to ParameterSet
    best_params = parameter_bank.array_to_instance(best_value)
    print(f"\nBest parameters:")
    print(best_params)

    # Print Pareto front information
    print(f"\nPareto front (sorted by first objective):")
    for i, param in enumerate(sorted(pareto_front, key=lambda p: p.losses[0])[:5]):
        print(f"  Solution {i+1}: losses = {[f'{l:.4f}' for l in param.losses]}")

    # Create a result object with the best solution
    class Result:
        def __init__(self, value, losses):
            self.value = value
            self._losses = losses

    result = Result(best_value, best_losses)

    return result, optimizer


def plot_fi_curve_result(
    result, parameter_bank, data_currents, data_frequencies, filepath=None
):
    """Plot the optimized F-I curve against target data.

    Args:
        result: Nevergrad recommendation object.
        parameter_bank: ParameterBank used for optimization.
        data_currents: Target current values.
        data_frequencies: Target frequency values.
        filepath: Path to save the plot. Default value None does not save.
    """
    import matplotlib.pyplot as plt

    # Get best parameters from recommendation
    best_params = parameter_bank.array_to_instance(result.value)
    # Use stored losses or compute them
    best_losses = getattr(result, "_losses", multiobjective_function(result.value))

    # Run simulation with best parameters
    exp = AdExExperiment(best_params)
    fi_results = exp.f_i_curve(data_currents)

    if not fi_results["success"]:
        print("Warning: Simulation with best parameters failed!")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot target data
    ax.plot(
        data_currents,
        data_frequencies,
        "o",
        color="C1",
        markersize=8,
        label="Target Data",
        alpha=0.7,
    )

    # Plot optimized model
    ax.plot(
        data_currents,
        fi_results["rates_inside_stim"],
        "-",
        color="C0",
        linewidth=2,
        label="Optimized Model",
    )

    ax.set_xlabel("Current Amplitude (pA)", fontsize=12)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    # Create title with all objective values
    obj_str = ", ".join(
        [
            f"{DEFAULT_OBJECTIVE_ORDER[i]}={best_losses[i]:.2f}"
            for i in range(len(best_losses))
        ]
    )
    ax.set_title(f"F-I Curve: Optimized vs Target\n({obj_str})", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)

    return fig, ax


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("AdEx-LIF Parameter Optimization using Nevergrad")
    print("=" * 80)
    print(f"\nNeuron type: {neuron_type}")
    print(f"Number of data points: {len(data_currents)}")
    print(f"Current range: {data_currents.min():.2f} - {data_currents.max():.2f} pA")
    print(
        f"Frequency range: {data_frequencies.min():.2f} - {data_frequencies.max():.2f} Hz"
    )

    # Run optimization
    # For multi-objective: use 'DE', 'TwoPointsDE', or 'NgIohTuned' (delegates to DE for multi-obj)
    # Other optimizers convert multi-objective to single objective via volume
    result, optimizer = run_optimization(
        budget=1000,  # Total function evaluations
        num_workers=10,
        seed=0,
        optimizer_name="NgIohTuned",
    )

    # You can access the result:
    # result.value - best parameter array
    # result.losses - list of objective values (one per objective)
    # optimizer.pareto_front() - all Pareto-optimal solutions

    run_label = f"nevergrad_{neuron_type}"
    result_filepath = Path(
        f"/home/jordan/repos/neuron_fitting/adexlif_ot_ficurve/results/{run_label}.txt"
    )
    plot_filepath = Path(
        f"/home/jordan/repos/neuron_fitting/adexlif_ot_ficurve/results/{run_label}.png"
    )

    # Get best solution from recommendation
    best_losses = getattr(result, "_losses", multiobjective_function(result.value))
    best_x = result.value

    # Save best solution
    with open(result_filepath, "w") as f:
        f.write(f"# Best solution from multi-objective optimization\n")
        f.write(f"# Objectives: {', '.join(DEFAULT_OBJECTIVE_ORDER)}\n")
        f.write(f"# Losses: {best_losses}\n")
        param_set = parameter_bank.array_to_instance(best_x)
        for name, val in param_set.items():
            f.write(f"{name}\t{val}\n")

    # Print results
    print(f"\nBest parameters:")
    print(best_x)
    print(f"\nBest objective losses: {best_losses}")

    # Optionally save Pareto front
    pareto_front = optimizer.pareto_front()
    print(f"\nSaving Pareto front with {len(pareto_front)} solutions...")
    pareto_filepath = Path(
        f"/home/jordan/repos/neuron_fitting/adexlif_ot_ficurve/results/{run_label}_pareto.txt"
    )
    with open(pareto_filepath, "w") as f:
        f.write("# Pareto front solutions\n")
        f.write(f"# Objectives: {', '.join(DEFAULT_OBJECTIVE_ORDER)}\n")
        for i, param in enumerate(pareto_front):
            f.write(f"\n# Solution {i+1}, losses: {param.losses}\n")
            param_set = parameter_bank.array_to_instance(param.value)
            for name, val in param_set.items():
                f.write(f"{name}\t{val}\n")

    # Plot the optimized F-I curve
    print("\nGenerating F-I curve plot...")
    plot_fi_curve_result(
        result, parameter_bank, data_currents, data_frequencies, plot_filepath
    )
