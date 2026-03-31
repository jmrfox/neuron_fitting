# AdEx LIF model fit to imc/ipc projecting OT neurons
# Fitted to FI curve from Goddard 2014 using scipy differential evolution

import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adexlif_model import *
from typing import Dict, Optional, List, Callable
import numpy as np
from scipy.optimize import differential_evolution

neuron_type = "imc"
parameter_bank = default_parameter_bank(neuron_type)
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


def get_bounds_for_de(bank: ParameterBank) -> list:
    """Extract bounds from ParameterBank for scipy.optimize.differential_evolution.

    Returns:
        List of (lower, upper) tuples for each sampled parameter.
    """
    lower = bank.lower_bounds
    upper = bank.upper_bounds

    # Handle both scalar and vector parameters
    if isinstance(lower, np.ndarray):
        # Vector parameters: create bounds for each element
        bounds = [(lower[i], upper[i]) for i in range(len(lower))]
    else:
        # Scalar parameters: create list of tuples
        bounds = list(zip(lower, upper))

    return bounds


def scalar_objective_function(parameter_array: np.ndarray) -> float:
    """Scalar objective function for differential_evolution.

    Combines the multi-objective vector into a single scalar by summing.
    Returns a large penalty if simulation fails.
    """
    try:
        param_instance = parameter_bank.array_to_instance(parameter_array)
        fi_metrics = compute_fi_and_metrics(param_instance, data_currents)

        # Check if simulation succeeded
        if not fi_metrics["success"]:
            return 1e10  # Large penalty for failed simulations

        # Get objective vector and sum to create scalar
        obj_vector = make_objective_vector(fi_metrics, param_instance, data_frequencies)
        return float(np.sum(obj_vector))

    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e10  # Large penalty for errors


def run_optimization(
    maxiter: int = 100,
    popsize: int = 15,
    seed: Optional[int] = None,
    workers: int = 1,
    polish: bool = True,
    **de_kwargs,
):
    """Run differential evolution optimization.

    Args:
        maxiter: Maximum number of generations.
        popsize: Population size multiplier (total population = popsize * n_params).
        seed: Random seed for reproducibility.
        workers: Number of parallel workers (-1 for all cores).
        polish: Whether to polish the best result with L-BFGS-B.
        **de_kwargs: Additional keyword arguments for differential_evolution.

    Returns:
        OptimizeResult from scipy.optimize.differential_evolution.
    """
    bounds = get_bounds_for_de(parameter_bank)
    n_params = len(bounds)

    print(f"Starting differential evolution optimization")
    print(f"Number of parameters: {n_params}")
    print(f"Population size: {popsize * n_params}")
    print(f"Max iterations: {maxiter}")
    print(f"Workers: {workers}")
    print(f"Bounds: {bounds}")

    result = differential_evolution(
        scalar_objective_function,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        workers=workers,
        polish=polish,
        disp=True,  # Display progress
        **de_kwargs,
    )

    print(f"\nOptimization complete!")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final objective value: {result.fun}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")

    # Convert best parameters back to ParameterSet
    best_params = parameter_bank.array_to_instance(result.x)
    print(f"\nBest parameters:")
    print(best_params)

    return result


def plot_fi_curve_result(
    result, parameter_bank, data_currents, data_frequencies, filepath=None
):
    """Plot the optimized F-I curve against target data.

    Args:
        result: OptimizeResult from differential_evolution.
        parameter_bank: ParameterBank used for optimization.
        data_currents: Target current values.
        data_frequencies: Target frequency values.
        filepath: Path to save the plot. Default value None does not save.
    """
    import matplotlib.pyplot as plt

    # Get best parameters
    best_params = parameter_bank.array_to_instance(result.x)

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
    ax.set_title(
        f"F-I Curve: Optimized vs Target (Objective = {result.fun:.4f})", fontsize=14
    )
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
    print("AdEx-LIF Parameter Optimization using Differential Evolution")
    print("=" * 80)
    print(f"\nNeuron type: {neuron_type}")
    print(f"Number of data points: {len(data_currents)}")
    print(f"Current range: {data_currents.min():.2f} - {data_currents.max():.2f} pA")
    print(
        f"Frequency range: {data_frequencies.min():.2f} - {data_frequencies.max():.2f} Hz"
    )

    # Run optimization
    result = run_optimization(
        maxiter=1,
        popsize=2,
        seed=0,
        workers=-1,
        polish=False,
    )

    # You can access the result:
    # result.x - best parameter array
    # result.fun - best objective value
    # result.success - whether optimization succeeded
    # result.nit - number of iterations
    # result.nfev - number of function evaluations

    run_label = f"scipy_de_{neuron_type}"
    result_filepath = Path(
        f"/home/jordan/repos/neuron_fitting/adexlif_ot_ficurve/results/{run_label}.txt"
    )
    plot_filepath = Path(
        f"/home/jordan/repos/neuron_fitting/adexlif_ot_ficurve/results/{run_label}.png"
    )
    np.savetxt(result_filepath, parameter_bank.array_to_instance(result.x))

    # Print results
    if result.success:
        print(f"\nOptimization successful!")
    else:
        print(f"\nOptimization failed:\n{result.message}")
    print(f"\nFinal parameters:")
    print(result.x)
    print(f"\nFinal objective value: {result.fun}")
    print(f"\nNumber of iterations: {result.nit}")
    print(f"\nNumber of function evaluations: {result.nfev}")

    # Plot the optimized F-I curve
    print("\nGenerating F-I curve plot...")
    plot_fi_curve_result(
        result, parameter_bank, data_currents, data_frequencies, plot_filepath
    )
