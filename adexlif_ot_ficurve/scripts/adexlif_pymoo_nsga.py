# AdEx LIF model fit to imc/ipc projecting OT neurons
# Fitted to FI curve from Goddard 2014 using pymoo

import sys
from pathlib import Path
import datetime
import logging
import argparse

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adexlif_model import *
from typing import Dict, Optional, List, Callable
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logger = logging.getLogger(__name__)


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


def get_parameter_count(parameter_bank):
    """Number of sampled parameters, for sanity check in optimizer"""
    return len(parameter_bank.get_default_values(return_array=True))


class AdExProblem(ElementwiseProblem):
    """Pymoo Problem definition for AdEx-LIF parameter optimization.

    Uses ElementwiseProblem for easier parallelization with multiprocessing.
    Parameters are normalized to (0, 1) using MinMaxScaler.
    """

    def __init__(self, parameter_bank, data_currents, data_frequencies, scaler):
        self.parameter_bank = parameter_bank
        self.data_currents = data_currents
        self.data_frequencies = data_frequencies
        self.scaler = scaler

        # Get bounds from parameter bank
        lower = parameter_bank.lower_bounds
        upper = parameter_bank.upper_bounds
        n_params = len(lower)

        # Initialize ElementwiseProblem with normalized bounds (0, 1)
        super().__init__(
            n_var=n_params,
            n_obj=len(DEFAULT_OBJECTIVE_ORDER),
            n_constr=0,
            xl=np.zeros(n_params),  # All parameters normalized to [0, 1]
            xu=np.ones(n_params),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions for a single solution.

        Args:
            x: 1D array containing normalized parameter values (0, 1) for a single solution
            out: Dictionary to store outputs
        """
        try:
            # Inverse transform from (0, 1) to original parameter space
            x_original = self.scaler.inverse_transform(x.reshape(1, -1)).flatten()

            param_instance = self.parameter_bank.array_to_instance(x_original)
            fi_metrics = compute_fi_and_metrics(param_instance, self.data_currents)

            # Check if simulation succeeded
            if not fi_metrics["success"]:
                out["F"] = np.full(self.n_obj, 1e10)
            else:
                obj_vector = make_objective_vector(
                    fi_metrics, param_instance, self.data_frequencies
                )
                out["F"] = obj_vector

        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            out["F"] = np.full(self.n_obj, 1e10)


def run_optimization(
    parameter_bank,
    data_currents,
    data_frequencies,
    scaler,
    n_gen: int = 100,
    pop_size: int = 100,
    seed: Optional[int] = None,
    algorithm: str = "NSGA2",
    n_workers: int = 1,
    **algorithm_kwargs,
):
    """Run pymoo multi-objective optimization.

    Args:
        parameter_bank: ParameterBank instance.
        data_currents: Target current values.
        data_frequencies: Target frequency values.
        scaler: MinMaxScaler for parameter normalization.
        n_gen: Number of generations.
        pop_size: Population size.
        seed: Random seed for reproducibility.
        algorithm: Algorithm to use (NSGA2 or NSGA3).
        n_workers: Number of parallel workers (1 = sequential, >1 = parallel).
        **algorithm_kwargs: Additional keyword arguments for the algorithm.

    Returns:
        Pymoo result object with .X (Pareto front parameters) and .F (Pareto front objectives).
    """
    t0 = datetime.datetime.now()
    # Create problem
    problem = AdExProblem(parameter_bank, data_currents, data_frequencies, scaler)
    n_params = problem.n_var
    n_objectives = problem.n_obj

    logger.info(f"Starting pymoo optimization with {algorithm}")
    logger.info(f"Number of parameters: {n_params}")
    logger.info(f"Number of objectives: {n_objectives}")
    logger.info(f"Objectives: {DEFAULT_OBJECTIVE_ORDER}")
    logger.info(f"Population size: {pop_size}")
    logger.info(f"Generations: {n_gen}")
    logger.info(f"Workers: {n_workers}")
    logger.debug(f"Lower bounds: {parameter_bank.lower_bounds}")
    logger.debug(f"Upper bounds: {parameter_bank.upper_bounds}")

    # Create algorithm
    if algorithm == "NSGA2":
        algo = NSGA2(pop_size=pop_size, **algorithm_kwargs)
    elif algorithm == "NSGA3":
        # NSGA3 requires reference directions
        ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **algorithm_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Set termination criterion
    termination = get_termination("n_gen", n_gen)

    # Setup parallelization if requested
    if n_workers > 1:
        from pymoo.parallelization.joblib import JoblibParallelization

        runner = JoblibParallelization(n_jobs=n_workers)
        problem.elementwise_runner = runner

        logger.info(f"Using JoblibParallelization with {n_workers} workers")
        logger.info("Running multi-objective optimization...")

        res = minimize(
            problem,
            algo,
            termination,
            seed=seed,
            verbose=True,
        )
    else:
        # Sequential execution
        logger.info("Running multi-objective optimization (sequential)...")
        res = minimize(
            problem,
            algo,
            termination,
            seed=seed,
            verbose=True,
        )

    logger.info("Optimization complete!")
    logger.info(f"Pareto front size: {len(res.F)}")

    # Print best solution (closest to ideal point)
    if len(res.F) > 0:
        # Find solution with minimum sum of objectives (compromise solution)
        best_idx = np.argmin(np.sum(res.F, axis=1))
        logger.info("Best compromise solution (min sum of objectives):")
        logger.info(f"Objective losses: {res.F[best_idx]}")
        for i, obj_name in enumerate(DEFAULT_OBJECTIVE_ORDER):
            logger.info(f"  {obj_name}: {res.F[best_idx][i]:.6f}")

        # Inverse transform normalized parameters back to original scale
        best_x_original = scaler.inverse_transform(
            res.X[best_idx].reshape(1, -1)
        ).flatten()

        # Convert best parameters back to ParameterSet
        best_params = parameter_bank.array_to_instance(best_x_original)
        logger.info("Best parameters:")
        logger.info(f"\n{best_params}")

        # Print top 5 solutions from Pareto front
        logger.info("Pareto front (sorted by first objective):")
        sorted_indices = np.argsort(res.F[:, 0])[:5]
        for i, idx in enumerate(sorted_indices):
            logger.info(
                f"  Solution {i+1}: losses = {[f'{l:.4f}' for l in res.F[idx]]}"
            )

    logger.info(f"Time to run: {datetime.datetime.now() - t0}")
    return res


def plot_fi_curve_result(
    result, parameter_bank, scaler, data_currents, data_frequencies, filepath=None
):
    """Plot the optimized F-I curve against target data.

    Args:
        result: Pymoo result object.
        parameter_bank: ParameterBank used for optimization.
        scaler: MinMaxScaler used for parameter normalization.
        data_currents: Target current values.
        data_frequencies: Target frequency values.
        filepath: Path to save the plot. Default value None does not save.
    """
    import matplotlib.pyplot as plt

    # Get best parameters (compromise solution with minimum sum)
    best_idx = np.argmin(np.sum(result.F, axis=1))
    # Inverse transform normalized parameters back to original scale
    best_x_original = scaler.inverse_transform(
        result.X[best_idx].reshape(1, -1)
    ).flatten()
    best_params = parameter_bank.array_to_instance(best_x_original)
    best_losses = result.F[best_idx]

    # Run simulation with best parameters
    exp = AdExExperiment(best_params)
    fi_results = exp.f_i_curve(data_currents)

    if not fi_results["success"]:
        logger.warning("Simulation with best parameters failed!")
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AdEx-LIF Parameter Optimization using pymoo NSGA-II/III",
    )

    # Required arguments
    parser.add_argument(
        "--neuron-type",
        type=str,
        choices=["imc", "ipc"],
        default="ipc",
        help="Neuron type to optimize (default: ipc)",
    )

    # Optimization parameters
    parser.add_argument(
        "--n-gen",
        type=int,
        default=100,
        help="Number of generations (default: 100)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=60,
        help="Population size (default: 60)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=10,
        help="Number of parallel workers, 1 = sequential (default: 10)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["NSGA2", "NSGA3"],
        default="NSGA2",
        help="Algorithm to use (default: NSGA2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--output-label",
        type=str,
        default=None,
        help="Custom label for output files. If not set, uses pymoo_<neuron_type>",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path. If not set, logs to console only",
    )

    return parser.parse_args()


def setup_logging(log_level: str, log_file: Optional[str] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


############################
#           MAIN           #
############################
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("=" * 80)
    logger.info("AdEx-LIF Parameter Optimization using pymoo")
    logger.info("=" * 80)

    # Load data and setup parameter bank
    neuron_type = args.neuron_type
    parameter_bank = default_parameter_bank(neuron_type, array_mode=True)
    data = get_goddard_ficurve_data(n_thin=2)
    data_currents = data[neuron_type]["current"]
    data_frequencies = data[neuron_type]["frequency"]

    # Create scaler to normalize parameters to (0, 1)
    scaler = MinMaxScaler()
    lower_bounds = np.array(parameter_bank.lower_bounds)
    upper_bounds = np.array(parameter_bank.upper_bounds)
    scaler.fit(np.vstack([lower_bounds, upper_bounds]))

    logger.info(f"Neuron type: {neuron_type}")
    logger.info(f"Number of data points: {len(data_currents)}")
    logger.info(
        f"Current range: {data_currents.min():.2f} - {data_currents.max():.2f} pA"
    )
    logger.info(
        f"Frequency range: {data_frequencies.min():.2f} - {data_frequencies.max():.2f} Hz"
    )

    # Run optimization
    result = run_optimization(
        parameter_bank=parameter_bank,
        data_currents=data_currents,
        data_frequencies=data_frequencies,
        scaler=scaler,
        n_gen=args.n_gen,
        pop_size=args.pop_size,
        seed=args.seed,
        algorithm=args.algorithm,
        n_workers=args.n_workers,
    )

    # Determine output label
    run_label = args.output_label if args.output_label else f"pymoo_{neuron_type}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_filepath = output_dir / f"{run_label}.txt"
    plot_filepath = output_dir / f"{run_label}.png"

    # Get best compromise solution
    best_idx = np.argmin(np.sum(result.F, axis=1))
    best_losses = result.F[best_idx]
    best_x_normalized = result.X[best_idx]

    # Inverse transform normalized parameters back to original scale
    best_x = scaler.inverse_transform(best_x_normalized.reshape(1, -1)).flatten()

    # Save best solution
    with open(result_filepath, "w") as f:
        f.write(f"# Best compromise solution from multi-objective optimization\n")
        f.write(f"# Objectives: {', '.join(DEFAULT_OBJECTIVE_ORDER)}\n")
        f.write(f"# Losses: {best_losses}\n")
        param_set = parameter_bank.array_to_instance(best_x)
        for name, val in param_set.items():
            f.write(f"{name}\t{val}\n")

    # Save Pareto front
    logger.info(f"Saving Pareto front with {len(result.F)} solutions...")
    pareto_filepath = output_dir / f"{run_label}_pareto.txt"
    with open(pareto_filepath, "w") as f:
        f.write("# Pareto front solutions\n")
        f.write(f"# Objectives: {', '.join(DEFAULT_OBJECTIVE_ORDER)}\n")
        for i in range(len(result.F)):
            f.write(f"\n# Solution {i+1}, losses: {result.F[i]}\n")
            # Inverse transform normalized parameters back to original scale
            x_original = scaler.inverse_transform(result.X[i].reshape(1, -1)).flatten()
            param_set = parameter_bank.array_to_instance(x_original)
            for name, val in param_set.items():
                f.write(f"{name}\t{val}\n")

    # Plot the optimized F-I curve
    logger.info("Generating F-I curve plot...")
    plot_fi_curve_result(
        result, parameter_bank, scaler, data_currents, data_frequencies, plot_filepath
    )

    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Best solution: {result_filepath}")
    logger.info(f"  - Pareto front: {pareto_filepath}")
    logger.info(f"  - Plot: {plot_filepath}")
    logger.info("Optimization complete!")
