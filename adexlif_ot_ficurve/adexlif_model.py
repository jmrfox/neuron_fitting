import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jscip import *
from collections.abc import Sequence
import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent

ADEXLIF_TEXNAMES = {
    "Vrest": r"$V_{rest}$ (mV)",
    "Vreset": r"$V_{reset}$ (mV)",
    "VTdelta": r"$V_{T\Delta}$ (mV)",  # VT is Vreset + VTdelta
    "Vthres": r"$V_{thres}$ (mV)",
    "VT": r"$V_{T}$ (mV)",
    "EL": r"$E_{L}$ (mV)",
    "Ew": r"$E_{w}$ (mV)",
    "Tref": r"$T_{ref}$ (ms)",
    "tRC": r"$\tau_{RC}$ (ms)",
    "tau_w": r"$\tau_{w}$ (ms)",
    "R": r"$R$ ($\Omega$)",
    "Del": r"$\Delta$ (mV)",
    "a": r"$a$ ($\frac{1}{ms}$)",
    "b": r"$b$ (nA)",
    "I": r"$I$ (pA)",
}


def adexlif_simulation(
    current_input: Sequence[float],  # input current timeseries in pA
    dt: float,  # time step in ms
    Vrest: float,  # resting potential in mV
    Vreset: float,  # reset potential in mV
    VT: float,  # threshold potential in mV
    Vthres: float,  # spike threshold in mV
    EL: float,  # leak reversal potential in mV
    Ew: float,  # reset potential of the adaptation current in mV
    Tref: float,  # refractory period in ms
    tRC: float,  # time constant of the leak in ms
    tau_w: float,  # time constant of the adaptation current in ms
    R: float,  # resistance in MOhm
    Del: float,  # slope factor in mV
    a: float,  # subthreshold adaptation conductance in nS
    b: float,  # spike-triggered adaptation current in pA
    jitter_range: float = 0.0,  # jitter range for spike times in ms
    spike_probability: float = 1.0,  # probability of spike occurrence
    Vcutoff: float = 0.0,  # cutoff value for membrane potential in mV
    **kwargs,  # additional parameters (not used in this function, but allows for flexibility in future use
):

    Vm_allowed_range = (-1000, 1000)  # Range for membrane potential values
    # Numerical integration
    eps = dt / tRC

    # Get population size and number of time steps in input signal
    n_timesteps = len(current_input)

    # Time
    t_domain = np.arange(0, n_timesteps) * dt

    # Intialize
    refractory_timer = 0
    spike_count = 0
    spike_times = []

    V = Vrest

    w = a * (V - Ew)  # nS * mV = pA

    Vm = np.ones(n_timesteps) * Vreset
    Vm[0] = V

    w_out = np.zeros(n_timesteps)
    w_out[0] = w

    # Run the simulation
    for n in range(n_timesteps - 1):

        # Membrane potential update

        exp_arg = (V - VT) / Del
        exp_arg = min(
            exp_arg, 88.7
        )  # Limit the exponent to avoid overflow ( ln(max(float32)) = 88.72) )
        dV = (EL - V) + Del * np.exp(exp_arg) - R * w + R * current_input[n]

        # Adaptation current
        w += dt / tau_w * (a * (V - Ew) - w)
        w_out[n + 1] = w

        # If out of refractory period, update membrane potential
        if refractory_timer <= 0:
            V += eps * dV
            Vm[n + 1] = V

        # Update adaptation current
        if V > Vthres:
            w += b

        # Randomly select spikes to keep
        spike_success = (V > Vthres) & (np.random.uniform() < spike_probability)

        # Record the spike times with jitter added to the spike time
        if spike_success:
            spike_count += 1
            spike_times.append(
                t_domain[n] + np.random.uniform(-0.5, 0.5) * jitter_range
            )

        if V > Vthres:
            # Reset the refactory period counter
            refractory_timer = Tref
            # Reset the membrane potential to the reset value
            V = Vreset

            Vm[n + 1] = Vcutoff

        # Decrease the refactory period counter for those in the refractory period
        if refractory_timer > 0:
            # Decrease the refractory period counter
            refractory_timer -= dt

    # END of time loop. Process the data to return
    if (
        any(np.isnan(Vm))
        or any(np.isinf(Vm))
        or any(Vm < Vm_allowed_range[0])
        or any(Vm > Vm_allowed_range[1])
    ):
        success = False
        # print("WARNING: Vm contains NaN, Inf, or out of bounds values.")
    else:
        success = True

    Vm[Vm > Vcutoff] = Vcutoff

    spike_times = np.array(spike_times)
    output_dict = {
        "t": t_domain,
        "Vm": Vm,
        "w": w_out,
        "spike_times": spike_times,
        "spike_count": spike_count,
        "success": success,
    }
    return output_dict


def get_goddard_ficurve_data(n_thin=1):
    """
    Load the FI curve data from the CSV file and return the IPC and IMC curves.

    Parameters:
    n_thin (int): The thinning factor for the data points.

    Returns:
    tuple: Two numpy arrays containing the IPC and IMC curves.
    """

    goddard_curves = pd.read_csv(
        PROJECT_DIR / "adexlif_ot_ficurve" / "data" / "fig5e_fiplot_datasets.csv",
        skiprows=2,
        names=["ipc_current", "ipc_rate", "imc_current", "imc_rate"],
    )

    # Extract IPC and IMC curves
    ipc_curve = goddard_curves[["ipc_current", "ipc_rate"]].dropna().values
    imc_curve = goddard_curves[["imc_current", "imc_rate"]].dropna().values

    # Prepend IMC curve with zeros up to starting point
    turnon_current = imc_curve[0, 0]
    zero_part = np.array([np.linspace(0, turnon_current * 0.9, 10), np.zeros(10)]).T
    imc_curve = np.vstack((zero_part, imc_curve))

    # Prepend IPC curve with a zero point
    zero_point = np.array([[0, 0]])
    ipc_curve = np.vstack((zero_point, ipc_curve))

    # Thin out data, keeping every n elements
    ipc_curve = ipc_curve[::n_thin]
    imc_curve = imc_curve[::n_thin]

    return {
        "ipc": {"current": ipc_curve[:, 0], "frequency": ipc_curve[:, 1]},
        "imc": {"current": imc_curve[:, 0], "frequency": imc_curve[:, 1]},
    }


def boxcar(
    amplitude: float,
    delay: float,
    total_duration: float,
    dt: float,
):
    """
    Generate a boxcar input signal,
    shaped like this: '_------_'
    Parameters:
        amplitude (float): Amplitude of the boxcar current.
        delay (float): Delay before the current step up, and after the step down.
        total_duration (float): Total simulation time in ms.
        dt (float): Time step in ms.
    Returns:
        np.ndarray: Boxcar current input signal.
    """
    stim_duration = total_duration - 2 * delay
    t = np.arange(0, total_duration + dt, dt)
    x = np.zeros(len(t))
    x[(t > delay) & (t < delay + stim_duration)] = amplitude
    return x


def sloped_boxcar(
    amplitude: float,
    delay: float,
    total_duration: float,
    dt: float,
    slope_time: float,
):
    """
    Generate a sloped boxcar input signal,
    shaped like this: _/-----\\_
    Parameters:
        amplitude (float): Amplitude of the boxcar current.
        delay (float): Delay before the current step up, and after the step down.
        total_duration (float): Total simulation time in ms.
        dt (float): Time step in ms.
        slope_time (float): Duration of the slopes in ms.
    Returns:
        np.ndarray: Sloped boxcar current input signal.
    """
    stim_duration = total_duration - 2 * delay
    time = np.arange(0, total_duration + dt, dt)
    x = np.zeros(len(time))
    for i, t in enumerate(time):
        if delay < t < delay + stim_duration:
            if t < delay + slope_time:
                x[i] = amplitude * (t - delay) / slope_time
            elif t > delay + stim_duration - slope_time:
                x[i] = amplitude * (delay + stim_duration - t) / slope_time
            else:
                x[i] = amplitude
    return x


def best_limits(x, padding=0.3):
    """For a given array x, return the best limits for plotting,
    with a padding factor applied to the range.
    padding: factor to apply to the range of x, default is 0.3"""
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    return (x_min - padding * x_range, x_max + padding * x_range)


def multi_y_plot(
    x_data: dict,
    y_data: dict,
    title: str = "",
    figsize: tuple = None,
    y_styles: dict = None,
    y_limits: dict = None,
    y_alphas: dict = None,
    **kwargs,
):
    """
    Plot multiple datasets with shared x-axis and multiple y-axes positioned horizontally on the left side.
    x_data: dict mapping label to x array
    y_data: dict mapping label to y array
    y_styles: Optional dict mapping label to line style
    y_limits: Optional dict mapping label to (ymin, ymax)
    """

    linewidth = 1.0

    if y_styles is None:
        y_styles = {}
    if y_limits is None:
        y_limits = {}
    if figsize is None:
        figsize = (10, 6)
    if y_alphas is None:
        y_alphas = {}

    y_labels = list(y_data.keys())
    n_series = len(y_labels)
    assert n_series > 0, "y_data must contain at least one series"
    assert len(x_data) == 1, "x_data must contain exactly one series for the x-axis"
    x_label = list(x_data.keys())[0]
    x = x_data[x_label]
    # the same x data is used for all y data, so we check they are the same length
    for label in y_labels:
        assert len(x) == len(
            y_data[label]
        ), f"x_data and y_data for {label} must have the same length"

    # Define expanded color bank
    color_bank = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    fig, ax = plt.subplots(figsize=figsize)
    all_axes = [ax]
    all_lines = []
    all_labels = []

    # Plot the first dataset on the main axis
    first_label = y_labels[0]
    y0 = y_data[first_label]
    color = color_bank[0]
    style = y_styles.get(first_label, "-")
    alpha = y_alphas.get(first_label, 1.0)
    line = ax.plot(
        x,
        y0,
        color=color,
        linewidth=linewidth,
        linestyle=style,
        label=first_label,
        alpha=alpha,
    )[0]
    ax.set_ylabel(first_label, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    if first_label in y_limits:
        ax.set_ylim(*y_limits[first_label])
    else:
        ax.set_ylim(best_limits(y0))
    all_lines.append(line)
    all_labels.append(first_label)

    # Create additional y-axes for remaining datasets
    for i, label in enumerate(y_labels[1:], start=1):
        color = color_bank[i % len(color_bank)]
        style = y_styles.get(label, "-")
        alpha = y_alphas.get(label, 1.0)
        new_ax = ax.twinx()
        new_ax.spines["left"].set_position(("outward", 60 * i))
        new_ax.spines["left"].set_visible(True)
        new_ax.spines["left"].set_color(color)
        new_ax.spines["right"].set_visible(False)
        new_ax.yaxis.set_ticks_position("left")
        new_ax.yaxis.set_label_position("left")
        y_i = y_data[label]
        line = new_ax.plot(
            x,
            y_i,
            color=color,
            linewidth=linewidth,
            linestyle=style,
            label=label,
            alpha=alpha,
        )[0]
        new_ax.set_ylabel(label, color=color)
        new_ax.tick_params(axis="y", labelcolor=color)
        if label in y_limits:
            new_ax.set_ylim(*y_limits[label])
        else:
            new_ax.set_ylim(best_limits(y_i))
        all_axes.append(new_ax)
        all_lines.append(line)
        all_labels.append(label)

    # Set x-axis label (use the first label's x)
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3)

    # Adjust left margin to accommodate multiple y-axes
    n_extra_axes = len(all_axes) - 1
    left_margin = 0.1 + (n_extra_axes * 0.08)
    plt.subplots_adjust(left=left_margin)

    # Create a single legend for all lines
    ax.legend(all_lines, all_labels, loc="upper right")

    # Set title
    if title:
        plt.title(title, fontsize=14)

    plt.show()

    return fig, all_axes


class AdExExperiment:
    """Class to run an AdExLIF simulation with a sloped boxcar input current.
    This class initializes with a parameter dictionary and runs the simulation."""

    def __init__(self, param_dict: ParameterSet):
        self.param_dict = param_dict

    def run(self, current_amplitude: float):
        """Run the AdExLIF simulation with a given current amplitude."""
        current_input = sloped_boxcar(
            current_amplitude,
            delay=self.param_dict["delay"],
            total_duration=self.param_dict["T"],
            dt=self.param_dict["dt"],
            slope_time=self.param_dict["slope_time"],
        )
        sim = adexlif_simulation(current_input, **self.param_dict)
        return sim

    def plot_run(
        self,
        current_amplitude: float,  # current amplitude to test
        show_w: bool = False,  # whether to show the adaptation current
        **kwargs,  # additional parameters passed to the plotting function
    ):
        """Run the AdExLIF simulation for a given current amplitude and plot the results."""

        sim = self.run(current_amplitude)
        if sim["success"] is False:
            print("Simulation failed. Check parameters and input current.")
            return None
        current = sloped_boxcar(
            current_amplitude,
            delay=self.param_dict["delay"],
            total_duration=self.param_dict["T"],
            dt=self.param_dict["dt"],
            slope_time=self.param_dict["slope_time"],
        )
        y_data = {
            "Vm (mV)": sim["Vm"],
            "I (pA)": current,
        }
        y_styles = {
            "Vm (mV)": "-",
            "I (pA)": "--",
        }
        y_alphas = {
            "Vm (mV)": 1.0,
            "I (pA)": 0.7,
        }
        if show_w:
            y_data["w (pA)"] = sim["w"]
            y_styles["w (pA)"] = "-."
            y_alphas["w (pA)"] = 0.7
        result = multi_y_plot(
            x_data={"t (ms)": sim["t"]},
            y_data=y_data,
            title=f"AdExLIF Simulation for Current Amplitude {current_amplitude:0.1f} pA",
            y_styles=y_styles,
            y_alphas=y_alphas,
            **kwargs,  # Pass additional parameters to the plotting function
        )
        return result

    def f_i_curve(
        self,
        current_amplitudes: Sequence[float],  # current amplitudes to test
    ) -> np.ndarray:
        """Run the AdExLIF simulation for a range of current amplitudes and return the results.

        The results include firing rates inside and outside the stimulation period,
        Parameters:
            current_amplitudes (Sequence[float]): Sequence of current amplitudes to test.
        Returns:
            dict: A dictionary containing the results of the simulation, including firing rates and potential means.
        """
        assert len(current_amplitudes.shape) == 1
        results = {
            "Vm_traces": [],
            "w_traces": [],
            "spike_times": [],
            "spikes_inside_stim": [],
            "spikes_outside_stim": [],
            "rates_inside_stim": [],
            "rates_outside_stim": [],
            "potential_means": [],
            "success": True,
        }
        for i in range(len(current_amplitudes)):
            sim = self.run(current_amplitudes[i])
            if sim["success"] is False:
                results["success"] = False
                break
            delay = self.param_dict["delay"]
            T = self.param_dict["T"]
            stim_span = T - 2 * delay
            spikes_inside_stim = len(
                sim["spike_times"][
                    (sim["spike_times"] >= delay) & (sim["spike_times"] <= T - delay)
                ]
            )
            spikes_outside_stim = len(
                sim["spike_times"][
                    (sim["spike_times"] < delay) | (sim["spike_times"] > T - delay)
                ]
            )
            results["spikes_outside_stim"].append(spikes_outside_stim)
            results["spikes_inside_stim"].append(spikes_inside_stim)
            results["rates_inside_stim"].append(1000 * spikes_inside_stim / stim_span)
            results["rates_outside_stim"].append(
                1000 * spikes_outside_stim / (2 * delay)
            )
            results["potential_means"].append(np.mean(sim["Vm"]))
            results["Vm_traces"].append(sim["Vm"])
            results["w_traces"].append(sim["w"])
            results["spike_times"].append(sim["spike_times"])
        results["spikes_outside_stim"] = np.array(results["spikes_outside_stim"])
        results["spikes_inside_stim"] = np.array(results["spikes_inside_stim"])
        results["rates_inside_stim"] = np.array(results["rates_inside_stim"])
        results["rates_outside_stim"] = np.array(results["rates_outside_stim"])
        results["potential_means"] = np.array(results["potential_means"])
        results["Vm_traces"] = np.array(results["Vm_traces"])
        results["w_traces"] = np.array(results["w_traces"])
        # spike_times is not converted to np.array because it is a list of arrays of different lengths
        return results

    def plot_f_i_curve(
        self,
        current_amplitudes: Sequence[float],
        data: Sequence[float] = None,
        title: str = None,
        figsize=(10, 6),
    ):
        """Plot the F-I curve results."""
        fig, ax = plt.subplots(figsize=figsize)
        if data is not None:
            assert len(current_amplitudes) == len(data)
            ax.plot(current_amplitudes, data, c="C1", ls="", marker="o", label="Data")
        results = self.f_i_curve(current_amplitudes)
        if results["success"] is False:
            print(
                "Simulation failed for one or more current amplitudes. Check parameters and input current."
            )
            return None
        ax.plot(
            current_amplitudes,
            results["rates_inside_stim"],
            c="C0",
            ls="-",
            marker=None,
            label="Simulated",
        )
        # if data is not None:
        #     ax.plot(current_amplitudes, data, c="C1", ls="", marker="o", label="Data")
        ax.set_xlabel("Current Amplitude (pA)")
        ax.set_ylabel("Firing Rate (Hz)")
        if title is None:
            title = "F-I Curve"
        ax.set_title(title)
        ax.legend()
        return fig, ax


def default_parameter_bank(neuron_type, array_mode=False):
    if neuron_type is "imc":
        bank = ParameterBank(
            parameters={
                "dt": IndependentScalarParameter(0.04),
                "T": IndependentScalarParameter(600),
                "delay": IndependentScalarParameter(200.0),
                "slope_time": IndependentScalarParameter(2.0),
                "Vthres": IndependentScalarParameter(0.0),
                "Tref": IndependentScalarParameter(0.5),
                "R": IndependentScalarParameter(
                    0.5, is_sampled=True, range=(0.01, 0.7)
                ),
                "tRC": IndependentScalarParameter(5, is_sampled=True, range=(1, 60)),
                "Vrest": IndependentScalarParameter(
                    -65, is_sampled=True, range=(-75, -55)
                ),
                "EL": DerivedScalarParameter(lambda p: p["Vrest"]),
                "Ew": DerivedScalarParameter(lambda p: p["Vrest"]),
                "Vreset": IndependentScalarParameter(
                    -55, is_sampled=True, range=(-60, -45)
                ),
                "VTdelta": IndependentScalarParameter(
                    10, is_sampled=True, range=(-20, 30)
                ),
                "VT": DerivedScalarParameter(lambda p: p["Vreset"] + p["VTdelta"]),
                "Del": IndependentScalarParameter(2, is_sampled=True, range=(0.1, 6)),
                "tau_w": IndependentScalarParameter(
                    30, is_sampled=True, range=(1, 300)
                ),
                "a": IndependentScalarParameter(0, is_sampled=True, range=(-15, 5)),
                "b": IndependentScalarParameter(50, is_sampled=True, range=(0, 130)),
            },
            array_mode=array_mode,
        )
    if neuron_type is "ipc":
        bank = ParameterBank(
            parameters={
                "dt": IndependentScalarParameter(0.04),
                "T": IndependentScalarParameter(600),
                "delay": IndependentScalarParameter(200.0),
                "slope_time": IndependentScalarParameter(2.0),
                "Vthres": IndependentScalarParameter(0.0),
                "Tref": IndependentScalarParameter(0.8),
                "R": IndependentScalarParameter(
                    0.5, is_sampled=True, range=(0.01, 0.7)
                ),
                "tRC": IndependentScalarParameter(5, is_sampled=True, range=(1, 60)),
                "Vrest": IndependentScalarParameter(
                    -65, is_sampled=True, range=(-75, -55)
                ),
                "EL": DerivedScalarParameter(lambda p: p["Vrest"]),
                "Ew": DerivedScalarParameter(lambda p: p["Vrest"]),
                "Vreset": IndependentScalarParameter(
                    -55, is_sampled=True, range=(-60, -45)
                ),
                "VTdelta": IndependentScalarParameter(
                    10, is_sampled=True, range=(0.0, 20)
                ),
                "VT": DerivedScalarParameter(lambda p: p["Vreset"] + p["VTdelta"]),
                "Del": IndependentScalarParameter(2, is_sampled=True, range=(0.1, 6)),
                "tau_w": IndependentScalarParameter(
                    30, is_sampled=True, range=(1, 300)
                ),
                "a": IndependentScalarParameter(0, is_sampled=True, range=(-15, 5)),
                "b": IndependentScalarParameter(50, is_sampled=True, range=(0, 130)),
            },
            constraints=[lambda p: p["a"] * p["R"] * p["tau_w"] < p["tRC"]],
            array_mode=array_mode,
        )
    return bank
