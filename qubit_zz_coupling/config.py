import yaml
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

def load_sim_config(path: str) -> Dict[str, Any]:
    """
    Load simulation configuration from a YAML file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    p = Path(path)
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_runtime_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert config dict into runtime objects commonly used in the notebook.

    Returns keys:
      - sample_args (list): positional args for sample_sine_curve (excluding plot kwarg)
      - sample_plot (bool)
      - sample_loop (dict)
      - experiment_params (dict): params suitable for Ramsey_experiment_2qtls
      - t_ns (np.ndarray)
      - save (dict), fitting (dict), plot (dict), postproc (dict)
    """
    # sample_sine_curve args
    ssc = cfg.get("sample_sine_curve", {})
    s_args = ssc.get("args", [])
    sample_plot = ssc.get("plot", False)

    # t_ns
    tcfg = cfg.get("t_ns", {})
    if tcfg:
        start = tcfg.get("start", 0)
        stop = tcfg.get("stop", 100000)
        points = tcfg.get("points", 146)
        t_ns = np.linspace(start, stop, points)
    else:
        t_ns = np.linspace(0, 100000, 146)

    # qubit & tls params -> experiment params
    qubits = cfg.get("qubits", {})
    tls = cfg.get("tls", {})
    experiment = cfg.get("experiment", {})

    params = {
        "fA": qubits.get("A", {}).get("f", 3.44e9),
        "fB": qubits.get("B", {}).get("f", 3.33e9),
        "detA": qubits.get("A", {}).get("det", 3e6),
        "detB": qubits.get("B", {}).get("det", 3e6),
        "T1_A": qubits.get("A", {}).get("T1", 150e-6),
        "T2_A": qubits.get("A", {}).get("T2", 180e-6),
        "T1_B": qubits.get("B", {}).get("T1", 150e-6),
        "T2_B": qubits.get("B", {}).get("T2", 180e-6),
        "t_ns": t_ns,
        "chi_MHz": experiment.get("chi_MHz", 0.01),
        "evo_ver": experiment.get("evo_ver", 2),
        "num_cpus": experiment.get("num_cpus", 1),
        "fT": tls.get("fT", 3.33e9),
        "T1_T": tls.get("T1_T", 200e-6),
        "T2_T": tls.get("T2_T", 210e-6),
        "chi_MHz_T": tls.get("chi_MHz_T", 0.0)
    }

    return {
        "sample_args": s_args,
        "sample_plot": sample_plot,
        "loop": cfg.get("loop", 1),
        "experiment_params": params,
        "name_str": experiment.get("name_str"),
        "t_ns": t_ns,
        "save": cfg.get("save", {}),
        "fitting": cfg.get("fitting", {}),
        "plot": cfg.get("plot", {}),
        "postproc": cfg.get("postproc", {}),
        "raw_cfg": cfg
    }

# Example usage in the notebook:
# cfg = load_sim_config("tls_simulation_config.yml")
# runtime = build_runtime_from_config(cfg)
# sample_args = runtime["sample_args"]        # list of positional args for sample_sine_curve
# sample_plot = runtime["sample_plot"]
# x, y = sample_sine_curve(*sample_args, plot=sample_plot)
# params = runtime["experiment_params"]
# exp = Ramsey_experiment_2qtls(params)
# expA, expB = exp.run()