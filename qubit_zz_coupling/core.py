from .utils import exp_decay, ramsey, make_population

from qutip import basis, tensor, sigmaz, sigmax, sigmam, mesolve, identity, Qobj
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from qtt.algorithms.functions import fit_gauss_ramsey

from typing import Dict, Tuple, List, Any, Union

def rwaCoupling(m1, m2):
    """
    coupling
    """
    return m1.dag()*m2 + m1*m2.dag()

def setup_operators(
    system_params: Dict[str, Any]
) -> Tuple[Qobj, List[Qobj], Qobj, Qobj, Qobj, Qobj]:
    """
    Setup Hamiltonian and collapse operators for the system.

    Args:
        system_params: Dictionary containing system parameters.

    Returns:
        Tuple containing:
            - Hamiltonian (Qobj)
            - List of collapse operators (List[Qobj])
            - sz_q1 (Qobj)
            - sz_q2 (Qobj)
            - sx_q1 (Qobj)
            - sx_q2 (Qobj)
    """
    sz_q1 = tensor(sigmaz(), identity(2), identity(2))
    sz_q2 = tensor(identity(2), sigmaz(), identity(2))
    sz_tls = tensor(identity(2), identity(2), sigmaz())

    sx_q1 = tensor(sigmax(), identity(2), identity(2))
    sx_q2 = tensor(identity(2), sigmax(), identity(2))

    sm_q1 = tensor(sigmam(), identity(2), identity(2))
    sm_q2 = tensor(identity(2), sigmam(), identity(2))
    sm_tls = tensor(identity(2), identity(2), sigmam())

    H_Q1 = 2 * np.pi * system_params["fq1"] / 2 * sz_q1
    H_Q2 = 2 * np.pi * system_params["fq2"] / 2 * sz_q2
    H_TLS = 2 * np.pi * system_params["fTLS"] / 2 * sz_tls
    H_ZZ = 2 * np.pi * system_params["Jzz"] * sz_q1 * sz_q2
    H_XX = 2 * np.pi * system_params["Jxx"] * sx_q1 * sx_q2
    H_Q1_TLS = 2 * np.pi * system_params["JTLS"] * sz_q1 * sz_tls

    # xtalk
    H_xtalk = 2 * np.pi * system_params["Jxt"] * rwaCoupling(sz_q1,sz_q2)

    # time dependance 
    #H_t = 

    H = H_Q1 + H_Q2 - H_ZZ + H_TLS + H_Q1_TLS + H_XX + H_xtalk

    c_ops = [
        np.sqrt(system_params["relaxation"]["q1"]) * sm_q1,
        np.sqrt(system_params["dephasing"]["q1"]) * sz_q1,
        np.sqrt(system_params["relaxation"]["q2"]) * sm_q2,
        np.sqrt(system_params["dephasing"]["q2"]) * sz_q2,
        np.sqrt(system_params["relaxation"]["TLS"]) * sm_tls,
        np.sqrt(system_params["dephasing"]["TLS"]) * sz_tls,
    ]

    return H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2 

def solve_t1(
    H: Qobj,
    psi0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj],
    e_ops: List[Qobj],
    ret_pop: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve the T1 relaxation problem and fit the result to an exponential decay.

    Args:
        H: Hamiltonian (Qobj)
        psi0: Initial state (Qobj)
        tlist: Time array (np.ndarray)
        c_ops: List of collapse operators (List[Qobj])
        e_ops: List of measurement operators (List[Qobj])
        ret_pop: If True, also return the population data

    Returns:
        Fit parameters, and optionally the population data
    """
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
    pop = make_population(result.expect[0])
    fit_par, _ = curve_fit(exp_decay, tlist, pop, p0=[1.0, 20, 0])
    return fit_par if not ret_pop else (fit_par, pop)

def plot_t1(
    H: Qobj,
    psi0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj],
    e_ops: List[Qobj],
    label_Qbit: str,
    system_params: Dict[str, Any]
) -> plt.Axes:
    """
    Plot T1 relaxation data and fit.

    Args:
        H: Hamiltonian (Qobj)
        psi0: Initial state (Qobj)
        tlist: Time array (np.ndarray)
        c_ops: List of collapse operators (List[Qobj])
        e_ops: List of measurement operators (List[Qobj])
        label_Qbit: Label for the qubit
        system_params: Dictionary of system parameters

    Returns:
        Matplotlib Axes object
    """
    fit_par, pop = solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=True)
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')
    ax.plot(tlist, exp_decay(tlist, *fit_par), 'r-', label=f'Fit: T1 = {fit_par[1]:.2f} μs')
    ax.set_title(f'T1 - {label_Qbit} (JTLS = {system_params["JTLS"]}, Jzz = {system_params["Jzz"]}, Jxx = {system_params["Jxx"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax 

def solve_t2(
    H: Qobj,
    psi0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj],
    e_ops: List[Qobj],
    first_guess: List[float],
    ret_pop: bool = False,
    use_fit_gauss: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve the T2 (Ramsey) problem and fit the result to the Ramsey decay function.

    Args:
        H: Hamiltonian (Qobj)
        psi0: Initial state (Qobj)
        tlist: Time array (np.ndarray)
        c_ops: List of collapse operators (List[Qobj])
        e_ops: List of measurement operators (List[Qobj])
        first_guess: Initial guess for fit parameters
        ret_pop: If True, also return the population data

    Returns:
        Fit parameters, and optionally the population data
    """
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
    pop = make_population(result.expect[0])

    if use_fit_gauss:
        first_guess, _ = fit_gauss_ramsey(tlist, pop)
        
    fit_par, _ = curve_fit(ramsey, tlist, pop, p0=first_guess)
    return fit_par if not ret_pop else (fit_par, pop)

def plot_t2(
    H: Qobj,
    psi0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj],
    e_ops: List[Qobj],
    first_guess: List[float],
    label_Qbit: str,
    system_params: Dict[str, Any]
) -> plt.Axes:
    """
    Plot T2 (Ramsey) data and fit.

    Args:
        H: Hamiltonian (Qobj)
        psi0: Initial state (Qobj)
        tlist: Time array (np.ndarray)
        c_ops: List of collapse operators (List[Qobj])
        e_ops: List of measurement operators (List[Qobj])
        first_guess: Initial guess for fit parameters
        label_Qbit: Label for the qubit
        system_params: Dictionary of system parameters

    Returns:
        Matplotlib Axes object
    """
    fit_par, pop = solve_t2(H, psi0, tlist, c_ops, e_ops, first_guess, ret_pop=True)
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')
    ax.plot(tlist, ramsey(tlist, *fit_par), 'r-', label=f'Fit: T2 = {fit_par[1]:.2f} μs, f = {fit_par[2]}')
    ax.set_title(f'T2 - {label_Qbit} (JTLS = {system_params["JTLS"]}, Jzz = {system_params["Jzz"]}, Jxx = {system_params["Jxx"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax