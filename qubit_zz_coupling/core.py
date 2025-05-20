from .utils import exp_decay, ramsey, make_population

from qutip import basis, tensor, sigmaz, sigmax, sigmam, mesolve, identity
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List

def setup_operators(system_params: Dict[str, any]) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
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

    H = H_Q1 + H_Q2 + H_ZZ + H_TLS + H_Q1_TLS + H_XX

    c_ops = [
        np.sqrt(system_params["relaxation"]["q1"]) * sm_q1,
        np.sqrt(system_params["dephasing"]["q1"] / 2) * sz_q1,
        np.sqrt(system_params["relaxation"]["q2"]) * sm_q2,
        np.sqrt(system_params["dephasing"]["q2"] / 2) * sz_q2,
        np.sqrt(system_params["relaxation"]["TLS"]) * sm_tls,
        np.sqrt(system_params["dephasing"]["TLS"] / 2) * sz_tls,
    ]

    return H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2 

def solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=False):
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
    pop = make_population(result.expect[0])
    fit_par, _ = curve_fit(exp_decay, tlist, pop, p0=[1.0, 20, 0])
    return fit_par if not ret_pop else (fit_par, pop)

def plot_t1(H, psi0, tlist, c_ops, e_ops, label_Qbit, system_params):
    fit_par, pop = solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=True)
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')
    ax.plot(tlist, exp_decay(tlist, *fit_par), 'r-', label=f'Fit: T1 = {fit_par[1]:.2f} μs')
    ax.set_title(f'T1 - {label_Qbit} (Jzztls = {system_params["JTLS"]}, Jzz = {system_params["Jzz"]}, Jxx = {system_params["Jxx"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax 

def solve_t2(H, psi0, tlist, c_ops, e_ops, first_guess, ret_pop=False):
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
    pop = make_population(result.expect[0])
    fit_par, _ = curve_fit(ramsey, tlist, pop, p0=first_guess)
    return fit_par if not ret_pop else (fit_par, pop)

def plot_t2(H, psi0, tlist, c_ops, e_ops, first_guess, label_Qbit, system_params):
    fit_par, pop = solve_t2(H, psi0, tlist, c_ops, e_ops, first_guess, ret_pop=True)
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')
    ax.plot(tlist, ramsey(tlist, *fit_par), 'r-', label=f'Fit: T2 = {fit_par[1]:.2f} μs')
    ax.set_title(f'T2 - {label_Qbit} (Jzztls = {system_params["JTLS"]}, Jzz = {system_params["Jzz"]}, Jxx = {system_params["Jxx"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax 