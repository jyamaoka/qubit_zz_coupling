from .utils import exp_decay, ramsey, make_population, fq_shift, f2w, parse_drive

from qutip import (basis, tensor, sigmaz, sigmax, sigmam, mesolve, identity,
                   Qobj, expect)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from qtt.algorithms.functions import fit_gauss_ramsey

from typing import Dict, Tuple, List, Any, Union

def make_3tensor(s1: int, s2: int, s3: int) -> Qobj:
    """
    Make 3 Tensor
    Args:
        s1 (int): state q1
        s2 (int): state q2
        s3 (int): state tls

    Return:
        Qobj: tensor state
    """
    return tensor(basis(2,s1), basis(2,s2), basis(2,s3))


def make_n(operator: Qobj) -> Qobj:
    """
    Make n operator

    Args:
        operator (Qobj): Operator

    Returns:
        Qobj: n expectation 
    """
    return operator.dag() * operator


def rwaCoupling(operator0: Qobj, operator1: Qobj) -> Qobj:
    """
    Rotating Wave Approximation Coupling term.

    Args:
        operator0 (Qobj): First lowering operator (e.g., sm_q1).
        operator1 (Qobj): Second lowering operator (e.g., sm_q2 or sm_tls).

    Returns:
        Qobj: The RWA coupling Hamiltonian term 
              (operator0† * operator1 + operator0 * operator1†).
    """
    return operator0.dag() * operator1 + operator0 * operator1.dag()


def setup_operators(
    system_params: Dict[str, Any],
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

    H_Q1 = (2 * np.pi * system_params["f_q1"]) / 2 * sz_q1 # must be positiv (+)
    H_Q2 = (2 * np.pi * system_params["f_q2"]) / 2 * sz_q2
    H_TLS = (2 * np.pi * system_params["f_tls"]) / 2 * sz_tls

    H_ZZ = 2 * np.pi * system_params["J_zz"] * sz_q1 * sz_q2
    H_Q1_TLS = 2 * np.pi * system_params["J_tls"] * sz_q1 * sz_tls

    if system_params["single"] :
        H = H_Q1
    else:
        H = H_Q1 + H_Q2 + H_TLS - H_ZZ - H_Q1_TLS

    c_ops = []
    ver1 = False
    try:
        ver1 = system_params["ver1"]
    except KeyError:
        ver1 = True

    if ver1:
        c_ops = [
            np.sqrt(1/system_params["T1"]["q1"]) * sm_q1,
            np.sqrt(1/system_params["T1"]["q2"]) * sm_q2,
            np.sqrt(1/system_params["T1"]["tls"]) * sm_tls,
            #
            np.sqrt(((1/system_params["T2"]["q1"]) -
                     (1/2/system_params["T1"]["q1"])) / 2) * sz_q1,
            np.sqrt(((1/system_params["T2"]["q2"]) -
                     (1/2/system_params["T1"]["q2"])) / 2) * sz_q2,
            np.sqrt(((1/system_params["T2"]["tls"]) -
                     (1/2/system_params["T1"]["tls"])) / 2) * sz_tls
        ]
    else:
        c_ops = [
            np.sqrt(1/system_params["T1"]["q1"]) * sm_q1,
            np.sqrt(1/system_params["T1"]["q2"]) * sm_q2,
            np.sqrt(1/system_params["T1"]["tls"]) * sm_tls,
            #
            np.sqrt(1/(2*system_params["T2"]["q1"])) * sz_q1,
            np.sqrt(1/(2*system_params["T2"]["q2"])) * sz_q2,
            np.sqrt(1/(2*system_params["T2"]["tls"])) * sz_tls
        ]

    return H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2, sm_q1, sm_q2


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
    ax.plot(tlist, exp_decay(tlist, *fit_par), 'r-',
            label=f'Fit: T1 = {fit_par[1]:.2f} μs')
    ax.set_title(
        f'T1 - {label_Qbit} (J_tls = {system_params["J_tls"]}, \
            J_zz = {system_params["J_zz"]})')
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
    #else:
    #    fit_par, _ = curve_fit(ramsey, tlist, pop, p0=first_guess, \ 
    # bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], \
    # [np.inf, np.inf, np.inf, 10.0, np.inf]))
        
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
    system_params: Dict[str, Any],
    use_fit_gauss: bool=False,
    no_fit: bool=False
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
    fit_par, pop = solve_t2(H, psi0, tlist, c_ops, e_ops,
                            first_guess, ret_pop=True, 
                            use_fit_gauss=use_fit_gauss)

    print(fit_par)
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')

    if no_fit == False:
        over_sample = np.linspace(tlist[0], tlist[-1], 10*len(tlist))
        ax.plot(over_sample, ramsey(over_sample, *fit_par), 'r-',
                label=f'Fit: T2 = {fit_par[1]:.2f} μs, f = {fit_par[2]}')

    ax.set_title(
        f'T2 - {label_Qbit} (J_tls = {system_params["J_tls"]}, \
            J_zz = {system_params["J_zz"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax


# Ramsey expectation value for both qubits driven, TLS not driven
def ramsey_expectation_drive_both(
    tau: float,
    w_d1: float,
    w_d2: float,
    t_pulse: float,
    system_params: Dict[str, Any],
    H0: Qobj,
    c_ops: List[Qobj],
    sz1: Qobj,
    sz2: Qobj,
    sx1: Qobj,
    sx2: Qobj,
    opts: Dict[str, Any] = None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
):
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        tau: Free evolution time
        w_d1: Drive frequency for Q1
        w_d2: Drive frequency for Q2
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2, sz_tls: Pauli-Z operators for Q1, Q2, TLS
        sx1, sx2: Pauli-X operators for Q1, Q2
        sm1, sm2, sm_tls: Lowering operators for Q1, Q2, TLS
        opts: QuTiP solver options

    Returns:
        Tuple of expectation values (sz1, sz2, sz_tls)
    """

    #psi0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))

    # First π/2 pulse on both qubits (TLS not driven)
    H1 = [
        H0,
        [sx1, lambda t_, args: system_params["omega1"] * np.cos(w_d1 * t_)],
        [sx2, lambda t_, args: system_params["omega2"] * np.cos(w_d2 * t_)]
    ]
    res1 = mesolve(H1, psi0, [0, t_pulse], c_ops, e_ops=[], options=opts)
    psi1 = res1.states[-1]

    # Free evolution (no drive)
    res2 = mesolve(H0, psi1, [0, tau], c_ops, e_ops=[], options=opts)
    psi2 = res2.states[-1]

    # Second π/2 pulse on both qubits (TLS not driven)
    H2 = [
        H0,
        [sx1, lambda t, args: system_params["omega1"] * np.cos(w_d1 * t)],
        [sx2, lambda t, args: system_params["omega2"] * np.cos(w_d2 * t)]
    ]
    res3 = mesolve(H2, psi2, [0, t_pulse], c_ops, e_ops=[], options=opts)
    psi_final = res3.states[-1]

    # Measure ⟨sz1⟩, ⟨sz2⟩, ⟨sz_tls⟩
    return expect(sz1, psi_final), expect(sz2, psi_final)

# Ramsey expectation value for both qubits driven, TLS not driven
def ramsey_population_drive_both(
    tau: float,
    w_d1: float,
    w_d2: float,
    t_pulse: float,
    system_params: Dict[str, Any],
    H0: Qobj,
    c_ops: List[Qobj],
    sz1: Qobj,
    sz2: Qobj,
    sx1: Qobj,
    sx2: Qobj,
    opts: Dict[str, Any] = None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
):
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        tau: Free evolution time
        w_d1: Drive frequency for Q1
        w_d2: Drive frequency for Q2
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2: Pauli-Z operators for Q1, Q2
        sx1, sx2: Pauli-X operators for Q1, Q2
        opts: QuTiP solver options

    Returns:
        Tuple of populations (pop_q1, pop_q2)
    """
    sz1_exp, sz2_exp = ramsey_expectation_drive_both(
        tau, w_d1, w_d2, t_pulse, system_params, H0, c_ops, sz1, sz2, sx1, sx2, opts, psi0
    )
    # Convert expectation values to populations in |1>
    pop_q1 = (1 + sz1_exp) / 2
    pop_q2 = (1 + sz2_exp) / 2
    return pop_q1, pop_q2


# Ramsey expectation value for either qubit1 or qubit2 driven, TLS not driven
def ramsey_expectation_drive_sep(
    qubit: int,
    tau: float,
    w_d: Union[float, tuple[float,float]],
    t_pulse: float,
    system_params: Dict[str, Any],
    H0: Qobj,
    c_ops: List[Qobj],
    sz1: Qobj,
    sz2: Qobj,
    sx1: Qobj,
    sx2: Qobj,
    opts: Dict[str, Any]=None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
):
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        qubit: Qubit to drive (1 or 2)
        tau: Free evolution time
        w_d: Drive frequency for Q1
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2, sz_tls: Pauli-Z operators for Q1, Q2, TLS
        sx1, sx2: Pauli-X operators for Q1, Q2
        sm1, sm2, sm_tls: Lowering operators for Q1, Q2, TLS
        opts: QuTiP solver options

    Returns:
        Tuple of expectation values (sz1, sz2, sz_tls)
    """

    sx = sx1
    omega = system_params["omega1"]

    if qubit == 2:
        sx = sx2
        omega = system_params["omega2"]  

    w_d1, w_d2 = parse_drive(w_d)

    # First π/2 pulse on qubits
    H1 = [
        H0,
        [sx, lambda t, args: omega * np.cos(w_d1 * t)]
    ]
    res1 = mesolve(H1, psi0, [0, t_pulse], c_ops, e_ops=[], options=opts)
    psi1 = res1.states[-1]

    # Free evolution (no drive w/ c_ops)
    res2 = mesolve(H0, psi1, [0, tau], c_ops, e_ops=[], options=opts)
    psi2 = res2.states[-1]

    # Second π/2 pulse on qubits
    H2 = [
        H0,
        [sx, lambda t_, args: omega * np.cos(w_d2 * t_)]
    ]
    res3 = mesolve(H2, psi2, [0, t_pulse], c_ops, e_ops=[], options=opts)
    psi_final = res3.states[-1]

    # Measure ⟨sz1⟩, ⟨sz2⟩, ⟨sz_tls⟩
    return expect(sz1, psi_final), expect(sz2, psi_final)

# Ramsey expectation value for either qubit1 or qubit2 driven, TLS not driven
def ramsey_population_drive_sep(
    qubit: int,
    tau: float,
    w_d: Union[float, tuple[float,float]],
    t_pulse: float,
    system_params: Dict[str, Any],
    H0: Qobj,
    c_ops: List[Qobj],
    sz1: Qobj,
    sz2: Qobj,
    sx1: Qobj,
    sx2: Qobj,
    opts: Dict[str, Any]=None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
):
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        qubit: Qubit to drive (1 or 2)
        tau: Free evolution time
        w_d: Drive frequency for Q1
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2: Pauli-Z operators for Q1, Q2, TLS
        sx1, sx2: Pauli-X operators for Q1, Q2
        sm1, sm2, sm_tls: Lowering operators for Q1, Q2, TLS
        opts: QuTiP solver options

    Returns:
        Tuple of populations (pop_q1, pop_q2)
    """
    sz1_exp, sz2_exp = ramsey_expectation_drive_sep(
        qubit, tau, w_d, t_pulse, system_params, H0, c_ops, sz1, sz2, sx1, sx2, opts, psi0
    )
    # Convert expectation values to populations in |1>
    pop_q1 = (1 + sz1_exp) / 2
    pop_q2 = (1 + sz2_exp) / 2
    return pop_q1, pop_q2


def rabi_results(taus, w_d, omega, H0, sx, c_ops=None, opts=None, psi0=None):
    """
    Simulate a Rabi oscillation experiment.

    Args:
        t (float): Pulse duration.
        w_d (float): Drive frequency.
        omega (float): Drive amplitude.
        H0 (Qobj): Bare Hamiltonian.
        sx (Qobj): Pauli-X operator for the driven qubit.
        sz (Qobj): Pauli-Z operator for the driven qubit.
        c_ops (list): List of collapse operators (optional).
        opts (Options): QuTiP solver options (optional).
        psi0 (Qobj): Initial state (optional, default |0>).

    Returns:
        float: Expectation value of sz after the pulse.
    """
    
    # Drive Hamiltonian
    H = [
        H0,
        [sx, lambda t, args: omega * np.cos(w_d * t)]
    ]

    # Evolve under the drive
    result = mesolve(H, psi0, taus, c_ops, e_ops=[], options=opts)

    # Return results
    return result


def rabi_results_multiplex(taus, w_d1, w_d2, omega, H0, sx1, sx2, c_ops=None, opts=None, psi0=None):
    """
    Simulate a Rabi oscillation experiment.

    Args:
        t (float): Pulse duration.
        w_d (float): Drive frequency.
        omega (float): Drive amplitude.
        H0 (Qobj): Bare Hamiltonian.
        sx (Qobj): Pauli-X operator for the driven qubit.
        sz (Qobj): Pauli-Z operator for the driven qubit.
        c_ops (list): List of collapse operators (optional).
        opts (Options): QuTiP solver options (optional).
        psi0 (Qobj): Initial state (optional, default |0>).

    Returns:
        float: Expectation value of sz after the pulse.
    """
    
    # Drive Hamiltonian
    H = [
        H0,
        [sx1, lambda t, args: omega * np.cos(w_d1 * t)],
        [sx2, lambda t, args: omega * np.cos(w_d2 * t)]
    ]

    # Evolve under the drive
    result = mesolve(H, psi0, taus, c_ops, e_ops=[], options=opts)

    # Return results
    return result

# Ramsey expectation value for either qubit1 or qubit2 driven, TLS not driven
def ramsey_result(
    tau: float,
    w_d: Union[float, tuple[float,float]],
    t_pulse: float,
    omega: float,
    H0: Qobj,
    sx: Qobj,
    c_ops: List[Qobj],
    opts: Dict[str, Any]=None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0)),
    step: float=1.0
)-> Qobj:
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        qubit: Qubit to drive (1 or 2)
        tau: Free evolution time
        w_d: Drive frequency for Q1
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2, sz_tls: Pauli-Z operators for Q1, Q2, TLS
        sx1, sx2: Pauli-X operators for Q1, Q2
        sm1, sm2, sm_tls: Lowering operators for Q1, Q2, TLS
        opts: QuTiP solver options

    Returns:
        Qobj of states
    """
    w_d1, w_d2 = parse_drive(w_d)

    # First π/2 pulse on qubits
    step = 2
    pi_vec=np.arange(0,t_pulse+step,step) # ns
    #print(pi_vec)
    #pi_vec = t_pulse

    H1 = [
        H0,
        [sx, lambda t, args: omega * np.cos(w_d1 * t)]
    ]

    res1 = mesolve(H1, psi0, pi_vec, c_ops, e_ops=[], options=opts)
    psi = res1.final_state

    # Free evolution (no drive w/ c_ops)
     # Times for the free oscillation time independant simulation
    step=1 # ns
    t=np.arange(0,tau+step,step) # ns 
  
    # shift to keep the phase consistent
    shift=pi_vec[-1]

    # for T=0 or smaller than step not to throw an error
    if tau>=step:
        #Ramsey free evolution simulation
        res2 = mesolve(H0, psi, (t+shift), c_ops, e_ops=[], options=opts)
        psi = res2.final_state
    shift+=t[-1]
    

    # Second π/2 pulse on qubits
    H2 = [
        H0,
        [sx, lambda t, args: omega * np.cos(w_d2 * t)]
    ]
    res3 = mesolve(H2, psi, (pi_vec+shift), c_ops, e_ops=[], options=opts)
    psi_final = res3.final_state

    # Retrun final state
    return psi_final

# Ramsey expectation value for either qubit1 or qubit2 driven, TLS not driven
def ramsey_result_multiplex(
    tau: float,
    w_d1: Union[float, tuple[float,float]],
    w_d2: Union[float, tuple[float,float]],
    t_pulse: float,
    omega: float,
    H0: Qobj,
    sx_q1: Qobj,
    sx_q2: Qobj,
    c_ops: List[Qobj],
    opts: Dict[str, Any]=None,
    psi0: Qobj = tensor(basis(2, 0), basis(2, 0), basis(2, 0)),
    step: float=1.0
)-> Qobj:
    """
    Ramsey experiment for three qubits (Q1, Q2 driven, TLS not driven).

    Args:
        qubit: Qubit to drive (1 or 2)
        tau: Free evolution time
        w_d: Drive frequency for Q1
        t_pulse: Pulse duration
        system_params: System parameters dictionary
        sz1, sz2, sz_tls: Pauli-Z operators for Q1, Q2, TLS
        sx1, sx2: Pauli-X operators for Q1, Q2
        sm1, sm2, sm_tls: Lowering operators for Q1, Q2, TLS
        opts: QuTiP solver options

    Returns:
        Qobj of states
    """
    w_d11, w_d12 = parse_drive(w_d1)
    w_d21, w_d22 = parse_drive(w_d2)

    # First π/2 pulse on qubits
    step = 2
    pi_vec=np.arange(0,t_pulse+step,step) # ns
    #print(pi_vec)
    #pi_vec = t_pulse

    H1 = [
        H0,
        [sx_q1, lambda t, args: omega * np.cos(w_d11 * t)],
        [sx_q2, lambda t, args: omega * np.cos(w_d21 * t)]

    ]

    res1 = mesolve(H1, psi0, pi_vec, c_ops, e_ops=[], options=opts)
    psi = res1.final_state

    # Free evolution (no drive w/ c_ops)
     # Times for the free oscillation time independant simulation
    step=1 # ns
    t=np.arange(0,tau+step,step) # ns 
  
    # shift to keep the phase consistent
    shift=pi_vec[-1]

    # for T=0 or smaller than step not to throw an error
    if tau>=step:
        #Ramsey free evolution simulation
        res2 = mesolve(H0, psi, (t+shift), c_ops, e_ops=[], options=opts)
        psi = res2.final_state
    shift+=t[-1]
    

    # Second π/2 pulse on qubits
    H2 = [
        H0,
       [sx_q1, lambda t, args: omega * np.cos(w_d11 * t)],
       [sx_q2, lambda t, args: omega * np.cos(w_d21 * t)]
    ]
    res3 = mesolve(H2, psi, (pi_vec+shift), c_ops, e_ops=[], options=opts)
    psi_final = res3.final_state

    # Retrun final state
    return psi_final