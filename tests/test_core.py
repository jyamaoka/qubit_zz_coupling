import pytest
from qubit_zz_coupling.core import solve_t1, solve_t2, setup_operators
from qutip import tensor, basis 
import numpy as np


@pytest.fixture
def system_params():
    return {
        "f_q1": 3.2,  # GHz, Qubit 1 frequency (3.2)
        "f_q2": 3.35,  # GHz, Qubit 2 frequency
        "f_tls": 3.2,  # GHz, TLS frequency (resonant with Qubit 1)
        "T1": {"q1": 20000, "q2": 25000, "tls": 1e8},  # (ns) Relaxation rates
        "T2": {"q1": 15000, "q2": 18000, "tls": 1e8},  # Dephasing rates
        "J_zz": 0.01, # Jzz coupling
        "J_tls": 0.01,  # GHz, Coupling strength between Qubit 1 and TLS
        "omega1": 2*np.pi * 0.01, # drive amplitude (sets π/2 duration)
        "omega2": 2*np.pi * 0.01, # drive amplitude (sets π/2 duration)
        "time_points": np.linspace(0, 300, 105),  # Time points in μs
        "ver1": True,
        "single": False
    }


def test_solve_t1(system_params):
    H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2, sm_q1, sm_q2 = setup_operators(system_params)
    tlist = system_params["time_points"]
    psi0 = tensor(basis(2, 0), basis(2, 1), basis(2, 1))

    fit_params = solve_t1(H, psi0, tlist, c_ops, [sz_q1], ret_pop=False)

    assert len(fit_params) == 3
    assert fit_params[1] > 0  # T1 should be positive


def test_solve_t2(system_params):
    H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2, sm_q1, sm_q2 = setup_operators(system_params)
    tlist = system_params["time_points"]
    fq1 = system_params["f_q1"]
    df1 = system_params["T2"]["q1"]
    psi0 = tensor((basis(2, 0) + basis(2, 1)).unit(), basis(2, 1), basis(2, 1))

    fit_params = solve_t2(H, psi0, tlist, c_ops, e_ops=[sx_q1], first_guess=[
                          0.5, 1.0 / df1, fq1, 0.0, 0.5], ret_pop=False)

    assert len(fit_params) == 5
    assert fit_params[1] > 0  # T2 should be positive


def test_setup_operators(system_params):
    H, c_ops, sz_q1, sz_q2, sx_q1, sx_q2, sm_q1, sm_q2 = setup_operators(system_params)

    assert H is not None
    assert len(c_ops) == 6  # Number of collapse operators
    assert sz_q1 is not None
    assert sz_q2 is not None
    assert sx_q1 is not None
    assert sx_q2 is not None
