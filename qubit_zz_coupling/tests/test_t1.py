import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.t1 import solve_t1, plot_t1
from src.utils import make_population

def test_solve_t1():
    # Example Hamiltonian and parameters for testing
    H = np.array([[0, 1], [1, 0]])  # Dummy Hamiltonian
    psi0 = np.array([1, 0])  # Initial state
    tlist = np.linspace(0, 10, 100)  # Time list
    c_ops = []  # Collapse operators
    e_ops = []  # Measurement operators

    # Call the function
    fit_params = solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=True)

    # Check if the returned parameters are as expected
    assert len(fit_params) > 0, "Fit parameters should not be empty"

def test_plot_t1():
    # Example Hamiltonian and parameters for testing
    H = np.array([[0, 1], [1, 0]])  # Dummy Hamiltonian
    psi0 = np.array([1, 0])  # Initial state
    tlist = np.linspace(0, 10, 100)  # Time list
    c_ops = []  # Collapse operators
    e_ops = []  # Measurement operators
    system_params = {"Jzz": 0.0}  # Example system parameters

    # Call the function
    plot_t1(H, psi0, tlist, c_ops, e_ops, "Qubit 1", system_params)

def test_make_population():
    # Example expectation values
    expect = np.array([-1, 0, 1])

    # Call the function
    pop = make_population(expect)

    # Check if the population values are as expected
    assert np.all(pop >= 0) and np.all(pop <= 1), "Population values should be between 0 and 1"