# Qubit ZZ Coupling Package

This package provides tools for simulating and analyzing the dynamics of qubits with ZZ coupling and a single transmon-like TLS (transmon-like superconducting qubit). It includes functionalities for measuring T1 and T2 times, analyzing results, and visualizing data.

## Features

- Simulate qubit dynamics with ZZ coupling.
- Measure T1 and T2 times for qubits.
- Analyze simulation results and extract meaningful parameters.
- Visualize results with various plotting functions.

## Installation

To install the package, clone the repository and run the following command:

```bash
pip install .
```

## Usage

Here is a basic example of how to use the package:

```python
from qubit_zz_coupling import core, analysis, plotting

# Define system parameters
system_params = {
    "fq1": 3.2,
    "fq2": 3.35,
    "fTLS": 3.2,
    "JTLS": 0.05,
    "relaxation": {"q1": 1/20, "q2": 1/25, "TLS": 1/5},
    "dephasing": {"q1": 1/15, "q2": 1/18, "TLS": 1/3},
    "Jzz": 0.1,
    "Jxx": 0.0,
    "time_points": np.linspace(0, 30, 100)
}

# Simulate T1 measurement
t1_results = core.solve_t1(system_params)

# Analyze results
fit_params = analysis.fit_t1(t1_results)

# Plot results
plotting.plot_t1(t1_results, fit_params)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.