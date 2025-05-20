# Qubit ZZ Coupling Package

This package provides tools for simulating and analyzing two-qubit systems with ZZ coupling and a single transmon-like qubit (TLS). It includes functionality for measuring T1 and T2 relaxation times, as well as utilities for population conversion and plotting results.

## Installation

To install the package, clone the repository and run the following command in the root directory:

```bash
pip install .
```

## Usage

### T1 Measurement

The package includes functions to perform T1 measurements on qubits. You can use the following functions:

- `solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=False)`: Solves the Lindblad master equation for T1 measurement. Set `ret_pop=True` to return the population values.

- `plot_t1(H, psi0, tlist, c_ops, e_ops, plable_Qbit, system_params)`: Plots the T1 measurement results, showing the data and the fitted decay curve.

### Population Conversion

The utility function `make_population(expect)` converts expectation values to population values for the |1⟩ state.

## Testing

Unit tests for the package can be found in the `tests` directory. To run the tests, use the following command:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.