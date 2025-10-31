import numpy as np
import qutip as qt

class Ramsey_experiment_2q:
    """
    Simulates a two-qubit Ramsey experiment with decoherence and ZZ coupling.

    Parameters
    ----------
    params : dict
        Dictionary of experiment parameters. Keys include:
            - 'fA', 'fB': Qubit frequencies (Hz)
            - 'detA', 'detB': Detunings (Hz)
            - 'T1_A', 'T1_B': Qubit T1 times (s)
            - 'T2_A', 'T2_B': Qubit T2 times (s)
            - 't_ns': Array of delay times (ns)
            - 'chi_MHz': ZZ coupling strength (MHz)
            - 'evo_ver': Evolution version (0, 1, or 2)
            - 'num_cpus': Number of CPUs for parallelization
    """

    def __init__(self, params: dict):
        self.fA: float = params['fA']
        self.fB: float = params['fB']
        self.detA: float = params['detA']
        self.detB: float = params['detB']
        self.T1_A: float = params['T1_A']
        self.T1_B: float = params['T1_B']
        self.T2_A: float = params['T2_A']
        self.T2_B: float = params['T2_B']
        self.t_ns: np.ndarray = params['t_ns']
        self.chi_MHz: float = params['chi_MHz']
        self.evo_ver: int = params['evo_ver']
        self.num_cpus: int = params.get('num_cpus', 1)

        # Derived quantities
        self.wA, self.wB = 2 * np.pi * np.array([self.fA, self.fB])
        self.chi = 2 * np.pi * self.chi_MHz * 1e6
        self.EdA, self.EdB = 0.011 * self.wA, 0.011 * self.wB
        self.pi2A, self.pi2B = np.pi / (2 * self.EdA), np.pi / (2 * self.EdB)
        self.tau_p = float(max(self.pi2A, self.pi2B))

        # Operators
        I, sx, sz, sm = qt.qeye(2), qt.sigmax(), qt.sigmaz(), qt.destroy(2)
        self.sxA, self.sxB = qt.tensor(sx, I), qt.tensor(I, sx)
        self.szA, self.szB = qt.tensor(sz, I), qt.tensor(I, sz)
        self.smA, self.smB = qt.tensor(sm, I), qt.tensor(I, sm)
        self.nA, self.nB = qt.tensor(qt.num(2), I), qt.tensor(I, qt.num(2))

        # Collapse operators
        #def collapse_ops(self, T1: float, T2: float, sm_op: qt.Qobj, sz_op: qt.Qobj) -> list:
        #    g1 = 1 / T1
        #    gphi = max(0, 1 / T2 - g1 / 2)
        #    return [np.sqrt(g1) * sm_op, np.sqrt(0.5 * gphi) * sz_op]

        #self.c_ops = (self.collapse_ops(self.T1_A, self.T2_A, self.smA, self.szA) +
        #              self.collapse_ops(self.T1_B, self.T2_B, self.smB, self.szB))
        
        # Collapse operators
        def collapse_ops(T1: float, T2: float, sm_op: qt.Qobj, sz_op: qt.Qobj) -> list:
            g1 = 1 / T1
            gphi = max(0, 1 / T2 - g1 / 2)
            return [np.sqrt(g1) * sm_op, np.sqrt(0.5 * gphi) * sz_op]

        self.c_ops: list = (
            collapse_ops(self.T1_A, self.T2_A, self.smA, self.szA)
            + collapse_ops(self.T1_B, self.T2_B, self.smB, self.szB)
        )

        
        # Static Hamiltonian
        self.H0 = 0.5 * self.wA * self.szA + 0.5 * self.wB * self.szB + self.chi / 4 * self.szA * self.szB

        # Initial State
        self.psi_g = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        # MLE Options
        self.opts = {'nsteps': 20000, 'store_final_state': True, 'atol': 1e-8, 'rtol': 1e-7}

    def gate_coeff(self, ω: float, amp: float, t0: float, t1: float):
        """
        Returns a time-dependent coefficient function for a gate pulse.

        Parameters
        ----------
        ω : float
            Drive frequency.
        amp : float
            Amplitude.
        t0 : float
            Start time.
        t1 : float
            End time.

        Returns
        -------
        function
            Coefficient function of time.
        """
        return lambda t, args=None: amp * np.cos(ω * t) if t0 <= t < t1 else 0.0

    def H_sequence_0(self, delay_s: float):
        """
        Returns the Hamiltonian and total time for evolution version 0.

        Parameters
        ----------
        delay_s : float
            Delay time in seconds.

        Returns
        -------
        tuple
            (Hamiltonian, total evolution time)
        """
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        return [self.H0,
                [self.sxA, self.gate_coeff(self.wA + 2 * np.pi * self.detA, self.EdA, t1s, t1e)],
                [self.sxB, self.gate_coeff(self.wB + 2 * np.pi * self.detB, self.EdB, t1s, self.pi2B)],
                [self.sxA, self.gate_coeff(self.wA + 2 * np.pi * self.detA, self.EdA, t2s + delay_s, t2e + delay_s)],
                [self.sxB, self.gate_coeff(self.wB + 2 * np.pi * self.detB, self.EdB, t2s + delay_s, self.pi2B + self.tau_p + delay_s)]], \
               2 * self.tau_p + delay_s

    def H_sequence_1(self, delay_s: float):
        """
        Returns the Hamiltonian and total time for evolution version 1.

        Parameters
        ----------
        delay_s : float
            Delay time in seconds.

        Returns
        -------
        tuple
            (Hamiltonian, total evolution time)
        """
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        wA_detA = self.wA + 2 * np.pi * self.detA
        wB_detB = self.wB + 2 * np.pi * self.detB
        H = [self.H0,
             [self.sxA, f"cos({wA_detA} * t) * (t >= {t1s}) * (t < {t1e}) + cos({wA_detA} * t) * (t >= {t2s} + delay_s) * (t < {t2e} + delay_s)"],
             [self.sxB, f"cos({wB_detB} * t) * (t >= {t1s}) * (t < {self.pi2B}) + cos({wB_detB} * t) * (t >= {t2s} + delay_s) * (t < {self.pi2B + self.tau_p} + delay_s)"]
            ]
        H[1][1] = f"{self.EdA}*({H[1][1]})"
        H[2][1] = f"{self.EdB}*({H[2][1]})"
        return H, 2 * self.tau_p + delay_s

    def ramsey_0(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=0) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        H, t_tot = self.H_sequence_0(tau)
        tlist = np.linspace(0, t_tot, int(t_tot / 2e-9) + 1)
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def ramsey_1(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=1) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        H, t_tot = self.H_sequence_1(tau)
        tlist = np.linspace(0, t_tot, int(t_tot / 2e-9) + 1)
        args = {"delay_s": tau}
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts, args=args).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def ramsey_2(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=2) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        t_tot = 2 * self.tau_p + tau
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        wA_detA = self.wA + 2 * np.pi * self.detA
        wB_detB = self.wB + 2 * np.pi * self.detB
        H = [self.H0,
             [self.sxA, f"{self.EdA}*cos({wA_detA} * t)"],
             [self.sxB, f"{self.EdB}*cos({wB_detB} * t)"]
            ]
        tlist = np.linspace(0, t1e, int(t1e / 2e-9) + 1)
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        t0 = tlist[-1]
        tlist = np.linspace(t0, t0 + tau, int(tau / 2e-9) + 1)
        if len(tlist) > 1:
            st = qt.mesolve(self.H0, st, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        t0 = tlist[-1]
        tlist = np.linspace(t0, t0 + t1e, int(t1e / 2e-9) + 1)
        st = qt.mesolve(H, st, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def run(self) -> tuple:
        """
        Runs the Ramsey experiment for all delay times.

        Returns
        -------
        tuple
            (Array of expectation values for qubit A, Array for qubit B)
        """
        ramsey_funcs = [self.ramsey_0, self.ramsey_1, self.ramsey_2]
        expA, expB = np.array(qt.parallel_map(ramsey_funcs[self.evo_ver], self.t_ns, progress_bar=False, map_kw={"num_cpus": self.num_cpus})).T
        return expA, expB
    

class Ramsey_experiment_2qtls:
    """
    Simulates a two-qubit Ramsey experiment with an additional TLS (two-level system),
    including decoherence and ZZ coupling.

    Parameters
    ----------
    params : dict
        Dictionary of experiment parameters. Keys include:
            - 'fA', 'fB': Qubit frequencies (Hz)
            - 'detA', 'detB': Detunings (Hz)
            - 'T1_A', 'T1_B': Qubit T1 times (s)
            - 'T2_A', 'T2_B': Qubit T2 times (s)
            - 't_ns': Array of delay times (ns)
            - 'chi_MHz': ZZ coupling strength (MHz)
            - 'evo_ver': Evolution version (0, 1, or 2)
            - 'num_cpus': Number of CPUs for parallelization
    """

    def __init__(self, params: dict):
        self.fA: float = params['fA']
        self.fB: float = params['fB']
        self.detA: float = params['detA']
        self.detB: float = params['detB']
        self.T1_A: float = params['T1_A']
        self.T1_B: float = params['T1_B']
        self.T2_A: float = params['T2_A']
        self.T2_B: float = params['T2_B']
        self.t_ns: np.ndarray = params['t_ns']
        self.chi_MHz: float = params['chi_MHz']
        self.evo_ver: int = params['evo_ver']
        self.num_cpus: int = params.get('num_cpus', 1)

        # TLS parameters
        self.fT: float = params['fT']
        self.T1_T: float = params['T1_T']
        self.T2_T: float = params['T2_T']
        self.chi_MHz_T: float = params['chi_MHz_T']

        # Derived quantities
        self.wA, self.wB, self.wT = 2 * np.pi * np.array([self.fA, self.fB, self.fT])
        self.chi: float = 2 * np.pi * self.chi_MHz * 1e6
        self.chi_T: float = 2 * np.pi * self.chi_MHz_T * 1e6
        self.EdA: float = 0.011 * self.wA
        self.EdB: float = 0.011 * self.wB
        self.pi2A: float = np.pi / (2 * self.EdA)
        self.pi2B: float = np.pi / (2 * self.EdB)
        self.tau_p: float = float(max(self.pi2A, self.pi2B))

        # Operators
        I, sx, sz, sm = qt.qeye(2), qt.sigmax(), qt.sigmaz(), qt.destroy(2)
        self.sxA, self.sxB, self.sxT = qt.tensor(sx, I, I), qt.tensor(I, sx, I), qt.tensor(I, I, sx)
        self.szA, self.szB, self.szT = qt.tensor(sz, I, I), qt.tensor(I, sz, I), qt.tensor(I, I, sz)
        self.smA, self.smB, self.smT = qt.tensor(sm, I, I), qt.tensor(I, sm, I), qt.tensor(I, I, sm)
        self.nA, self.nB, self.nT = qt.tensor(qt.num(2), I, I), qt.tensor(I, qt.num(2), I), qt.tensor(I, I, qt.num(2))

        # Collapse operators
        def collapse_ops(T1: float, T2: float, sm_op: qt.Qobj, sz_op: qt.Qobj) -> list:
            g1 = 1 / T1
            gphi = max(0, 1 / T2 - g1 / 2)
            return [np.sqrt(g1) * sm_op, np.sqrt(0.5 * gphi) * sz_op]

        self.c_ops: list = (
            collapse_ops(self.T1_A, self.T2_A, self.smA, self.szA)
            + collapse_ops(self.T1_B, self.T2_B, self.smB, self.szB)
            + collapse_ops(self.T1_T, self.T2_T, self.smT, self.szT)
        )

        # Static Hamiltonian
        self.H0: qt.Qobj = (
            0.5 * self.wA * self.szA
            + 0.5 * self.wB * self.szB
            + self.chi / 4 * self.szA * self.szB
            + 0.5 * self.wT * self.szT
            + self.chi_T / 4 * self.szA * self.szT
        )

        # Initial State
        self.psi_g: qt.Qobj = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
        
        # MLE Options
        self.opts: dict = {'nsteps': 20000, 'store_final_state': True, 'atol': 1e-8, 'rtol': 1e-7}

    def gate_coeff(self, ω: float, amp: float, t0: float, t1: float):
        """
        Returns a time-dependent coefficient function for a gate pulse.

        Parameters
        ----------
        ω : float
            Drive frequency.
        amp : float
            Amplitude.
        t0 : float
            Start time.
        t1 : float
            End time.

        Returns
        -------
        function
            Coefficient function of time.
        """
        return lambda t, args=None: amp * np.cos(ω * t) if t0 <= t < t1 else 0.0

    def H_sequence_0(self, delay_s: float) -> tuple:
        """
        Returns the Hamiltonian and total time for evolution version 0.

        Parameters
        ----------
        delay_s : float
            Delay time in seconds.

        Returns
        -------
        tuple
            (Hamiltonian, total evolution time)
        """
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        return [
            self.H0,
            [self.sxA, self.gate_coeff(self.wA + 2 * np.pi * self.detA, self.EdA, t1s, t1e)],
            [self.sxB, self.gate_coeff(self.wB + 2 * np.pi * self.detB, self.EdB, t1s, self.pi2B)],
            [self.sxA, self.gate_coeff(self.wA + 2 * np.pi * self.detA, self.EdA, t2s + delay_s, t2e + delay_s)],
            [self.sxB, self.gate_coeff(self.wB + 2 * np.pi * self.detB, self.EdB, t2s + delay_s, self.pi2B + self.tau_p + delay_s)]
        ], 2 * self.tau_p + delay_s

    def H_sequence_1(self, delay_s: float) -> tuple:
        """
        Returns the Hamiltonian and total time for evolution version 1.

        Parameters
        ----------
        delay_s : float
            Delay time in seconds.

        Returns
        -------
        tuple
            (Hamiltonian, total evolution time)
        """
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        wA_detA = self.wA + 2 * np.pi * self.detA
        wB_detB = self.wB + 2 * np.pi * self.detB
        H = [
            self.H0,
            [self.sxA, f"cos({wA_detA} * t) * (t >= {t1s}) * (t < {t1e}) + cos({wA_detA} * t) * (t >= {t2s} + delay_s) * (t < {t2e} + delay_s)"],
            [self.sxB, f"cos({wB_detB} * t) * (t >= {t1s}) * (t < {self.pi2B}) + cos({wB_detB} * t) * (t >= {t2s} + delay_s) * (t < {self.pi2B + self.tau_p} + delay_s)"]
        ]
        H[1][1] = f"{self.EdA}*({H[1][1]})"
        H[2][1] = f"{self.EdB}*({H[2][1]})"
        return H, 2 * self.tau_p + delay_s

    def ramsey_0(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=0) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        H, t_tot = self.H_sequence_0(tau)
        tlist = np.linspace(0, t_tot, int(t_tot / 2e-9) + 1)
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def ramsey_1(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=1) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        H, t_tot = self.H_sequence_1(tau)
        tlist = np.linspace(0, t_tot, int(t_tot / 2e-9) + 1)
        args = {"delay_s": tau}
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts, args=args).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def ramsey_2(self, delay_ns: float) -> tuple:
        """
        Simulates Ramsey experiment (evo_ver=2) for a given delay.

        Parameters
        ----------
        delay_ns : float
            Delay in nanoseconds.

        Returns
        -------
        tuple
            (Expectation value for qubit A, Expectation value for qubit B)
        """
        tau = delay_ns * 1e-9
        t_tot = 2 * self.tau_p + tau
        t1s, t1e = 0.0, self.pi2A
        t2s, t2e = self.tau_p, self.tau_p + self.pi2A
        wA_detA = self.wA + 2 * np.pi * self.detA
        wB_detB = self.wB + 2 * np.pi * self.detB
        H = [
            self.H0,
            [self.sxA, f"{self.EdA}*cos({wA_detA} * t)"],
            [self.sxB, f"{self.EdB}*cos({wB_detB} * t)"]
        ]
        tlist = np.linspace(0, t1e, int(t1e / 2e-9) + 1)
        st = qt.mesolve(H, self.psi_g, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        t0 = tlist[-1]
        tlist = np.linspace(t0, t0 + tau, int(tau / 2e-9) + 1)
        if len(tlist) > 1:
            st = qt.mesolve(self.H0, st, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        t0 = tlist[-1]
        tlist = np.linspace(t0, t0 + t1e, int(t1e / 2e-9) + 1)
        st = qt.mesolve(H, st, tlist, self.c_ops, e_ops=[], options=self.opts).states[-1]
        return qt.expect(self.nA, st), qt.expect(self.nB, st)

    def run(self) -> tuple:
        """
        Runs the Ramsey experiment for all delay times.

        Returns
        -------
        tuple
            (Array of expectation values for qubit A, Array for qubit B)
        """
        ramsey_funcs = [self.ramsey_0, self.ramsey_1, self.ramsey_2]
        expA, expB = np.array(qt.parallel_map(
            ramsey_funcs[self.evo_ver],
            self.t_ns,
            progress_bar=False,
            map_kw={"num_cpus": self.num_cpus}
        )).T
        return expA, expB