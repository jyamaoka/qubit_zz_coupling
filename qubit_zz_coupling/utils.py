import numpy as np
from typing import Union


def make_population(
    expect: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert expectation value <σ> to |1⟩ population.
    
    Args:
        expect: Expectation value(s) of a Pauli operator.
    Returns:
        Population(s) in the |1⟩ state.
    """
    pop = (1 + expect) / 2
    return pop


def exp_decay(t: np.ndarray, a: float, T1: float, c: float) -> np.ndarray:
    """
    Exponential decay function for T1 relaxation.
    
    Args:
        t: Time array.
        a: Amplitude.
        T1: Relaxation time.
        c: Offset.
    Returns:
        Decayed signal as a numpy array.
    """
    return a * np.exp(-t / T1) + c


def ramsey(
    t: np.ndarray,
    A: float,
    T2_star: float,
    f: float,
    phi: float,
    C: float
) -> np.ndarray:
    """
    Ramsey decay function.
    
    Args:
        t: Time array.
        A: Amplitude.
        T2_star: Coherence time.

        f: Frequency.
        phi: Phase.
        C: Offset.
    Returns:
        Ramsey signal as a numpy array.
    """
    return A * np.exp(-t/T2_star) * np.cos((2*np.pi*f*t) + phi) + C


def fq_shift(
    t: Union[float, np.ndarray],
    f0: float,
    v0: float,
    T: float,
    phi: float = 0,
    jump: bool = False,
    noise_level: float = 0
) -> Union[float, np.ndarray]:
    """
    Calculate a frequency shift as a sinusoidal function of time.

    Args:
        t: Time or array of times.
        f0: Center frequency.
        v0: Amplitude of frequency modulation.
        T: Period of the modulation.
        phi: Phase offset (default 0).
        jump: Make piece wise jumps (default False)
        add_noise: Add noise (default False)
    Returns:
        Frequency shift at time(s) t.
    """
    if jump:
        # Samples of Gaussian noise
        noise = np.random.normal(loc=0, scale=noise_level, size=len(t))

        return f0 - ((1-noise) *
                     v0 * np.sign(np.sin((2 * np.pi * t / T) + phi + .01)))

    return f0 - (v0 * np.sin((2 * np.pi * t / T) + phi))
