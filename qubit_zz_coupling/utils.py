import numpy as np
from typing import Union

def make_population(expect: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

def ramsey(t: np.ndarray, A: float, T2_star: float, f: float, phi: float, C: float) -> np.ndarray:
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
    return A * np.exp(-t/T2_star) * np.cos(2*np.pi*f*t + phi) + C