import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfftfreq, rfft

# ---------- models -------------------------------------------------
def double_cos(
        t_s: np.ndarray,
        A: float,
        f1: float,
        f2: float,
        phi1: float,
        phi2: float,
        T2: float,
        C: float
) -> np.ndarray:
    """
    Computes a sum of two cosine functions with exponential decay envelope.
    The function returns:
        env * (cos(2πf1*t_s + phi1) + cos(2πf2*t_s + phi2)) + C
    where env = A * exp(-t_s / T2).
    Args:
        t_s (np.ndarray): Array of time values.
        A (float): Amplitude of the envelope.
        f1 (float): Frequency of the first cosine (in Hz).
        f2 (float): Frequency of the second cosine (in Hz).
        phi1 (float): Phase offset of the first cosine (in radians).
        phi2 (float): Phase offset of the second cosine (in radians).
        T2 (float): Decay constant for the exponential envelope.
        C (float): Constant offset.
    Returns:
        np.ndarray: The computed double cosine values at each time in t_s.
    """
    env = A * np.exp(-t_s / T2)
    return env * (np.cos(2*np.pi*f1*t_s + phi1) +
              np.cos(2*np.pi*f2*t_s + phi2)) + C

def single_cos(
    t_s: np.ndarray,
    A: float,
    f1: float,
    f2: float,
    phi1: float,
    phi2: float,
    T2: float,
    C: float
) -> np.ndarray:
    """
    Computes a single cosine function with exponential decay envelope.

    Args:
        t_s (np.ndarray): Array of time values.
        A (float): Amplitude of the envelope.
        f1 (float): Frequency of the cosine (in Hz).
        f2 (float): Unused parameter (for API compatibility).
        phi1 (float): Phase offset of the cosine (in radians).
        phi2 (float): Unused parameter (for API compatibility).
        T2 (float): Decay constant for the exponential envelope.
        C (float): Constant offset.

    Returns:
        np.ndarray: The computed single cosine values at each time in t_s.
    """
    env = A * np.exp(-t_s / T2)
    return env * np.cos(2 * np.pi * f1 * t_s + phi1) + C

def two_amp_cos(
    t_s: np.ndarray,
    A1: float,
    A2: float,
    f1: float,
    f2: float,
    phi1: float,
    phi2: float,
    T2: float,
    C: float
) -> np.ndarray:
    """
    Model function: sum of two decaying cosines with independent amplitudes and phases.

    Args:
        t_s (np.ndarray): Time array (seconds).
        A1 (float): Amplitude of the first cosine.
        A2 (float): Amplitude of the second cosine.
        f1 (float): Frequency of the first cosine (Hz).
        f2 (float): Frequency of the second cosine (Hz).
        phi1 (float): Phase of the first cosine (radians).
        phi2 (float): Phase of the second cosine (radians).
        T2 (float): Decay time constant (seconds).
        C (float): Offset.

    Returns:
        np.ndarray: The evaluated model at each time point.
    """
    env = np.exp(-t_s / T2)
    return env * (A1 * np.cos(2 * np.pi * f1 * t_s + phi1) +
                  A2*np.cos(2*np.pi*f2*t_s + phi2)) + C

def two_amp_cos_cns(
    t_s: np.ndarray,
    A1: float,
    f1: float,
    f2: float,
    phi1: float,
    phi2: float,
    T2: float,
    C: float
) -> np.ndarray:
    """
    Model function: sum of two decaying cosines with a constrained amplitude relationship.

    Args:
        t_s (np.ndarray): Time array (seconds).
        A1 (float): Amplitude scaling for the first cosine (second is 1-A1).
        f1 (float): Frequency of the first cosine (Hz).
        f2 (float): Frequency of the second cosine (Hz).
        phi1 (float): Phase of the first cosine (radians).
        phi2 (float): Phase of the second cosine (radians).
        T2 (float): Decay time constant (seconds).
        C (float): Offset.

    Returns:
        np.ndarray: The evaluated model at each time point.
    """
    env = np.exp(-t_s / T2)
    return env * (A1 * np.cos(2 * np.pi * f1 * t_s + phi1) +
                  (1 - A1) * np.cos(2 * np.pi * f2 * t_s + phi2)) + C

def cos_product(
    t_s: np.ndarray,
    A: float,
    f1: float,
    f2: float,
    phi: float,
    T2: float,
    C: float
) -> np.ndarray:
    """
    Model function: product of two cosines with exponential decay and offset.

    Args:
        t_s (np.ndarray): Time array (seconds).
        A (float): Amplitude scaling factor.
        f1 (float): Frequency of the first cosine (Hz).
        f2 (float): Frequency of the second cosine (Hz).
        phi (float): Phase offset for the second cosine (radians).
        T2 (float): Decay time constant (seconds).
        C (float): Offset.

    Returns:
        np.ndarray: The evaluated model at each time point.
    """
    env = np.exp(-t_s / T2)
    return env * A * np.cos(2 * np.pi * f1 * t_s) * np.cos(2 * np.pi * f2 * t_s + phi) + C

# ---------- initial-guess helper (FFT) ---------------------------

def guess_params(
        t_ns: np.ndarray,
        y: np.ndarray
) -> list[float]:
    """
    Estimate initial parameters for fitting a two-frequency oscillatory signal.
    This function analyzes the input signal `y` sampled at times `t_ns` (in nanoseconds)
    and returns a list of initial guesses for parameters typically used in fitting
    a sum of two oscillatory components plus an offset.
    Parameters:
        t_ns (np.ndarray): 1D array of time points in nanoseconds.
        y (np.ndarray): 1D array of signal values corresponding to `t_ns`.
    Returns:
        list[float]: A list containing initial guesses for the following parameters:
            [A0, f1, f2, 0, 0, 20e-6, C0], where:
                - A0: Estimated amplitude.
                - f1: Frequency of the largest spectral peak (Hz).
                - f2: Frequency of the second largest spectral peak (Hz).
                - 0: Placeholder for phase or other parameter.
                - 0: Placeholder for phase or other parameter.
                - 20e-6: Initial guess for a decay or time constant (seconds).
                - C0: Estimated signal offset (mean value).
    """
    t_s = t_ns*1e-9
    C0  = y.mean()
    A0  = 0.5*(y.max()-y.min())
    fs  = rfftfreq(len(t_s), t_s[1]-t_s[0])
    spec= np.abs(rfft(y-C0))
    peak_idx = spec.argsort()[-2:][::-1]   # two largest peaks
    f1, f2 = fs[peak_idx]
    return [A0, f1, f2, 0, 0, 20e-6, C0] # 5e-6

def guess_params_2a(t_ns: np.ndarray, y: np.ndarray) -> list:
    """
    Generate initial parameter guesses for the two_amp_cos model using FFT.

    Args:
        t_ns (np.ndarray): Time array in nanoseconds.
        y (np.ndarray): Signal data.

    Returns:
        list: Initial parameter guesses [A1, A2, f1, f2, phi1, phi2, T2, C0] for two_amp_cos.
    """
    t_s = t_ns * 1e-9
    C0  = y.mean()
    y0  = y - C0
    A_guess = 0.5 * (y.max() - y.min())
    fs   = rfftfreq(len(t_s), t_s[1] - t_s[0])
    spec = np.abs(rfft(y0))
    idx  = spec.argsort()[-2:][::-1]          # two largest peaks
    f1, f2 = fs[idx]
    return [A_guess, A_guess, f1, f2, 0.0, 0.0, 5e-6, C0]

def guess_params_2a_cns(t_ns: np.ndarray, y: np.ndarray) -> list:
    """
    Generate initial parameter guesses for the two_amp_cos_cns model using FFT.

    Args:
        t_ns (np.ndarray): Time array in nanoseconds.
        y (np.ndarray): Signal data.

    Returns:
        list: Initial parameter guesses [A1, f1, f2, phi1, phi2, T2, C0] for two_amp_cos_cns.
    """
    ret = guess_params_2a(t_ns, y)
    return ret[1:]

# --- report -------------------------------------------------------

def report(label, p):
    print(f"{label}:  f1 = {p[1]/1e6:.3f} MHz   f2 = {p[2]/1e6:.3f} MHz   "
          f"T2 = {p[4]*1e6:.1f} µs")
    
def report_2a(label, p):
    print(f"{label}:  f1 = {p[2]/1e6:.3f} MHz   "
          f"f2 = {p[3]/1e6:.3f} MHz   "
          f"A1 = {p[0]:.2f}  A2 = {p[1]:.2f}   "
          f"T2 = {p[6]*1e6:.1f} µs", f"{p[4]} {p[5]}")
    