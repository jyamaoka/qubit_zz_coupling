import numpy as np

def sample_bimodal_gaussian_split(
    mu1: float, sigma1: float,
    mu2: float, sigma2: float,
    size: int = 1,
    weight: float = 0.5
) -> np.ndarray:
    """
    Generate random samples from a bimodal Gaussian (mixture of two Gaussians).

    Args:
        mu1 (float): Mean of the first Gaussian.
        sigma1 (float): Std dev of the first Gaussian.
        mu2 (float): Mean of the second Gaussian.
        sigma2 (float): Std dev of the second Gaussian.
        weight (float): Probability of sampling from the first Gaussian (0 to 1).
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of random samples from the bimodal distribution.
    """
    choices = np.random.rand(size) < weight
    samples_lo = np.empty(np.sum(choices))
    samples_hi = np.empty(np.sum(~choices))

    samples_lo = np.random.normal(mu1, sigma1, np.sum(choices))
    samples_hi = np.random.normal(mu2, sigma2, np.sum(~choices))

    samples_lo[samples_lo < 0] = 0
    samples_hi[samples_hi < 0] = 0

    return samples_lo, samples_hi 

def sample_bimodal_gaussian(
    mu1: float, sigma1: float,
    mu2: float, sigma2: float,
    size: int = 1,
    weight: float = 0.5
) -> np.ndarray:
    """
    Generate random samples from a bimodal Gaussian (mixture of two Gaussians).

    Args:
        mu1 (float): Mean of the first Gaussian.
        sigma1 (float): Std dev of the first Gaussian.
        mu2 (float): Mean of the second Gaussian.
        sigma2 (float): Std dev of the second Gaussian.
        weight (float): Probability of sampling from the first Gaussian (0 to 1).
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of random samples from the bimodal distribution.
    """
    choices = np.random.rand(size) < weight
    samples = np.empty(size)
    samples[choices] = np.random.normal(mu1, sigma1, np.sum(choices))
    samples[~choices] = np.random.normal(mu2, sigma2, np.sum(~choices))
    samples[samples < 0] = 0
    return samples 

def sample_bimodal_gaussian2D(
    mu1: np.array, sigma1: np.array,
    mu2: np.array, sigma2: np.array,
    size: int = 1,
    weight: float = 0.5
) -> np.ndarray:
    """
    Generate random samples from a bimodal Gaussian (mixture of two Gaussians).

    Args:
        mu1 (float): Mean of the first Gaussian.
        sigma1 (float): Std dev of the first Gaussian.
        mu2 (float): Mean of the second Gaussian.
        sigma2 (float): Std dev of the second Gaussian.
        weight (float): Probability of sampling from the first Gaussian (0 to 1).
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of random samples from the bimodal distribution.
    """
    choices = np.random.rand(size) < weight
    samples0 = np.empty(size)
    samples0[choices] = np.random.normal(mu1[0], sigma1[0], np.sum(choices))
    samples0[~choices] = np.random.normal(mu2[0], sigma2[0], np.sum(~choices))
    samples0[samples0 < 0] = 0

    samples1 = np.empty(size)
    samples1[choices] = np.random.normal(mu1[1], sigma1[1], np.sum(choices))
    samples1[~choices] = np.random.normal(mu2[1], sigma2[1], np.sum(~choices))
    samples1[samples1 < 0] = 0
    return samples0, samples1

def correlated_2d_normal(mean, cov, size=10000):
    """
    Generate and plot a correlated 2D normal distribution.

    Args:
        mean (list): Means for each dimension [mean_x, mean_y].
        cov (list of lists): Covariance matrix [[var_x, cov_xy], [cov_yx, var_y]].
        num_samples (int): Number of samples to generate.
        bins (int): Number of bins for the 2D histogram.
        cmap (str): Colormap for the heatmap.

    Returns:
        t1_samples (np.ndarray): Samples for the first variable.
        t2_samples (np.ndarray): Samples for the second variable.
    """
    samples = np.random.multivariate_normal(mean, cov, size)
    t1_samples = samples[:, 0]
    t2_samples = samples[:, 1]

    return t1_samples, t2_samples

def double_correlated_2d_normal(mean1, cov1, mean2, cov2, size=1, weight=0.5):
    """
    Generate and plot two correlated 2D normal distributions.

    Args:
        mean1, mean2: Means for each distribution [mean_x, mean_y].
        cov1, cov2: Covariance matrices for each distribution.
        num_samples: Number of samples for each distribution.
        bins: Number of bins for the 2D histogram.
        cmap: Colormap for the heatmap.

    Returns:
        (t1a, t2a), (t1b, t2b): Samples for both distributions.
    """
    choices = np.random.rand(size) < weight
    t1a, t2a = correlated_2d_normal(mean1, cov1, np.sum(choices))
    t1b, t2b = correlated_2d_normal(mean2, cov2, np.sum(choices))
    return np.concatenate([t1a,t1b]), np.concatenate([t2a,t2b])

def select_periodic(T_Hi, T_Lo, hr, size, time):
    """
    Select random items from T_Hi or T_Lo based on a periodic function with period hr.

    Args:
        T_Hi (np.ndarray): Array of high values.
        T_Lo (np.ndarray): Array of low values.
        hr (float): Period of the function (in same units as index).
        size (int): Number of items to select.
        phase (float): Optional phase offset (default 0).

    Returns:
        np.ndarray: Selected values, length=size.
    """
    tline = np.linspace(0,time,size)
    # Use a sine function to alternate selection
    selector = (np.sin(2 * np.pi * (tline / hr)) > 0)
    # Randomly select from T_Hi or T_Lo
    hi_choices = np.random.choice(T_Hi, size)
    lo_choices = np.random.choice(T_Lo, size)
    result = np.where(selector, hi_choices, lo_choices)
    return result, tline

def select_periodicCorr(T_HiA, T_LoA, T_HiB, T_LoB, hr, size, time):
    """
    Select random items from T_Hi or T_Lo based on a periodic function with period hr.

    Args:
        T_Hi (np.ndarray): Array of high values.
        T_Lo (np.ndarray): Array of low values.
        hr (float): Period of the function (in same units as index).
        size (int): Number of items to select.
        phase (float): Optional phase offset (default 0).

    Returns:
        np.ndarray: Selected values, length=size.
    """
    tline = np.linspace(0,time,size)
    # Use a sine function to alternate selection
    selector = (np.sin(2 * np.pi * (tline / hr)) > 0)
    # Randomly select from T_Hi or T_Lo
    #T_Lo = list(zip(T_LoA,T_LoB))
    #T_Hi = list(zip(T_HiA,T_HiB))
    #hi_choices = np.random.choice(T_Hi, size)
    #lo_choices = np.random.choice(T_Lo, size)

    T_Lo = np.column_stack((T_LoA, T_LoB))
    T_Hi = np.column_stack((T_HiA, T_HiB))
    hi_idx = np.random.choice(len(T_Hi), size)
    lo_idx = np.random.choice(len(T_Lo), size)
    hi_choices = T_Hi[hi_idx]
    lo_choices = T_Lo[lo_idx]

    result = np.where(selector[:, None], hi_choices, lo_choices)
    return result, tline

def make_t2(t1, tphi):
    """
    Calculate T2 given T1 (array) and Tphi (float).
    1/T2 = 1/(2*T1) + 1/Tphi

    Args:
        t1 (np.ndarray): Array of T1 values.
        tphi (float): Tphi value.

    Returns:
        np.ndarray: Array of T2 values.
    """
    a = 1 / (2 * t1)
    b = 1 / tphi
    return 1 / (a + b)