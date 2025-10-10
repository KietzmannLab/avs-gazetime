"""
Module for computing phase-amplitude coupling (PAC).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from scipy.signal import hilbert
from mne.filter import filter_data
from joblib import Parallel, delayed
from tqdm import tqdm
from avs_gazetime.utils.simple_timing import simple_timer, tic, toc

# Import optimized n_jobs settings
from avs_gazetime.pac.pac_dataloader import optimize_n_jobs

# Get optimized parallel job settings
PARALLEL_JOBS = optimize_n_jobs()

def circular_linear_correlation(phase_data, amplitude_data):
    """
    Compute the circular-linear correlation coefficient between phase and amplitude.
    
    Parameters:
    - phase_data: np.ndarray
        Array of phase angles (in radians) for the lower frequency oscillations.
    - amplitude_data: np.ndarray
        Array of amplitude values for the higher frequency oscillations.
    
    Returns:
    - rho: float
        Circular-linear correlation coefficient.
    """
    # Ensure data are numpy arrays
    phase_data = np.asarray(phase_data)
    amplitude_data = np.asarray(amplitude_data)
    
    # Convert phase data to complex numbers using sine and cosine
    sin_phase = np.sin(phase_data)
    cos_phase = np.cos(phase_data)
    
    # Compute correlations
    rca = np.corrcoef(cos_phase, amplitude_data)[0, 1]
    rsa = np.corrcoef(sin_phase, amplitude_data)[0, 1]
    rcs = np.corrcoef(sin_phase, cos_phase)[0, 1]
    
    # Calculate the circular-linear correlation coefficient
    rho = np.sqrt(rca**2 + rsa**2 - 2 * rca * rsa * rcs) / np.sqrt(1 - rcs**2)
    
    return rho

def compute_modulation_index(theta_phase, gamma_amplitude, n_bins=18):
    """
    Compute the Modulation Index based on the Kullback-Leibler distance
    for phase-amplitude coupling as suggested by Tort et al. (2010).
    
    Parameters:
    - theta_phase: np.ndarray
        Array of theta phases for a specific channel (in radians).
    - gamma_amplitude: np.ndarray
        Array of gamma amplitudes for the same channel.
    - n_bins: int
        Number of phase bins to use for coupling calculation.

    Returns:
    - mi: float
        The Modulation Index value.
    """
    # Create phase bins
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # Initialize the binned amplitude array
    binned_amplitude = np.zeros(n_bins)

    # Compute the mean gamma amplitude for each phase bin
    for i in range(n_bins):
        # Find indices for the current phase bin
        indices = np.where((theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1]))
        # Calculate the mean amplitude in the current bin
        if len(indices[0]) > 0:
            binned_amplitude[i] = np.mean(gamma_amplitude[indices])

    # Normalize the binned amplitude to get a probability distribution
    binned_amplitude += 1e-10  # Avoid division by zero
    binned_amplitude /= binned_amplitude.sum()

    # Calculate Kullback-Leibler divergence from uniform distribution
    uniform_distribution = np.ones(n_bins) / n_bins  # Uniform distribution
    kl_divergence = np.sum(binned_amplitude * np.log(binned_amplitude / uniform_distribution))

    # Normalize the KL divergence to get the Modulation Index (MI)
    max_entropy = np.log(n_bins)  # Maximum entropy for uniform distribution
    mi = kl_divergence / max_entropy  # Normalize by max entropy

    return mi

def direct_pac_ozturk(theta_phase, gamma_amplitude):
    """
    Compute the PAC value using the method proposed by Ozturk et al. (2019).
    
    Parameters:
    - theta_phase: np.ndarray
        Array of theta phases for a specific channel.
    - gamma_amplitude: np.ndarray
        Array of gamma amplitudes for the same channel.
    
    Returns:
    - pac: float
        The PAC value.
    """
    # Compute the direct PAC estimate
    sum_aH_eiL = np.sum(gamma_amplitude * np.exp(1j * theta_phase))
    sum_aH2 = np.sum(gamma_amplitude**2)
    
    direct_pac = np.abs(sum_aH_eiL) / np.sqrt(sum_aH2)
    return direct_pac

def compute_phase_shuffle_bootstrap(theta_phase, gamma_amplitude, pac_function, n_bootstraps, rng):
    """
    Compute bootstrap samples by randomly shuffling phase data.
    
    Parameters:
    - theta_phase: np.ndarray
        Array of theta phases.
    - gamma_amplitude: np.ndarray
        Array of gamma amplitudes.
    - pac_function: callable
        Function to compute PAC.
    - n_bootstraps: int
        Number of bootstrap samples.
    - rng: numpy.random.Generator
        Random number generator.
        
    Returns:
    - baseline_bootstrap: np.ndarray
        Array of bootstrap PAC values.
    """
    baseline_bootstrap = np.zeros(n_bootstraps)
    
    for b in range(n_bootstraps):
        # For 1D arrays
        if theta_phase.ndim == 1:
            shuffled_indices = rng.permutation(len(theta_phase))
            baseline_bootstrap[b] = pac_function(theta_phase[shuffled_indices], gamma_amplitude)
        # For 2D arrays (epochs, times)
        else:
            # Shuffle each epoch independently
            shuffled_phase = np.zeros_like(theta_phase)
            for epoch in range(theta_phase.shape[0]):
                shuffled_indices = rng.permutation(theta_phase.shape[1])
                shuffled_phase[epoch] = theta_phase[epoch, shuffled_indices]
            baseline_bootstrap[b] = pac_function(shuffled_phase, gamma_amplitude)
    
    return baseline_bootstrap


def compute_session_aware_bootstrap(theta_phase, gamma_amplitude, pac_function, sessions, n_bootstraps, rng):
    """
    Compute bootstrap samples while preserving session structure.
    
    Parameters:
    - theta_phase: np.ndarray
        Array of theta phases.
    - gamma_amplitude: np.ndarray
        Array of gamma amplitudes.
    - pac_function: callable
        Function to compute PAC.
    - sessions: np.ndarray
        Array of session indices.
    - n_bootstraps: int
        Number of bootstrap samples.
    - rng: numpy.random.Generator
        Random number generator.
        
    Returns:
    - baseline_bootstrap: np.ndarray
        Array of bootstrap PAC values.
    """
    baseline_bootstrap = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        # Shuffle within each session
        shuffled_indices = np.zeros(theta_phase.shape[0], dtype=int)
        for s in np.unique(sessions):
            session_indices = np.where(sessions == s)[0]
            shuffled_indices[session_indices] = session_indices[rng.permutation(len(session_indices))]
        # Compute the PAC value with shuffled phases
        baseline_bootstrap[b] = pac_function(theta_phase[shuffled_indices], gamma_amplitude)
    return baseline_bootstrap
#@simple_timer

def generate_single_cut_surrogate(data, random_seed=None):
    """
    Generate a single-point cut surrogate for a time series.
    
    This method cuts the time series at a single randomly chosen point and
    exchanges the two resulting segments, minimizing distortion of the
    original dynamics while destroying the specific relationship between
    different time points.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data to generate surrogate from with shape (epochs, times).
    random_seed : int or None
        Seed for the random number generator.
        
    Returns:
    --------
    surrogate : np.ndarray
        Surrogate time series with same shape as input data.
    """
    # Set random seed if provided
    rng = np.random.RandomState(random_seed)
    
    # Create a copy of the original data
    surrogate = data.copy()
    
    # Get the number of epochs and timepoints
    n_epochs, n_times = data.shape
    
    # For each epoch, cut at a different random point and swap segments
    for i in range(n_epochs):
        # Choose a random cut point (not at the edges)
        cut_point = rng.randint(1, n_times - 1)
        
        # Swap the two segments for this specific epoch
        surrogate[i] = np.concatenate((data[i, cut_point:], data[i, :cut_point]))
    
    return surrogate

def compute_single_cut_bootstrap(theta_phase, gamma_amplitude, pac_function, n_bootstraps, rng):
    """
    Compute bootstrap samples using the single-point cut method.
    
    This applies the single-point cut method to the phase data only,
    while keeping the amplitude data intact, as we want to test
    specifically for phase-amplitude relationships.
    
    Parameters:
    - theta_phase: np.ndarray
        Array of theta phases with shape (epochs, times).
    - gamma_amplitude: np.ndarray
        Array of gamma amplitudes with shape (epochs, times).
    - pac_function: callable
        Function to compute PAC.
    - n_bootstraps: int
        Number of bootstrap samples.
    - rng: numpy.random.Generator
        Random number generator.
        
    Returns:
    - baseline_bootstrap: np.ndarray
        Array of bootstrap PAC values.
    """
    baseline_bootstrap = np.zeros(n_bootstraps)
    
    for b in range(n_bootstraps):
        # Generate a seed for each bootstrap iteration
        seed = rng.choice(np.arange(1, 10000))
        
        # Generate surrogate using single-point cut method
        surrogate_phase = generate_single_cut_surrogate(theta_phase, seed)
        
        # Flatten the 2D arrays to 1D for PAC computation if needed
        # This is necessary if the PAC function expects 1D arrays
        if pac_function.__name__ == 'compute_modulation_index':
            # For modulation index, we need to flatten the arrays
            flattened_phase = surrogate_phase.flatten()
            flattened_amplitude = gamma_amplitude.flatten()
            baseline_bootstrap[b] = pac_function(flattened_phase, flattened_amplitude)
        else:
            # For other PAC methods, we pass the 2D arrays directly
            baseline_bootstrap[b] = pac_function(surrogate_phase, gamma_amplitude)
    
    return baseline_bootstrap

#@simple_timer
def compute_pac_hilbert(data, sfreq, channel, theta_band=(5, 10), gamma_band=(60, 140),
                       times=None, time_window=(0.150, 0.400), n_bootstraps=200,
                       plot=False, verbose=True, durations=None,
                       method='modulation_index', sessions=None, random_seed=42,
                       surrogate_style='phase_shuffle', theta_data_prefiltered=None,
                       gamma_data_prefiltered=None):
    """
    Compute phase-amplitude coupling using the Hilbert transform for a specific channel.

    Parameters:
    - data: np.ndarray
        The MEG/EEG data array of shape (n_epochs, n_channels, n_times).
        Only used if theta_data_prefiltered and gamma_data_prefiltered are None.
    - sfreq: float
        The sampling frequency of the data.
    - channel: int
        The channel index to compute PAC for.
    - theta_band: tuple
        Frequency range for theta band (low_freq, high_freq).
    - gamma_band: tuple
        Frequency range for gamma band (low_freq, high_freq).
    - times: np.ndarray
        Array of time points corresponding to the data.
    - time_window: tuple
        Time window in seconds to isolate for PAC computation.
    - n_bootstraps: int
        Number of bootstraps for baseline computation.
    - plot: bool
        Whether to plot the PAC results.
    - verbose: bool
        Whether to print detailed diagnostics.
    - durations: np.ndarray
        Array of durations for each epoch.
    - method: str
        Method to use for PAC computation ('modulation_index', 'circular_linear_corr', or 'direct_pac').
    - sessions: np.ndarray
        Array of session indices to optionally guide the baseline computation.
    - random_seed: int
        Seed for random number generator to ensure reproducibility.
    - surrogate_style: str
        Method to generate surrogate data ('phase_shuffle', 'session_aware', or 'single_cut').
    - theta_data_prefiltered: np.ndarray or None
        Pre-filtered theta data of shape (n_epochs, n_channels, n_times). If provided,
        skips filtering step for theta band.
    - gamma_data_prefiltered: np.ndarray or None
        Pre-filtered gamma data of shape (n_epochs, n_channels, n_times). If provided,
        skips filtering step for gamma band.

    Returns:
    - z_scores: float
        Z-scored phase-amplitude coupling value for the specified channel.
    """
    # Validate input parameters
    if method not in ['modulation_index', 'circular_linear_corr', 'direct_pac']:
        raise ValueError("Invalid method. Choose from 'modulation_index', 'circular_linear_corr', or 'direct_pac'.")
    
    if surrogate_style not in ['phase_shuffle', 'session_aware', 'single_cut']:
        raise ValueError("Invalid surrogate_style. Choose from 'phase_shuffle', 'session_aware', or 'single_cut'.")
    
    if verbose:
        print(f"Filtering data for channel {channel} in theta band {theta_band} and gamma band {gamma_band}")
        print(f"Time window: {time_window}")
        print("Data shape:", data.shape)
        print(f"Using surrogate style: {surrogate_style}")
    
    # Select appropriate PAC function based on method
    if method == 'modulation_index':
        pac_function = compute_modulation_index
    elif method == 'direct_pac':
        pac_function = direct_pac_ozturk
    else:  # circular_linear_corr
        pac_function = circular_linear_correlation
        
    if verbose:
        print(theta_band, gamma_band)
        if data is not None:
            print(data.shape)

    # Use pre-filtered data if provided, otherwise filter now
    if theta_data_prefiltered is not None and gamma_data_prefiltered is not None:
        if verbose:
            print(f"Using pre-filtered data for channel {channel}")
        theta_data = theta_data_prefiltered[:, channel, :]
        gamma_data = gamma_data_prefiltered[:, channel, :]
    else:
        if verbose:
            print(f"Filtering data for channel {channel}")
        tic("filter theta")
        # Filter data for theta and gamma bands (causal filter, minimum phase)
        theta_data = filter_data(data[:, channel, :].astype(float), sfreq,
                               theta_band[0], theta_band[1],
                               method='fir', phase='minimum', n_jobs=PARALLEL_JOBS["filter"], verbose=0)
        toc("filter theta")

        tic("filter gamma")
        gamma_data = filter_data(data[:, channel, :].astype(float), sfreq,
                               gamma_band[0], gamma_band[1],
                               method='fir', phase='minimum', n_jobs=PARALLEL_JOBS["filter"], verbose=0)
        toc("filter gamma")
    
    if verbose:
        print(f"Theta data shape: {theta_data.shape}, Gamma data shape: {gamma_data.shape}")

    # Identify valid epochs (epochs that are long enough)
    valid_epochs = durations > time_window[1]
    if np.sum(valid_epochs) == 0:
        raise ValueError("No valid epochs found. All epochs are shorter than the time window.")
    
    # Apply valid epoch mask
    theta_data = theta_data[valid_epochs, :]
    gamma_data = gamma_data[valid_epochs, :]
    
    # Create time mask for the specified time window
    times_mask = (times >= time_window[0]) & (times <= time_window[1])
    
    # Print information about valid epochs
    if verbose:
        print(f"Valid epochs: {np.sum(valid_epochs)}")
    
    # Update sessions array if provided
    if sessions is not None:
        sessions = sessions[valid_epochs]
    
    # Plot median of theta and gamma data if requested
    if plot:
        plot_median_theta_gamma(theta_data, gamma_data, times, times_mask, channel, 
                               theta_band, gamma_band,
                               PLOTS_DIR=os.environ.get('PLOTS_DIR', './plots'))
    
    # Compute the Hilbert transform to get phase and amplitude
    theta_phase = np.angle(hilbert(theta_data, axis=1))
    gamma_amplitude = np.abs(hilbert(gamma_data, axis=1))
    
   
    # Apply the time mask
    theta_phase = theta_phase[:, times_mask]
    gamma_amplitude = gamma_amplitude[:, times_mask]
    
    if verbose:
        print("Theta phase shape:", theta_phase.shape, "min:", np.min(theta_phase), "max:", np.max(theta_phase))
        print("Gamma amplitude shape:", gamma_amplitude.shape, "min:", np.min(gamma_amplitude), "max:", np.max(gamma_amplitude))
        print(f"Computing PAC values for channel {channel} using {method} method and {surrogate_style} surrogate style")
    
    # Plot phase histograms if requested
    if plot:
        plot_theta_phase_histogram(theta_phase, channel, 
                                  PLOTS_DIR=os.environ.get('PLOTS_DIR', './plots'))
        plot_pac_histogram(theta_phase, gamma_amplitude, channel, 
                          PLOTS_DIR=os.environ.get('PLOTS_DIR', './plots'))
    
    # Compute PAC values
    # Flatten arrays for PAC computation if needed
    if method == 'modulation_index':
        theta_phase_flat = theta_phase.flatten()
        gamma_amplitude_flat = gamma_amplitude.flatten()
        pac_value = pac_function(theta_phase_flat, gamma_amplitude_flat)
    else:
        pac_value = pac_function(theta_phase, gamma_amplitude)
    
    if verbose:
        print(f"PAC value: {pac_value}")
    
    # Compute the baseline distribution with bootstrapping
    rng = np.random.default_rng(seed=random_seed)
    tic("baseline")
    
    # Choose surrogate method based on surrogate_style
    if surrogate_style == 'session_aware' and sessions is not None:
        # Session-aware shuffling
        if method == 'modulation_index':
            # For modulation index, use flattened arrays
            baseline_bootstrap = compute_session_aware_bootstrap(
                theta_phase_flat, gamma_amplitude_flat, pac_function, sessions, n_bootstraps, rng)
        else:
            baseline_bootstrap = compute_session_aware_bootstrap(
                theta_phase, gamma_amplitude, pac_function, sessions, n_bootstraps, rng)
    elif surrogate_style == 'single_cut':
        # Single-point cut method (Aru et al.)
        baseline_bootstrap = compute_single_cut_bootstrap(
            theta_phase, gamma_amplitude, pac_function, n_bootstraps, rng)
    else:
        # Regular phase shuffling
        if method == 'modulation_index':
            # For modulation index, use flattened arrays
            baseline_bootstrap = compute_phase_shuffle_bootstrap(
                theta_phase_flat, gamma_amplitude_flat, pac_function, n_bootstraps, rng)
        else:
            baseline_bootstrap = compute_phase_shuffle_bootstrap(
                theta_phase, gamma_amplitude, pac_function, n_bootstraps, rng)
    
    baseline_bootstrap = np.array(baseline_bootstrap)
    toc("baseline")
    
    # Fit a normal distribution to calculate mean and std
    tic("fit normal")
    baseline_normal = stats.norm.fit(baseline_bootstrap)
    baseline_mean, baseline_std = baseline_normal
    toc("fit normal")
    
    if verbose:
        print(f"Baseline bootstrap shape: {baseline_bootstrap.shape}")
        print(f"Baseline mean: {baseline_mean}")
        print(f"Baseline std: {baseline_std}")
    
    # Compute the z-score
    z_score = (pac_value - baseline_mean) / baseline_std
    
    # Plot results if requested
    if plot:
        plot_pac_results(baseline_bootstrap, baseline_mean, baseline_std, pac_value, 
                        channel, theta_band, gamma_band, surrogate_style,
                        PLOTS_DIR=os.environ.get('PLOTS_DIR', './plots'))
    
    return z_score

def compute_full_cross_frequency_pac_matrix(data, sfreq, significant_channels, theta_band, gamma_band, times,
                                           time_window, n_bootstraps=200, method='modulation_index',
                                           random_seed=42, steps_theta=1, steps_gamma=10, durations=None, 
                                           sessions=None, surrogate_style='phase_shuffle'):
    """
    Compute the full cross-frequency PAC matrix for all significant channels.
    
    Parameters:
    -----------
    data: np.ndarray
        MEG/EEG data array of shape (n_epochs, n_channels, n_times).
    sfreq: float
        Sampling frequency.
    significant_channels: list
        List of channel indices for which to compute PAC matrices.
    theta_band: tuple
        Frequency range for theta band.
    gamma_band: tuple
        Frequency range for gamma band.
    times: np.ndarray
        Time points corresponding to the data.
    time_window: tuple
        Time window for analysis.
    n_bootstraps: int
        Number of bootstraps for baseline computation.
    method: str
        Method for PAC computation ('modulation_index', 'circular_linear_corr', or 'direct_pac').
    random_seed: int
        Random seed for reproducibility.
    steps_theta: int
        Step size for theta frequency sweep.
    steps_gamma: int
        Step size for gamma frequency sweep.
    durations: np.ndarray
        Durations for each epoch.
    sessions: np.ndarray
        Session indices for each epoch.
    surrogate_style: str
        Method to generate surrogate data ('phase_shuffle', 'session_aware', or 'single_cut').
        
    Returns:
    --------
    fill_pac_results: dict
        Dictionary with channel indices as keys and PAC matrices as values.
    """
    # Compute the full cross-frequency PAC matrix for all significant channels
    theta_frequencies = np.arange(theta_band[0], theta_band[1]+1, steps_theta)
    gamma_frequencies = np.arange(gamma_band[0], gamma_band[1]+1, steps_gamma)
    
    # Prepare a results dict with channel as key and pac matrix as value
    fill_pac_results = {}
    
    for channel in significant_channels:
        pac_matrix = np.zeros((len(theta_frequencies), len(gamma_frequencies)))
        # Precompute the theta-gamma pairs to use joblib
        theta_gamma_pairs = []
        matrix_ids = []
        
        for i, theta_freq in enumerate(theta_frequencies):
            # Fixed bandwidth of 2 Hz for phase (±1 Hz around center frequency)
            low_theta = theta_freq - 1
            high_theta = theta_freq + 1
            
            for j, gamma_freq in enumerate(gamma_frequencies):
                # Variable bandwidth for amplitude based on phase frequency
                # Bandwidth = 2 × center frequency of the phase signal (see Aru et al. 2015)
                amp_bandwidth = 2 * theta_freq
                low_gamma = gamma_freq - amp_bandwidth/2
                high_gamma = gamma_freq + amp_bandwidth/2
                
                # Ensure the frequency bounds are valid
                low_gamma = max(1, low_gamma)  # Prevent negative or zero frequencies
                
                matrix_ids.append((i, j))
                theta_gamma_pairs.append((low_theta, high_theta, low_gamma, high_gamma))
        
        # Use joblib to parallelize the computation
        pac_values = Parallel(n_jobs=PARALLEL_JOBS["cross_freq"])(
            delayed(compute_pac_hilbert)(
                data, sfreq, channel, theta_band=(low_theta, high_theta), gamma_band=(low_gamma, high_gamma), 
                times=times, time_window=time_window, n_bootstraps=n_bootstraps, method=method, durations=durations, 
                sessions=sessions, random_seed=random_seed, verbose=False, surrogate_style=surrogate_style
            ) for low_theta, high_theta, low_gamma, high_gamma in tqdm(theta_gamma_pairs, desc="Computing PAC matrix", unit="pair")
        )
        
        # Fill the pac matrix
        for i, pac_value in enumerate(pac_values):
            pac_matrix[matrix_ids[i]] = pac_value
        
        fill_pac_results[channel] = pac_matrix
    
    return fill_pac_results

# Utility functions for plotting
def plot_median_theta_gamma(theta_data, gamma_data, times, times_mask, channel, 
                           theta_band, gamma_band, PLOTS_DIR='./plots'):
    """Plot the median of theta and gamma data over epochs."""
    # Ensure plot directory exists
    os.makedirs(os.path.join(PLOTS_DIR, "pac"), exist_ok=True)
    
    # Calculate median and 95% confidence intervals
    theta_median = np.median(theta_data, axis=0)
    gamma_median = np.median(gamma_data, axis=0)
    
    # Calculate bootstrap confidence intervals
    n_bootstraps = 1000
    theta_bootstraps = []
    gamma_bootstraps = []
    
    for _ in range(n_bootstraps):
        bootstrap_idx = np.random.choice(theta_data.shape[0], theta_data.shape[0], replace=True)
        theta_bootstraps.append(np.median(theta_data[bootstrap_idx], axis=0))
        gamma_bootstraps.append(np.median(gamma_data[bootstrap_idx], axis=0))
    
    theta_bootstraps = np.array(theta_bootstraps)
    gamma_bootstraps = np.array(gamma_bootstraps)
    
    theta_lower = np.percentile(theta_bootstraps, 2.5, axis=0)
    theta_upper = np.percentile(theta_bootstraps, 97.5, axis=0)
    gamma_lower = np.percentile(gamma_bootstraps, 2.5, axis=0)
    gamma_upper = np.percentile(gamma_bootstraps, 97.5, axis=0)
    
    # Create figure with two subplots
    sns.set_context("poster")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot theta data
    ax1.plot(times, theta_median, color='blue', label=f'Theta ({theta_band[0]}-{theta_band[1]} Hz)')
    ax1.fill_between(times, theta_lower, theta_upper, color='blue', alpha=0.3)
    ax1.axvspan(times[times_mask][0], times[times_mask][-1], color='gray', alpha=0.2, label='Analysis Window')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Median Theta Band Activity - Channel {channel}')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot gamma data
    ax2.plot(times, gamma_median, color='red', label=f'Gamma ({gamma_band[0]}-{gamma_band[1]} Hz)')
    ax2.fill_between(times, gamma_lower, gamma_upper, color='red', alpha=0.3)
    ax2.axvspan(times[times_mask][0], times[times_mask][-1], color='gray', alpha=0.2, label='Analysis Window')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Median Gamma Band Activity - Channel {channel}')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    ch_name = f"MEG{channel + 1}"
    fname = f"median_theta_gamma_channel_{channel}.png"
    fig.savefig(os.path.join(PLOTS_DIR, "pac", fname))
    plt.close(fig)

def plot_theta_phase_histogram(theta_phase, channel, PLOTS_DIR='./plots'):
    """Plot histogram of theta phase values."""
    # Ensure plot directory exists
    os.makedirs(os.path.join(PLOTS_DIR, "pac"), exist_ok=True)
    
    sns.set_context("poster")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    hist, bins = np.histogram(theta_phase.flatten(), bins=36)
    ax.bar(bins[:-1], hist)
    ax.set_xlabel("phase [rad]")
    ax.set_ylabel("count")
    ax.set_title("Theta phase histogram")
    
    fig.tight_layout()
    fname = f"theta_phase_hist_channel_{channel}.png"
    print("Saving histogram to", os.path.join(PLOTS_DIR, "pac", fname))
    fig.savefig(os.path.join(PLOTS_DIR, "pac", fname))
    plt.close(fig)

def plot_pac_histogram(theta_phase, gamma_amplitude, channel, PLOTS_DIR='./plots'):
    """Plot histogram of gamma amplitude binned by theta phase."""
    # Ensure plot directory exists
    os.makedirs(os.path.join(PLOTS_DIR, "pac"), exist_ok=True)
    
    # Compute mean gamma amplitude per phase bin
    n_bins = 18
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    binned_amplitude = np.zeros(n_bins)
    
    for i in range(n_bins):
        indices = np.where((theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1]))
        if len(indices[0]) > 0:
            binned_amplitude[i] = np.mean(gamma_amplitude[indices])
    
    # Normalize
    binned_amplitude += 1e-10  # Avoid division by zero
    binned_amplitude /= binned_amplitude.sum()
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.bar(np.arange(len(binned_amplitude)), binned_amplitude)
    ax.set_xlabel("phase bin")
    ax.set_ylabel("gamma amplitude [normalized]")
    ax.set_title("PAC histogram")
    
    fig.tight_layout()
    fname = f"mi_hist_{channel}.png"
    fig.savefig(os.path.join(PLOTS_DIR, "pac", fname))
    plt.close(fig)

def plot_pac_results(baseline_bootstrap, baseline_mean, baseline_std, pac_values, 
                    channel, theta_band, gamma_band, PLOTS_DIR='./plots'):
    """Plot PAC results against baseline distribution."""
    # Ensure plot directory exists
    os.makedirs(os.path.join(PLOTS_DIR, "pac"), exist_ok=True)
    
    sns.set_context("poster")
    
    # Z-score the values
    baseline_bootstrap_z = (baseline_bootstrap - baseline_mean) / baseline_std
    pac_zscores = (pac_values - baseline_mean) / baseline_std
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(baseline_bootstrap_z, bins=20, color="gray", alpha=0.5, label="baseline")
    ax.axvline(pac_zscores, color="r", label="PAC value")
    ax.axvline(np.mean(baseline_bootstrap_z), color="k", linestyle="--", label="baseline mean")
    ax.legend(frameon=False)
    ax.set_xlabel("PAC [z-scored]")
    
    # Channel name
    ch_name = f"MEG{channel + 1}"
    ax.set_title(f"theta-gamma PAC for channel {ch_name}")
    
    fig.tight_layout()
    fname = f"PAC_channel_{channel}_theta_{theta_band[0]}-{theta_band[1]}_gamma_{gamma_band[0]}-{gamma_band[1]}.png"
    fig.savefig(os.path.join(PLOTS_DIR, "pac", fname))
    plt.close(fig)