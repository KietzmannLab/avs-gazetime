"""
Module for computing neural dynamics over time.
"""
import numpy as np
import scipy.spatial.distance as ssd
from joblib import Parallel, delayed
from tqdm import tqdm

def compute_change_over_time(meg_data, delta_t_samples, stride, change_metric="correlation", n_jobs=-2, zscore=False):
    """
    Compute the change over time for neural data by measuring the distance between
    multivariate patterns at time t and time t+delta_t.
    
    Parameters:
    -----------
    meg_data : np.ndarray
        MEG data of shape (n_epochs, n_channels, n_times).
    delta_t_samples : int
        Number of samples (timepoints) to advance for computing change.
    stride : int
        Step size for computing change (every stride-th timepoint).
    change_metric : str, optional
        Distance metric to use ('correlation', 'cosine', 'euclidean', 'mahalanobis').
    n_jobs : int, optional
        Number of parallel jobs to run.
    zscore : bool, optional
        Whether to z-score the data per epoch over time.
        
    Returns:
    --------
    dynamics : np.ndarray
        Array of shape (n_epochs, n_timepoints) containing change values.
    """
    # Check data dimensionality and prepare it if needed
    if len(np.shape(meg_data)) == 4:
        # Take norm over channel dimension if necessary (e.g., for gradiometer pairs)
        meg_data = np.linalg.norm(meg_data, axis=-2)
        print(f"Taking norm over channel dimension, new shape: {np.shape(meg_data)}")
    
    # Z-score the data if requested
    if zscore:
        meg_data = (meg_data - np.mean(meg_data, axis=2, keepdims=True)) / np.std(meg_data, axis=2, keepdims=True)
    
    # Get data dimensions
    n_epochs, n_channels, n_timepoints = meg_data.shape
    
    # Precompute the pairs of timepoints for comparison
    timepoint_pairs = [(t, t + delta_t_samples) for t in range(0, n_timepoints - delta_t_samples, stride)]
    print(f"Number of timepoint pairs: {len(timepoint_pairs)}")
    
    def change_over_time_per_epoch(timepoint_pairs, popcode_epoch):
        """Compute change over time for a single epoch."""
        cot_epoch = np.full(len(timepoint_pairs), np.nan)
        
        for i, (t1, t2) in enumerate(timepoint_pairs):
            # Get data at timepoints t1 and t2
            popcode_t1 = popcode_epoch[:,t1]
            popcode_t2 = popcode_epoch[:,t2]
            
            # Compute distance using specified metric
            if change_metric == "correlation":
                change = ssd.correlation(popcode_t1, popcode_t2)
            elif change_metric == "cosine":
                change = ssd.cosine(popcode_t1, popcode_t2)
            elif change_metric == "euclidean":
                change = ssd.euclidean(popcode_t1, popcode_t2)
            elif change_metric == "mahalanobis":
                # For mahalanobis, we need to compute the inverse covariance matrix
                # Use a window of samples around t1 to estimate covariance
                t1_window_start = max(0, t1 - delta_t_samples)
                popcode_window = popcode_epoch[:,t1_window_start:t1]
                # Use pseudoinverse for numerical stability
                inv_cov = np.linalg.pinv(np.cov(popcode_window.T, rowvar=False))
                change = ssd.mahalanobis(popcode_t1, popcode_t2, inv_cov)
                
            cot_epoch[i] = change
        
        return cot_epoch
    
    # Compute change over time for all epochs in parallel
    dynamics = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(change_over_time_per_epoch)(timepoint_pairs, meg_data[i]) 
        for i in tqdm(range(n_epochs), desc="Computing change over time")
    )
    
    # Convert to array
    dynamics = np.array(dynamics)
    
    return dynamics
