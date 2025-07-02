
# make imports
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from avs_gazetime.config import PLOTS_DIR_NO_SUB


def load_dynamics_data(subject_id, roi_name, event_type, hemi, change_metric, delta_t):
    """
    Load dynamics data and metadata for a specific subject and ROI.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID.
    roi_name : str
        Name of the ROI.
    event_type : str
        Type of events ('fixation' or 'saccade').
    hemi : str
        Hemisphere ('lh', 'rh', or 'both').
    change_metric : str
        Distance metric used for dynamics.
    delta_t : int
        Time difference in ms used for dynamics.
        
    Returns:
    --------
    dynamics : np.ndarray
        Dynamics data (concatenated if hemi='both').
    metadata : pd.DataFrame
        Metadata for the dynamics.
    times : np.ndarray
        Timepoints corresponding to the dynamics.
    """
    # Construct path for subject's data
    sub_name = f"as{str(subject_id).zfill(2)}"  # Fixed: convert to string first
    dynamics_dir = os.path.join(PLOTS_DIR_NO_SUB, sub_name, "dynamics", sub_name)
    
    print(f"Loading dynamics data for {sub_name}, {roi_name}, hemi={hemi}")
    
    # If hemi is 'both', load and concatenate both hemispheres
    if hemi == 'both':
        # Load left hemisphere
        dynamics_fname_lh = os.path.join(
            dynamics_dir, f"dynamics_{event_type}_{roi_name}_lh_{change_metric}_{delta_t}ms.npy"
        )
        metadata_fname = os.path.join(dynamics_dir, f"metadata_{event_type}.csv")
        times_fname = os.path.join(
            dynamics_dir, f"times_{event_type}_{change_metric}_{delta_t}ms.npy"
        )
        
        # Load right hemisphere
        dynamics_fname_rh = os.path.join(
            dynamics_dir, f"dynamics_{event_type}_{roi_name}_rh_{change_metric}_{delta_t}ms.npy"
        )
        
        try:
            # Load data from both hemispheres
            dynamics_lh = np.load(dynamics_fname_lh)
            dynamics_rh = np.load(dynamics_fname_rh)
            metadata = pd.read_csv(metadata_fname)
            times = np.load(times_fname)
            
            print(f"  LH shape: {dynamics_lh.shape}, RH shape: {dynamics_rh.shape}")
            
            # Concatenate along the epochs (first) dimension
            # Assuming dynamics shape is (n_epochs, n_timepoints)
            dynamics = np.concatenate([dynamics_lh, dynamics_rh], axis=0)
            
            # Duplicate metadata for both hemispheres
            metadata_lh = metadata.copy()
            metadata_lh['hemisphere'] = 'lh'
            metadata_rh = metadata.copy()
            metadata_rh['hemisphere'] = 'rh'
            metadata = pd.concat([metadata_lh, metadata_rh], ignore_index=True)
            
            print(f"  Combined shape: {dynamics.shape}")
            print(f"  Combined metadata shape: {metadata.shape}")
            
            return dynamics, metadata, times
            
        except Exception as e:
            print(f"Error loading bilateral data for {sub_name}, {roi_name}: {e}")
            print("Attempting to load individual hemispheres...")
            
            # Try to load individual hemispheres if bilateral loading fails
            try:
                if os.path.exists(dynamics_fname_lh):
                    dynamics_lh = np.load(dynamics_fname_lh)
                    metadata = pd.read_csv(metadata_fname)
                    metadata['hemisphere'] = 'lh'
                    times = np.load(times_fname)
                    print(f"  Loaded only LH data: {dynamics_lh.shape}")
                    return dynamics_lh, metadata, times
                elif os.path.exists(dynamics_fname_rh):
                    dynamics_rh = np.load(dynamics_fname_rh)
                    metadata = pd.read_csv(metadata_fname)
                    metadata['hemisphere'] = 'rh'
                    times = np.load(times_fname)
                    print(f"  Loaded only RH data: {dynamics_rh.shape}")
                    return dynamics_rh, metadata, times
                else:
                    print(f"No data found for either hemisphere")
                    return None, None, None
            except Exception as e2:
                print(f"Error loading individual hemisphere data: {e2}")
                return None, None, None
    else:
        # Load single hemisphere (original behavior)
        dynamics_fname = os.path.join(
            dynamics_dir, f"dynamics_{event_type}_{roi_name}_{hemi}_{change_metric}_{delta_t}ms.npy"
        )
        metadata_fname = os.path.join(dynamics_dir, f"metadata_{event_type}.csv")
        times_fname = os.path.join(
            dynamics_dir, f"times_{event_type}_{change_metric}_{delta_t}ms.npy"
        )
        
        try:
            dynamics = np.load(dynamics_fname)
            metadata = pd.read_csv(metadata_fname)
            metadata['hemisphere'] = hemi
            times = np.load(times_fname)
            print(f"  Loaded {hemi} data: {dynamics.shape}")
            return dynamics, metadata, times
        except Exception as e:
            print(f"Error loading data for {sub_name}, {roi_name}, {hemi}: {e}")
            return None, None, None

def filter_dynamics(dynamics, times, cutoff_hz=50):
    """Apply boxcar filter with cutoff frequency in Hz."""
    from scipy.ndimage import uniform_filter1d
    
    sfreq = 1 / np.mean(np.diff(times))
    print("estimated sfreq", sfreq)
    window_size = max(1, int(sfreq / (2 * cutoff_hz)))
    
    # Filter along the last dimension, or along the only dimension if 1D
    if dynamics.ndim == 1:
        ax_filt = 0
    else:
        ax_filt = -1
    
    print(f"Boxcar filter: {cutoff_hz}Hz → {window_size} samples ({window_size/sfreq*1000:.1f}ms)")
    return uniform_filter1d(dynamics.astype(float), size=window_size, axis=ax_filt, mode='nearest')

def get_downward_flank_halfway(dynamics, times, duration_post, tmin_peak=0.050, tmax_peak=0.250,
                              max_t_to_halfway=0.100, max_post_fixation=0.150, 
                              tolerance_pct=5.0):
    """
    Compute the timepoint where the dynamics reach halfway down from the peak.
    Uses tolerance-based approach to handle oscillations in the data.
    
    Parameters:
    -----------
    dynamics : np.ndarray
        Dynamics data (epochs x time or just time).
    times : np.ndarray
        Timepoints corresponding to the dynamics.
    duration_post : float or np.ndarray
        Fixation duration(s) corresponding to the dynamics.
    tmin_peak, tmax_peak : float
        Time window for peak detection.
    max_t_to_halfway : float
        Maximum time after peak to search for halfway point.
    max_post_fixation : float
        Maximum time after fixation end for valid results.
    tolerance_pct : float
        Tolerance as percentage of peak-to-min range for halfway detection.
        Higher values are more robust to oscillations but less precise.
        
    Returns:
    --------
    t_halfway : float or np.ndarray
        Timepoint(s) where dynamics reach halfway down from peak.
    """
    
    def find_halfway_with_tolerance(dynamics_search, times_search, halfway_val, 
                                   peak_val, min_val, tolerance_pct):
        """
        Find halfway point with tolerance to handle oscillations.
        
        Parameters:
        -----------
        dynamics_search : np.ndarray
            Dynamics values in search window
        times_search : np.ndarray
            Time values in search window
        halfway_val : float
            Target halfway value
        peak_val, min_val : float
            Peak and minimum values for calculating tolerance
        tolerance_pct : float
            Tolerance as percentage of peak-to-min range
            
        Returns:
        --------
        float or np.nan
            Time of halfway crossing or np.nan if not found
        """
        if len(dynamics_search) == 0:
            return np.nan
            
        # Calculate tolerance based on signal range
        signal_range = peak_val - min_val
        tolerance = signal_range * tolerance_pct / 100
        
        # Find points within tolerance of halfway value
        within_tolerance = np.abs(dynamics_search - halfway_val) <= tolerance
        
        if not np.any(within_tolerance):
            # If no points within tolerance, fall back to closest point
            closest_idx = np.nanargmin(np.abs(dynamics_search - halfway_val))
            print(f"No points within tolerance, using closest point at index {closest_idx}")
            return times_search[closest_idx]
        
        # Return the first point within tolerance (closest to peak in time)
        first_idx = np.where(within_tolerance)[0][0]
        return times_search[first_idx]
    
    def process_single_epoch(dynamics_1d, times_1d, duration_single):
        """Process a single epoch of dynamics data."""
        # Apply peak detection time window
        peak_mask = (times_1d >= tmin_peak) & (times_1d <= tmax_peak)
        if not np.any(peak_mask):
            return np.nan
            
        dynamics_peak_window = dynamics_1d[peak_mask]
        times_peak_window = times_1d[peak_mask]
        
        # Find peak within detection window
        peak_idx_windowed = np.nanargmax(dynamics_peak_window)
        if np.isnan(dynamics_peak_window[peak_idx_windowed]):
            return np.nan
            
        t_peak = times_peak_window[peak_idx_windowed]
        peak_val = dynamics_peak_window[peak_idx_windowed]
        min_val = np.nanmin(dynamics_1d)
        halfway_val = min_val + (peak_val - min_val) / 2
        
        # Search for halfway point from peak to tmax_peak + max_t_to_halfway
        halfway_search_end = min(t_peak + max_t_to_halfway, times_1d[-1])
        halfway_mask = (times_1d >= t_peak) & (times_1d <= halfway_search_end)
        
        if not np.any(halfway_mask):
            return np.nan
            
        times_halfway_search = times_1d[halfway_mask]
        dynamics_halfway_search = dynamics_1d[halfway_mask]
        
        if len(times_halfway_search) <= 1:
            return np.nan
        
        # Use tolerance-based approach to find halfway point
        t_halfway = find_halfway_with_tolerance(
            dynamics_halfway_search, times_halfway_search, halfway_val,
            peak_val, min_val, tolerance_pct
        )
        
        # Check if halfway point is at edge of search window
        if (t_halfway == times_halfway_search[0] or
            t_halfway == times_halfway_search[-1]):
            return np.nan
        
        # Check if halfway point is too far after fixation end
        if t_halfway > duration_single + max_post_fixation:
            return np.nan
            
        return t_halfway
    
    # Handle single vs multiple epochs
    if dynamics.ndim == 1:
        # Single epoch
        return process_single_epoch(dynamics, times, duration_post)
    else:
        # Multiple epochs - ensure consistent processing
        n_epochs = dynamics.shape[0]
        t_halfway_results = np.full(n_epochs, np.nan)
        
        for epoch_idx in range(n_epochs):
            # Get duration for this epoch
            if isinstance(duration_post, np.ndarray):
                dur = duration_post[epoch_idx]
            else:
                dur = duration_post
            
            # Process this epoch
            t_halfway_results[epoch_idx] = process_single_epoch(
                dynamics[epoch_idx], times, dur
            )
            
        return t_halfway_results

def get_peak_latency(dynamics, times, duration_post, tmin_peak=0.050, tmax_peak=0.250, max_post_fixation=0.150):
    """
    Compute the latency of the peak in dynamics data.
    
    Parameters:
    -----------
    dynamics : np.ndarray
        Dynamics data.
    times : np.ndarray
        Timepoints corresponding to the dynamics.
    duration_post : float or np.ndarray
        Fixation duration(s).
    tmin_peak, tmax_peak : float
        Time window for peak detection.
    max_post_fixation : float
        Maximum time after fixation end for valid results.
        
    Returns:
    --------
    peak_latencies : np.ndarray
        Peak latencies.
    """
    # Apply peak detection time window
    time_mask = (times >= tmin_peak) & (times <= tmax_peak)
    dynamics_windowed = dynamics[:, time_mask]
    times_windowed = times[time_mask]
    
    # Find peaks for each epoch
    peak_indices = np.nanargmax(dynamics_windowed, axis=1)
    peak_latencies = times_windowed[peak_indices]
    
    # Check if peak is too far after fixation end
    if isinstance(duration_post, np.ndarray):
        valid_mask = peak_latencies <= duration_post + max_post_fixation
        peak_latencies = np.where(valid_mask, peak_latencies, np.nan)
    else:
        peak_latencies = np.where(
            peak_latencies <= duration_post + max_post_fixation, 
            peak_latencies, 
            np.nan
        )
        
    return peak_latencies

def get_peak_amplitude(dynamics, times, duration_post, tmin_peak=0.050, tmax_peak=0.250, max_post_fixation=0.150):
    """
    Compute the amplitude of the peak in dynamics data.
    
    Parameters:
    -----------
    dynamics : np.ndarray
        Dynamics data.
    times : np.ndarray
        Timepoints corresponding to the dynamics.
    duration_post : float or np.ndarray
        Fixation duration(s).
    tmin_peak, tmax_peak : float
        Time window for peak detection.
    max_post_fixation : float
        Maximum time after fixation end for valid results.
        
    Returns:
    --------
    peak_amplitudes : np.ndarray
        Peak amplitudes.
    """
    # Apply peak detection time window
    time_mask = (times >= tmin_peak) & (times <= tmax_peak)
    dynamics_windowed = dynamics[:, time_mask]
    
    # Find peak amplitudes for each epoch
    peak_amplitudes = np.nanmax(dynamics_windowed, axis=1)
    
    return peak_amplitudes