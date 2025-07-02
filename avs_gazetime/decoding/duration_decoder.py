"""Not used for the paper, but kept for future reference."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

# Import utility functions from the existing codebase
import avs_gazetime.utils.load_data as load_data
from avs_gazetime.utils.sensors_mapping import grads, mags
from avs_gazetime.utils.tools import compute_quantiles, get_quantile_data

# Configuration parameters - these would typically be imported from a params file
from avs_gazetime.config import (
    S_FREQ,
    SESSIONS,
    SUBJECT_ID,
    ET_DIR,
    PLOTS_DIR,
    MEG_DIR,
    CH_TYPE
)

# Parameters for the decoder
EVENT_TYPE = "fixation"  # or "saccade" depending on your analysis
DECIM = 5  # Decimation factor
TIME_TO_END = 0.025  # Time buffer in seconds to ensure epoch hasn't ended
APPLY_MEDIAN_SCALE = True  # Whether to apply median scaling to MEG data
N_FOLDS = 5  # Number of cross-validation folds
N_JOBS = -2  # Number of parallel jobs (-1 means use all processors)
TARGET_COL = "duration"  # Column to predict (e.g., fixation duration)

# New parameters for enhancements
LOG_TRANSFORM = False  # Whether to apply log transformation to duration values
HYPERPARAMETER_SEARCH = True  # Whether to perform hyperparameter search
ALPHAS = np.logspace(-3, 10, 7)  # Alpha values to try for Ridge regression
HP_CV_FOLDS = 3  # Number of CV folds for hyperparameter search

# New parameter for outlier clipping
CLIP_OUTLIERS = True  # Whether to clip outliers
CLIP_STD_THRESHOLD = 10  # Number of standard deviations for clipping

# scaler settings
SCALER = "robust"  # or "robust" or "minmax"

def clip_outliers(data, std_threshold=3):
    """
    Clip outliers in data beyond a certain number of standard deviations.
    
    Parameters:
    - data: np.ndarray
        Data array to be clipped
    - std_threshold: float
        Number of standard deviations beyond which to clip values
        
    Returns:
    - clipped_data: np.ndarray
        Data array with outliers clipped
    """
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - std_threshold * std
    upper_bound = mean + std_threshold * std
    
    return np.clip(data, lower_bound, upper_bound)

def prepare_data_for_timepoint_decoder(meg_data, meta_df, times, time_to_end, target_col="duration", 
                                     log_transform=False, clip_outliers_flag=False, clip_std_threshold=3):
    """
    Prepare data for timepoint-by-timepoint decoding, respecting the time_to_end constraint.
    
    Parameters:
    - meg_data: np.ndarray
        MEG data array of shape (n_epochs, n_channels, n_times)
    - meta_df: pd.DataFrame
        Metadata DataFrame containing duration information
    - times: np.ndarray
        Array of time points corresponding to the MEG data
    - time_to_end: float
        Minimum time (in seconds) that must remain in an epoch for it to be used
    - target_col: str
        Column name in meta_df containing the target values to predict
    - log_transform: bool
        Whether to apply log transformation to target values
    - clip_outliers_flag: bool
        Whether to clip outliers in the MEG data
    - clip_std_threshold: float
        Number of standard deviations beyond which to clip values
        
    Returns:
    - X_per_timepoint: list of np.ndarray
        List of feature matrices for each timepoint
    - y_per_timepoint: list of np.ndarray
        List of target vectors for each timepoint
    - valid_epochs_per_timepoint: list of np.ndarray
        List of boolean masks indicating valid epochs for each timepoint
    """
    # Initialize lists to store data for each timepoint
    X_per_timepoint = []
    y_per_timepoint = []
    valid_epochs_per_timepoint = []
    
    # Get target values
    y_all = meta_df[target_col].values
    
    #describe the data
    print(f"Data description: {meta_df[target_col].describe()}")
    # debug
    #from IPython import embed; embed()
    
    # Apply log transformation if requested
    if log_transform:
        print(f"Applying log transformation to {target_col}")
        y_all = np.log(y_all)
    
    # Clip outliers in MEG data if requested
    if clip_outliers_flag:
        print(f"Clipping outliers in MEG data beyond {clip_std_threshold} standard deviations")
        # Create a copy to avoid modifying the original data
        clipped_meg_data = meg_data.copy()
        
        # Clip outliers for each channel separately
        for ch_idx in range(meg_data.shape[1]):
            for t_idx in range(meg_data.shape[2]):
                channel_data = meg_data[:, ch_idx, t_idx]
                clipped_meg_data[:, ch_idx, t_idx] = clip_outliers(channel_data, clip_std_threshold)
        
        # Replace original data with clipped data
        meg_data = clipped_meg_data
        
        # Log the effect of clipping
        clipped_ratio = np.sum(meg_data != clipped_meg_data) / meg_data.size
        print(f"Clipped {clipped_ratio:.4%} of MEG data points")
    
    # Loop through each timepoint
    for t_idx, t in enumerate(times):
        # Calculate remaining time for each epoch at this timepoint
        remaining_time = meta_df[target_col].values - t
        
        # Create mask for valid epochs (those with enough remaining time)
        valid_mask = remaining_time > time_to_end
        
        if np.sum(valid_mask) > 0:  # Check if there are any valid epochs
            # Extract MEG data for this timepoint
            X_t = meg_data[valid_mask, :, t_idx]
            
            # Extract target values for valid epochs
            y_t = y_all[valid_mask]
            
            # transform the durations into time left to end
            y_t = y_t - t
            
            # Store the data
            X_per_timepoint.append(X_t)
            y_per_timepoint.append(y_t)
            valid_epochs_per_timepoint.append(valid_mask)
        else:
            # No valid epochs at this timepoint
            X_per_timepoint.append(None)
            y_per_timepoint.append(None)
            valid_epochs_per_timepoint.append(valid_mask)
    
    return X_per_timepoint, y_per_timepoint, valid_epochs_per_timepoint

def train_timepoint_decoder(X, y, hyperparameter_search=False, alphas=None, n_folds=5, n_jobs=1, hp_cv_folds=3):
    """
    Train a Ridge regression decoder on a single timepoint.
    
    Parameters:
    - X: np.ndarray
        Feature matrix for this timepoint
    - y: np.ndarray
        Target vector for this timepoint
    - hyperparameter_search: bool
        Whether to perform hyperparameter search
    - alphas: np.ndarray
        Alpha values to try for Ridge regression
    - n_folds: int
        Number of cross-validation folds for final evaluation
    - n_jobs: int
        Number of parallel jobs
    - hp_cv_folds: int
        Number of CV folds for hyperparameter search
        
    Returns:
    - model: Ridge
        Trained Ridge regression model
    - score: float
        Cross-validation R² score
    - coef: np.ndarray
        Model coefficients
    - best_alpha: float
        Best alpha value found (only if hyperparameter_search=True)
    """
    if X is None or y is None or len(y) < n_folds:
        return None, np.nan, None, np.nan
    
    try:
        # Standardize features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform hyperparameter search if requested
        if hyperparameter_search and alphas is not None:
            # Ensure we don't use more folds than samples
            hp_cv = min(hp_cv_folds, len(y))
            
            # Use RidgeCV for efficient alpha selection
            model = ElasticNetCV(
                l1_ratio=0.5,  # Ridge regression
                alphas=alphas,
                cv=hp_cv,
                n_jobs=n_jobs
            )
            model.fit(X_scaled, y)
            best_alpha = model.alpha_
            
            # Print best alpha found
            # print(f"Best alpha: {best_alpha:.6f}")
        else:
           
            model = Ridge(alpha=0.1)
            
            model.fit(X_scaled, y)
            best_alpha = model.alpha_
        
        # Perform cross-validation for final evaluation
        cv = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2', n_jobs=n_jobs)
        mean_score = np.mean(cv_scores)
        
        return model, mean_score, model.coef_, best_alpha
    
    except Exception as e:
        print(f"Error in training: {e}")
        return None, np.nan, None, np.nan

def run_timepoint_decoder(meg_data, meta_df, times, time_to_end, target_col="duration", 
                         log_transform=False, hyperparameter_search=False, alphas=None,
                         n_folds=5, n_jobs=1, hp_cv_folds=3, 
                         clip_outliers_flag=False, clip_std_threshold=3):
    """
    Run a timepoint-by-timepoint decoder with the time_to_end constraint.
    
    Parameters:
    - meg_data: np.ndarray
        MEG data array of shape (n_epochs, n_channels, n_times)
    - meta_df: pd.DataFrame
        Metadata DataFrame containing duration information
    - times: np.ndarray
        Array of time points corresponding to the MEG data
    - time_to_end: float
        Minimum time (in seconds) that must remain in an epoch for it to be used
    - target_col: str
        Column name in meta_df containing the target values to predict
    - log_transform: bool
        Whether to apply log transformation to target values
    - hyperparameter_search: bool
        Whether to perform hyperparameter search
    - alphas: np.ndarray
        Alpha values to try for Ridge regression
    - n_folds: int
        Number of cross-validation folds for final evaluation
    - n_jobs: int
        Number of parallel jobs
    - hp_cv_folds: int
        Number of CV folds for hyperparameter search
    - clip_outliers_flag: bool
        Whether to clip outliers in the MEG data
    - clip_std_threshold: float
        Number of standard deviations beyond which to clip values
        
    Returns:
    - results: dict
        Dictionary containing decoder results
    """
    # Prepare data for each timepoint
    X_per_timepoint, y_per_timepoint, valid_epochs_per_timepoint = prepare_data_for_timepoint_decoder(
        meg_data, meta_df, times, time_to_end, target_col, log_transform, 
        clip_outliers_flag, clip_std_threshold)
    
    # Initialize results containers
    scores = np.zeros(len(times))
    n_samples = np.zeros(len(times))
    coefficients = [None] * len(times)
    models = [None] * len(times)
    best_alphas = np.zeros(len(times))
    
    # Train decoder for each timepoint
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_timepoint_decoder)(
            X_per_timepoint[t_idx], 
            y_per_timepoint[t_idx], 
            hyperparameter_search,
            alphas,
            n_folds, 
            1,  # Use 1 job within each parallel process
            hp_cv_folds
        ) for t_idx in tqdm(range(len(times)), desc="Training decoders")
    )
    
    # Unpack results
    for t_idx, (model, score, coef, best_alpha) in enumerate(results):
        scores[t_idx] = score
        if X_per_timepoint[t_idx] is not None:
            n_samples[t_idx] = len(y_per_timepoint[t_idx])
        coefficients[t_idx] = coef
        models[t_idx] = model
        best_alphas[t_idx] = best_alpha
    
    # Prepare results dictionary
    return {
        'times': times,
        'scores': scores,
        'n_samples': n_samples,
        'coefficients': coefficients,
        'models': models,
        'valid_epochs_per_timepoint': valid_epochs_per_timepoint,
        'best_alphas': best_alphas,
        'log_transform': log_transform,
        'clip_outliers': clip_outliers_flag,
        'clip_std_threshold': clip_std_threshold
    }

def plot_decoder_performance(results, title='Decoder Performance', save_path=None):
    """
    Plot the performance of the timepoint decoder.
    
    Parameters:
    - results: dict
        Dictionary containing decoder results
    - title: str
        Plot title
    - save_path: str or None
        Path to save the figure, if None the figure is displayed
    """
    times = results['times']
    scores = results['scores']
    n_samples = results['n_samples']
    

    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot R² scores
    ax1.plot(times, scores, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('R² Score')
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 0.10)
    
    # Plot number of samples
    ax2.plot(times, n_samples, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Number of Valid Samples')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight significance
    sig_times = times[scores > 0]
    if len(sig_times) > 0:
        ax1.axhspan(0, max(scores), xmin=np.min(sig_times)/np.max(times),
                   xmax=np.max(sig_times)/np.max(times), alpha=0.2, color='g')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_decoder_weights_over_time(results, channel_names=None, top_n=10, save_path=None):
    """
    Plot the evolution of decoder weights over time.
    
    Parameters:
    - results: dict
        Dictionary containing decoder results
    - channel_names: list or None
        List of channel names, if None uses generic names
    - top_n: int
        Number of top channels to highlight
    - save_path: str or None
        Path to save the figure, if None the figure is displayed
    """
    times = results['times']
    coefficients = results['coefficients']
    
    # Filter out None values
    valid_times = []
    valid_coeffs = []
    for t, coef in zip(times, coefficients):
        if coef is not None:
            valid_times.append(t)
            valid_coeffs.append(coef)
    
    if not valid_coeffs:
        print("No valid coefficients to plot")
        return
    
    # Convert to array
    valid_times = np.array(valid_times)
    weight_matrix = np.array(valid_coeffs)
    
    # Create channel names if not provided
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(weight_matrix.shape[1])]
    
    # Compute average absolute weights across time
    avg_abs_weights = np.mean(np.abs(weight_matrix), axis=0)
    
    # Find indices of top_n channels
    top_indices = np.argsort(avg_abs_weights)[-top_n:]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot weights for all channels
    plt.imshow(weight_matrix.T, aspect='auto', cmap='RdBu_r', 
              extent=[valid_times[0], valid_times[-1], 0, weight_matrix.shape[1]],
              interpolation='nearest')
    
    plt.colorbar(label='Weight')
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title('Decoder Weights Over Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    # Plot top channels
    plt.figure(figsize=(14, 8))
    for idx in top_indices:
        plt.plot(valid_times, weight_matrix[:, idx], label=channel_names[idx])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.title(f'Weights of Top {top_n} Channels Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_path_top = save_path.replace('.png', '_top_channels.png')
        plt.savefig(save_path_top, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_hyperparameter_selection(results, save_path=None):
    """
    Plot the hyperparameter selection results.
    
    Parameters:
    - results: dict
        Dictionary containing decoder results
    - save_path: str or None
        Path to save the figure, if None the figure is displayed
    """
    if 'best_alphas' not in results:
        print("No hyperparameter selection results to plot")
        return
    
    times = results['times']
    best_alphas = results['best_alphas']
    scores = results['scores']
    
    # Filter out invalid values
    valid_times = []
    valid_alphas = []
    valid_scores = []
    
    for t, alpha, score in zip(times, best_alphas, scores):
        if not np.isnan(alpha) and not np.isnan(score):
            valid_times.append(t)
            valid_alphas.append(alpha)
            valid_scores.append(score)
    
    if not valid_times:
        print("No valid hyperparameters to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot R² scores
    ax1.plot(valid_times, valid_scores, 'b-', linewidth=2)
    ax1.set_ylabel('R² Score')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Hyperparameter Selection Results')
    
    # Plot best alpha values (in log scale)
    ax2.semilogy(valid_times, valid_alphas, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Best Alpha (log scale)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_clipping_effect(original_data, clipped_data, clip_std_threshold, save_path=None):
    """
    Plot the effect of clipping outliers.
    
    Parameters:
    - original_data: np.ndarray
        Original MEG data
    - clipped_data: np.ndarray
        MEG data after clipping
    - clip_std_threshold: float
        Number of standard deviations used for clipping
    - save_path: str or None
        Path to save the figure, if None the figure is displayed
    """
    # Flatten the arrays for histogram
    original_flat = original_data.flatten()
    clipped_flat = clipped_data.flatten()
    
    # Calculate statistics
    original_mean = np.mean(original_flat)
    original_std = np.std(original_flat)
    clipped_mean = np.mean(clipped_flat)
    clipped_std = np.std(clipped_flat)
    
    # Create histogram plot
    plt.figure(figsize=(14, 8))
    
    # Plot histograms with logarithmic y-scale
    plt.hist(original_flat, bins=100, alpha=0.5, label=f'Original (μ={original_mean:.2f}, σ={original_std:.2f})', log=True)
    plt.hist(clipped_flat, bins=100, alpha=0.5, label=f'Clipped (μ={clipped_mean:.2f}, σ={clipped_std:.2f})', log=True)
    
    # Add vertical lines showing clipping thresholds
    clip_upper = original_mean + clip_std_threshold * original_std
    clip_lower = original_mean - clip_std_threshold * original_std
    plt.axvline(x=clip_upper, color='r', linestyle='--', label=f'+{clip_std_threshold}σ threshold')
    plt.axvline(x=clip_lower, color='r', linestyle='--', label=f'-{clip_std_threshold}σ threshold')
    
    plt.xlabel('MEG Signal Value')
    plt.ylabel('Count (log scale)')
    plt.title(f'Effect of Clipping Outliers Beyond {clip_std_threshold} Standard Deviations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate percentage of clipped values
    clipped_ratio = np.sum(original_data != clipped_data) / original_data.size
    plt.figtext(0.5, 0.01, f"Clipped {clipped_ratio:.4%} of data points", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Load metadata
    merged_df = load_data.merge_meta_df(EVENT_TYPE)
    
    # Load MEG data
    meg_data = load_data.process_meg_data_for_roi(
        CH_TYPE, EVENT_TYPE, SESSIONS, apply_median_scale=APPLY_MEDIAN_SCALE)
    
    # Load time points
    times = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    
    print(f"Loaded data: {meg_data.shape} epochs, {len(times)} timepoints")
    print(f"Metadata: {len(merged_df)} events")
    
    # Remove outlier durations
    longest_dur = np.percentile(merged_df[TARGET_COL], 98)
    shortest_dur = np.percentile(merged_df[TARGET_COL], 2)
    dur_mask = (merged_df[TARGET_COL] < longest_dur) & (merged_df[TARGET_COL] > shortest_dur)
    
    merged_df = merged_df[dur_mask]
    meg_data = meg_data[dur_mask, :, :]
    
    # Reset index after filtering
    merged_df.reset_index(drop=True, inplace=True)
    
    print(f"After duration filtering: {meg_data.shape[0]} epochs")
    
    # Create a copy of the original MEG data for comparison if clipping is enabled
    original_meg_data = None
    if CLIP_OUTLIERS:
        original_meg_data = meg_data.copy()
        
        # Create a clipped version of MEG data
        clipped_meg_data = meg_data.copy()
        
        # Clip outliers for each channel separately
        for ch_idx in range(meg_data.shape[1]):
            for t_idx in range(meg_data.shape[2]):
                channel_data = meg_data[:, ch_idx, t_idx]
                clipped_meg_data[:, ch_idx, t_idx] = clip_outliers(channel_data, CLIP_STD_THRESHOLD)
        
        # Replace original data with clipped data
        meg_data = clipped_meg_data
        
        # Create plots directory if it doesn't exist
        os.makedirs(os.path.join(PLOTS_DIR, "decoder"), exist_ok=True)
        
        # Plot the effect of clipping
        plot_clipping_effect(
            original_meg_data, 
            clipped_meg_data, 
            CLIP_STD_THRESHOLD,
            save_path=os.path.join(PLOTS_DIR, "decoder", f"clipping_effect_{SUBJECT_ID}_{EVENT_TYPE}_{CLIP_STD_THRESHOLD}std.png")
        )
    
    # Generate analysis suffix for file naming
    analysis_suffix = f"{SUBJECT_ID}_{EVENT_TYPE}_{TIME_TO_END}"
    if LOG_TRANSFORM:
        analysis_suffix += "_log"
    if HYPERPARAMETER_SEARCH:
        analysis_suffix += "_hp"
    if CLIP_OUTLIERS:
        analysis_suffix += f"_clip{CLIP_STD_THRESHOLD}std"
    
    # Run the timepoint decoder
    results = run_timepoint_decoder(
        meg_data, merged_df, times, TIME_TO_END, 
        target_col=TARGET_COL, 
        log_transform=LOG_TRANSFORM,
        hyperparameter_search=HYPERPARAMETER_SEARCH,
        alphas=ALPHAS,
        n_folds=N_FOLDS, 
        n_jobs=N_JOBS,
        hp_cv_folds=HP_CV_FOLDS,
        clip_outliers_flag=False,  # We've already clipped the data above
        clip_std_threshold=CLIP_STD_THRESHOLD
    )
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.join(PLOTS_DIR, "decoder"), exist_ok=True)
    
    # Plot decoder performance
    plot_title = f'Timepoint Decoder Performance (time_to_end={TIME_TO_END}s)'
    if LOG_TRANSFORM:
        plot_title += ', log-transformed'
    if HYPERPARAMETER_SEARCH:
        plot_title += ', with HP search'
    if CLIP_OUTLIERS:
        plot_title += f', outliers clipped at {CLIP_STD_THRESHOLD}σ'
        
    plot_decoder_performance(
        results, 
        title=plot_title,
        save_path=os.path.join(PLOTS_DIR, "decoder", f"decoder_performance_{analysis_suffix}.png")
    )
    
    # Plot decoder weights
    plot_decoder_weights_over_time(
        results,
        save_path=os.path.join(PLOTS_DIR, "decoder", f"decoder_weights_{analysis_suffix}.png")
    )
    
    # Plot hyperparameter selection results if applicable
    if HYPERPARAMETER_SEARCH:
        plot_hyperparameter_selection(
            results,
            save_path=os.path.join(PLOTS_DIR, "decoder", f"hyperparameter_selection_{analysis_suffix}.png")
        )
    
    # Save results
    np.save(
        os.path.join(PLOTS_DIR, "decoder", f"decoder_results_{analysis_suffix}.npy"),
        results
    )
    
    print("Decoder analysis completed!")