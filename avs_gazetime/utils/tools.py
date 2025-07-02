

import numpy as np
import pandas as pd

import avs_gazetime.utils.load_data as load_data
from avs_gazetime.config import (
    S_FREQ,)


import numpy as np
import pandas as pd

def compute_quantiles(merged_df, dur_col, quantiles):
    """
    Compute quantiles for the given duration column and add them as a new column to the DataFrame.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        The input DataFrame containing the duration column.
    dur_col : str
        The name of the duration column.
    quantiles : int
        The number of quantiles to compute.

    Returns:
    --------
    
    merged_df : pd.DataFrame
    """
    if quantiles > len(merged_df):
        raise ValueError("Number of quantiles cannot be greater than the number of rows in the DataFrame.")
    epoch_per_bin = len(merged_df) // quantiles
    
    merged_df = merged_df.sort_values(dur_col)
    fake_quantiles = np.full(len(merged_df), -1)
    for q in range(quantiles):
        start_idx = q * epoch_per_bin
        end_idx = start_idx + epoch_per_bin
        fake_quantiles[start_idx:end_idx] = q
    # Handle any remaining rows
    fake_quantiles[end_idx:] = q
    merged_df["quantile"] = fake_quantiles
    print(merged_df["quantile"].value_counts())
    return merged_df

        
def get_quantile_data(merged_df, grad_data, dur_col, quantiles):
    """
    Get quantile data for the given DataFrame and MEG data.
    Parameters:
    -----------
    merged_df : pd.DataFrame
        The input DataFrame containing the duration column.
    grad_data : np.ndarray
        The MEG data array.
    dur_col : str
        The name of the duration column.
    quantiles : int
        The number of quantiles to compute.
    Returns:
    --------
    grad_data_quantiles : np.ndarray
        The quantile-based MEG data.
    merged_df_quantiles : pd.DataFrame
        The DataFrame with quantile-based durations.
    """
    
    merged_df = compute_quantiles(merged_df, dur_col, quantiles)
    grad_data = grad_data[merged_df.index, :, :]
    quantiles = merged_df["quantile"].values
    grad_data_quantiles = np.zeros((len(np.unique(quantiles)), grad_data.shape[1], grad_data.shape[2]))
    merged_df_quantiles = pd.DataFrame(columns=merged_df.columns)
    for q_count, q in enumerate(np.unique(quantiles)):
        grad_data_quantiles[q_count, :, :] = np.median(grad_data[quantiles == q, :, :], axis=0)
        mean_dur = np.mean(merged_df[dur_col][quantiles == q])
        new_row = pd.DataFrame({dur_col: [mean_dur]})
        new_row.index = [q_count]
        merged_df_quantiles = pd.concat([merged_df_quantiles, new_row], axis=0)
    return grad_data_quantiles, merged_df_quantiles


def get_idx_saccade_onset(timepoints=None, EVENT_TYPE=None):
    """
    Get the index of the saccade onset from the given timepoints.

    Parameters:
    timepoints (array-like, optional): An array of timepoints. If not provided,
                                    the function will load timepoints using
                                    `load_data.read_hd5_timepoints()`.

    Returns:
    int: The index of the saccade onset in the timepoints array.

    Raises:
    IndexError: If no timepoint equal to zero is found in the array.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    return np.where(timepoints == 0)[0][0]

def get_idx_fix_onset(sac_onset_idx=None, saccade_duration=None, timepoints=None, EVENT_TYPE=None):
    """
    Calculate the index of the fixation onset based on the saccade onset index and saccade duration.

    Parameters:
    sac_onset_idx (int, optional): The index of the saccade onset. Defaults to None.
    saccade_duration (float, optional): The duration of the saccade in seconds. Defaults to None.
    timepoints (array-like, optional): The timepoints data. If None, it will be loaded using `load_data.read_hd5_timepoints()`. Defaults to None.

    Returns:
    int: The index of the fixation onset.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    fix_onset_idx = sac_onset_idx + int((saccade_duration * 1000) / (1000 / S_FREQ))
    return fix_onset_idx


def get_idx_from_time(time, timepoints=None, EVENT_TYPE=None):
    """
    Get the index of the closest timepoint to the given time.

    Parameters:
    time (float): The time value to find the closest index for.
    timepoints (array-like, optional): An array of timepoints to search within. 
                                    If None, the function will load timepoints 
                                    using load_data.read_hd5_timepoints().

    Returns:
    int: The index of the closest timepoint to the given time.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    return np.abs(timepoints - time).argmin()


def interpolate(n, start, end):
    """
    Interpolates `n` evenly spaced values between `start` and `end`.

    Parameters:
    n (int): The number of values to interpolate.
    start (float): The starting value of the interpolation range.
    end (float): The ending value of the interpolation range.

    Returns:
    numpy.ndarray: An array of `n` interpolated and rounded values.
    """
    return np.round(np.linspace(start, end, n + 2)[1:-1])
