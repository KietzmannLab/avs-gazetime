

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
    fake_quantiles[end_idx:] = quantiles - 1
    merged_df["quantile"] = fake_quantiles
    print(merged_df["quantile"].value_counts())
    return merged_df

def comopute_quantiles_legacy(merged_df, dur_col, QUANTILES):
    """this is the old version of the function"""
    epoch_per_bin = len(merged_df) // QUANTILES
    merged_df = merged_df.sort_values(dur_col)
    fake_quantiles = np.zeros(len(merged_df))
    for q in range(QUANTILES):
        fake_quantiles[q*epoch_per_bin:(q+1)*epoch_per_bin] = q
    merged_df["quantile"] = fake_quantiles
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
        mean_dur = np.median(merged_df[dur_col][quantiles == q])
        new_row = pd.DataFrame({dur_col: [mean_dur]})
        new_row.index = [q_count]
        merged_df_quantiles = pd.concat([merged_df_quantiles, new_row], axis=0)
    return grad_data_quantiles, merged_df_quantiles


# contrast the two compute_quantiles functions
if __name__ == "__main__":
    from avs_saccade_locking.config import SESSIONS, SUBJECT_ID, MEG_DIR, PLOTS_DIR
    import os
    EVENT_TYPE = "saccade"
    from avs_saccade_locking.utils.load_data import merge_meta_df
    merged_df = merge_meta_df(EVENT_TYPE, sessions=SESSIONS)
    #merged_df = pd.DataFrame({"duration": np.random.rand(3000)})
    dur_col = "duration"
    quantiles = 160
    merged_df1 = compute_quantiles(merged_df.copy(), dur_col, quantiles)
    merged_df2 = comopute_quantiles_legacy(merged_df.copy(), dur_col, quantiles)
    # how many observations are in each quantile

    print(merged_df2["quantile"].value_counts())
    # plot the observations per quantile
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.histplot(merged_df1["quantile"], bins=quantiles,label="new")
    sns.histplot(merged_df2["quantile"], bins=quantiles,label="old") 
    plt.legend()
    #save
    plt.savefig(os.path.join(PLOTS_DIR, "quantiles_comparison.png"))
    

    