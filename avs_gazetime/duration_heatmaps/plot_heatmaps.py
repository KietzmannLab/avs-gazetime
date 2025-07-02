"""This script is used to compute the per epooch/duration bin heatmaps and run the peak latency regression on them."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.signal import hilbert
import mne
import scipy

import avs_gazetime.utils.load_data as load_data
from avs_gazetime.utils.sensors_mapping import grads, mags
from avs_gazetime.utils.tools import compute_quantiles, get_quantile_data
from avs_machine_room.prepro.eye_tracking.avs_prep import avs_combine_events


from avs_gazetime.memorability.mem_tools import get_memorability_scores
from joblib import Parallel, delayed

from avs_gazetime.config import (
    S_FREQ,
    SESSIONS,
    SUBJECT_ID,
    ET_DIR,
    PLOTS_DIR,
    MEG_DIR,
    CH_TYPE)
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from tqdm import tqdm

from params_theta import (QUANTILES, EVENT_TYPE, BANDS as bands, DECIM as decim, MEM_SCORE_TYPE, PHASE_OR_POWER, remove_erfs)
from sklearn.model_selection import GroupShuffleSplit

import fracridge


from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
# import SVR
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
  
    
    
frontal_mag_sensors = [
    "MEG0121",
    "MEG0341",
    "MEG0811",
    "MEG0821",
    "MEG0541",
    "MEG0931",
    "MEG1411",
    "MEG1221",
]

occipital_mag_sensors = [
    "MEG1921",
    "MEG2341",
    "MEG2111",
    "MEG1731",
    "MEG2511",
]


def plot_heatmap(meg_data, offsets, times, output_dir, event_type, subject, channel, binned=True, band_name="", tlims=(-0.2, 0.500), 
                 pac_box=True, pac_box_params={"tmin": 0.15, "tmax": 0.4}):
    """
    Plot the heatmap for the population code across sessions. 
    The data can be pre-binned (duration quantiles) or not.
    """
    sns.set_context("poster")
    meg_data = meg_data[:, channel, :]
    
    print(meg_data.shape)
    # zscore the data per quantile
    # if binned:
    #     meg_data = (meg_data - np.nanmean(meg_data)) / np.nanstd(meg_data)
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    cbar_label = band_name

    # if band_name != "raw" or band_name != "PAC":
    #     meg_data = np.log(meg_data)
    if band_name == "PAC":
        pass
        cmap = "magma"
        vmin = np.percentile(meg_data, 0.01)
        vmax = np.percentile(meg_data, 99.99)
    else:
        # add PHASE_OR_POWER to the cbar label
        
        # log transform all values beyond 0
        if band_name != "raw":
            cbar_label += f" {PHASE_OR_POWER}"
            if PHASE_OR_POWER == "power":
                pos_mask = meg_data > 0
                meg_data[pos_mask] = np.log(meg_data[pos_mask])
                cmap = "magma"
                vmin = np.percentile(meg_data, 0.01)
                vmax = np.percentile(meg_data, 99.99)
            else:
                cmap = 'twilight_shifted'
                vmin = -np.pi
                vmax = np.pi
                
        
                
        else:
            cbar_label = "residual activation [fT]"
            cmap = "icefire"
            vmin = np.percentile(meg_data, 0.01)
            vmax = np.percentile(meg_data, 99.99)
            
  
   
    cax = ax.pcolormesh(times * 1000, np.arange(1, len(meg_data) + 1), meg_data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.plot(offsets * 1000, np.arange(1, len(meg_data) + 1), color="w", linestyle=":", linewidth=4)
    fig.colorbar(cax, ax=ax, label=cbar_label)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    ax.axvline(0, color="w", linestyle="--", linewidth=4)
    
    ax.set_xlim(tlims[0] * 1000, tlims[1] * 1000)
    # ax.set_xlabel("")
    ax.set_ylabel("fixation duration")
    ax.set_yticks([])
    ax.tick_params(length=0)
    fig = plt.gcf()
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    # despine the figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # add time label to the x axis
    ax.set_xlabel("time [ms]")
    
    # write fixation onset to the 0 ms line "horizontally"
    ax.text(0, 40, "fixation\nonset", color="white", fontsize=22, ha="center", va="center", rotation=90)
    
    if pac_box:
      
        # make the box (from the top of the y axis to bin with the clostest duration to pac_box_params["tmax"])
        # which is the last bin to be considered for the PAC box
        # get the closest bin to the tmax
        cuttoff_bin_index = np.argmin(np.abs(offsets - pac_box_params["tmax"]))
        # get the y axis value for the cuttoff bin
        cuttoff_bin_y = cuttoff_bin_index +2
        # add a box from the top of the y axis to the cuttoff bin
        ax.fill_betweenx([cuttoff_bin_y, len(meg_data)], pac_box_params["tmin"] * 1000, pac_box_params["tmax"] * 1000, color="gray", alpha=0.5)
        # add a frame to the box
        ax.plot([pac_box_params["tmin"] * 1000, pac_box_params["tmax"] * 1000], [cuttoff_bin_y, cuttoff_bin_y], color="white", linewidth=3)
        ax.plot([pac_box_params["tmin"] * 1000, pac_box_params["tmin"] * 1000], [cuttoff_bin_y, len(meg_data)], color="white", linewidth=3)
        ax.plot([pac_box_params["tmax"] * 1000, pac_box_params["tmax"] * 1000], [cuttoff_bin_y, len(meg_data)], color="white", linewidth=3)
        ax.plot([pac_box_params["tmin"] * 1000, pac_box_params["tmax"] * 1000], [len(meg_data), len(meg_data)], color="white", linewidth=3)
        # write PAC into the box in white font
        #ax.text((pac_box_params["tmin"] + pac_box_params["tmax"]) * 500, (cuttoff_bin_y + len(meg_data)) / 2, "PAC?", color="white", fontsize=22, ha="center")
        
    
    if not os.path.exists(os.path.join(output_dir, "heatmaps")):
        os.makedirs(os.path.join(output_dir, "heatmaps"))
    fname = f"epoch_heatmap_{event_type}_{subject}_{channel}_binned_{binned}_{band_name}"
    if PHASE_OR_POWER == "phase":
        fname = f"epoch_heatmap_{event_type}_{subject}_{channel}_binned_{binned}_{band_name}_phase"
    # plot the offsets
    
    fig.savefig(os.path.join(output_dir, "heatmaps", fname + ".png"), transparent=False)
    #fig.savefig(os.path.join(output_dir, "heatmaps", fname + ".pdf"))
    return

def freq_avg(data, method, axis=0):
    """
    Average frequency data using the specified method along the given axis.
    
    Parameters:
    - data: np.ndarray
        The frequency data to be averaged.
    - method: str
        The averaging method to use. Options are 'raw', 'power', or 'phase'.
        
    - axis: int
        The axis along which to average the data.
    
    Returns:
    - np.ndarray
        The averaged frequency data.
    """
    
    if method == 'raw' or method == 'power':
        return np.nanmedian(data, axis=axis)
    elif method == 'phase':
        # make sure data is radians
        if np.max(np.abs(data)) > np.pi:
            data = np.deg2rad(data)
        
        return np.angle(np.nanmedian(np.exp(1j * np.array(data)), axis=axis))
    else:
        raise ValueError("Invalid method. Choose from 'raw', 'power', or 'phase'.")

if __name__ == "__main__":
    # read the results frm all subjects
    # check the system arguments for a channel chunk to process (this is for the case of the full stc to be processed)
    # otherwise it will take too much RAM.
    import sys
    print(sys.argv)
    if len(sys.argv) > 1:
        # "14 is the first chunk of 4, 24 is the second chunk of 4 etc
        channel_chunk = sys.argv[1].split("_")
        
    else:
        channel_chunk = None 
    print("Processing channel chunk", channel_chunk)
    #assert CH_TYPE == "mag" # only mag for now
    
    if CH_TYPE == "stc" and channel_chunk is not None:
        
    #  check available channels
        #check_available_channels(session: str, roi: str, event_type: str, subject_name=None)
        channels = load_data.check_available_channels("a", CH_TYPE, EVENT_TYPE)
        # subselect the channels based on the chunk
        num_chunks = int(channel_chunk[1])
        chunk_idx = int(channel_chunk[0])
        chunk_size = len(channels) // num_chunks
        start_idx = chunk_idx * chunk_size
        end_idx = (chunk_idx + 1) * chunk_size
        if chunk_idx == num_chunks - 1:
            end_idx = len(channels)
        channels = [channels[start_idx:end_idx]]
        
    else:
    
        channels = [None]
        
    if remove_erfs:
        merged_df_saccade = load_data.merge_meta_df("saccade")
        # combine the dataframes
        merged_df_fixation = load_data.merge_meta_df("fixation")
        print(merged_df_saccade)
        merged_df = load_data.match_saccades_to_fixations(merged_df_fixation, merged_df_saccade, saccade_type="pre-saccade")
    print(channels)
    # iterate over the channels
    for channel_idx in channels:
            
        if remove_erfs:
            meg_data = load_data.process_meg_data_for_roi(CH_TYPE, "saccade", SESSIONS, apply_median_scale=True, channel_idx=channel_idx)
            
            # apply the index mask to the meg data
            print(len(merged_df), meg_data.shape)
            meg_data = meg_data[merged_df.index, :, :]
            # reset the index
            merged_df.reset_index(drop=True, inplace=True)
            
            
            # remove the sacades ERF 
            if "saccade" in remove_erfs:
                print("removing saccade erf")
                sacc_erf = np.median(meg_data, axis=0)
                meg_data = meg_data - sacc_erf
            
            # roll the data to the right by the duration of the saccade
            # get the duration of the saccade
            # describe the duration
            merged_df_saccade["duration"].describe()
            saccade_duration = merged_df_saccade["duration"].values
            
            n_shifts_per_event = (saccade_duration * S_FREQ).astype(int)
            
            # print("n shifts", n_shifts_per_event)
            
            # #mean_shift = np.percentile(n_shifts_per_event, 80)
            # #mean_shift = int(mean_shift)
            # #print("mean shift", mean_shift)
            # # roll the data to the right by the duration of the saccade (in parralel)
            meg_data = np.array([np.roll(meg_data[i], -n_shifts_per_event[i]) for i in range(meg_data.shape[0])])
            
            
            if "fixation" in remove_erfs:
                # remove the fixation ERF
                print("removing fixation erf")
                fix_erf = np.median(meg_data, axis=0)
                meg_data = meg_data - fix_erf
                
            # remove events without associated fixation duration
            print("removing events without associated fixation duration")
            print("before", len(merged_df))
            merged_df = merged_df[~np.isnan(merged_df["associated_fixation_duration"])]
            print("after", len(merged_df))
        else:
            merged_df = load_data.merge_meta_df(EVENT_TYPE)
            meg_data = load_data.process_meg_data_for_roi(CH_TYPE, EVENT_TYPE, SESSIONS, apply_median_scale=True, channel_idx=channel_idx)
    
        # this is a bit of a clumsy way to get the duration_pre, but it works for now
        #print(merged_df.columns)
        dur_col = "duration" if EVENT_TYPE == "fixation" else "associated_fixation_duration"
        
        
        
        print(len(merged_df), meg_data.shape)
        # exclude the longest and shortest durations
        #if channel_idx is not None and channel_idx > 0:
        #    pass
        #else:
        longest_dur = np.percentile(merged_df[dur_col], 99)
        shortest_dur = np.percentile(merged_df[dur_col],1)
        dur_mask = (merged_df[dur_col] < longest_dur) & (merged_df[dur_col] > shortest_dur)
        merged_df = merged_df[dur_mask]
        meg_data = meg_data[dur_mask, :, :]
        # reset the index
        merged_df.reset_index(drop=True, inplace=True)
        
        times = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
            


        #print("getting memorability scores")
        # get memorability scores and attach them to the merged_df
        #print(merged_df.columns)
        if EVENT_TYPE == "saccade":
            # change type column entires to fixation and fill start_time with associated_start_time
            merged_df["type"] = "fixation"
            merged_df["start_time"] = merged_df["associated_fix_start_time"]
            
    
        #merged_df = get_memorability_scores(merged_df, SUBJECT_ID, [MEM_SCORE_TYPE], "regression", crop_size_pix = 164)

        # plot the mean memorability scores per quantile
        

        
        # we will transform the trials based data into dur_col based quantiles medians
        # lets form 200 quantiles
        
        if QUANTILES:
            # reset the index
            print(merged_df)
            merged_df = compute_quantiles(merged_df, dur_col, QUANTILES)
            # sort the data by quantile
            meg_data = meg_data[merged_df.index, :, :]
            # pydebug here
            print(merged_df)
            # reset the index
            merged_df.reset_index(drop=True, inplace=True)
            # sort the data by quantile
            binned = True
        else:
            binned = False
                
    
        for band_name, tf_band in bands.items():
            
        
            if band_name == "raw" or band_name == "PAC":
                meg_data_band = meg_data
                #meg_data_band = np.clip(meg_data_band, np.percentile(meg_data_band, 5), np.percentile(meg_data_band, 95))
                # decimate the data
                if band_name == "raw":
                    meg_data_band = meg_data_band[:, :, ::decim]
                
            else:
                band_size = tf_band[1] - tf_band[0]
                steps = int(band_size / 2)
                freqs = np.arange(tf_band[0]+1, tf_band[1]+1, steps)
                
                meg_data_band = mne.time_frequency.tfr_array_morlet(meg_data, sfreq=500, freqs=freqs, n_cycles=2, output=PHASE_OR_POWER, n_jobs=-2, decim=decim)
                # use a multitaper approach
                #meg_data_band = mne.time_frequency.tfr_array_multitaper(meg_data, sfreq=500, freqs=freqs, n_cycles=2, output=phase_or_power, n_jobs=-1, decim=decim)
                print(meg_data_band.shape)

                meg_data_band = freq_avg(meg_data_band.data, PHASE_OR_POWER, axis=2)
                band_max = np.percentile(meg_data_band, 98)
                meg_data_band = np.clip(meg_data_band, None, band_max)
            
        
                
            times_band = times[::decim]
            frontal_mag_sensors_idx = [mags.index(sensor) for sensor in frontal_mag_sensors]
            occipital_mag_sensors_idx = [mags.index(sensor) for sensor in occipital_mag_sensors]
            
    
           
            durantions_per_quantile = merged_df.groupby("quantile")[dur_col].median().values
            
          
            meg_data_band_per_quantile = np.full((len(np.unique(merged_df["quantile"])), meg_data_band.shape[1], meg_data_band.shape[2]), fill_value=np.nan)
            def process_quantile(q):
                q_mask = merged_df["quantile"] == q
                print(f"Processing quantile {q} with {np.sum(q_mask)} trials")
                print(np.max(meg_data_band[q_mask, :, :]), np.min(meg_data_band[q_mask, :, :]), np.mean(meg_data_band[q_mask, :, :]))
                return freq_avg(meg_data_band[q_mask, :, :], PHASE_OR_POWER, axis=0)

            results = Parallel(n_jobs=-1)(delayed(process_quantile)(q) for q in np.unique(merged_df["quantile"]))
            for q, result in enumerate(results):
                meg_data_band_per_quantile[q, :, :] = result
            meg_data_band = meg_data_band_per_quantile
                    
            frontal_mag_sensors_idx = [mags.index(sensor) for sensor in frontal_mag_sensors]
            occipital_mag_sensors_idx = [mags.index(sensor) for sensor in occipital_mag_sensors]
            all_idx = np.arange(meg_data_band.shape[1])
            areas = ["frontal", "occipital",]
            channels = [frontal_mag_sensors_idx, occipital_mag_sensors_idx]
            for area, channel_sel in zip(areas, channels):
                for channel in tqdm(channel_sel, desc=f"Channels in {area}"):
                    print(f"Processing channel {channel}", area)
            
                    
                    plot_heatmap(meg_data=meg_data_band, times=times_band, channel=channel, output_dir=PLOTS_DIR, event_type=EVENT_TYPE, subject=SUBJECT_ID , band_name=band_name, offsets=durantions_per_quantile)
                    




