import sys
import os
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import mne
from joblib import Parallel, delayed
import logging

from avs_gazetime.utils.sensors_mapping import grads
from avs_gazetime.config import (
    SUBJECT,
    SESSIONS,
    PLOTS_DIR,
    MEG_DIR,
    SUBJECT_ID,
    DEBUG, 
    S_FREQ,
    
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_meg_filepath(session: str, event_type: str, subject_name=None) -> Path:
    """Get the file path for MEG data."""
    if subject_name:
        MEG_DIR_str = str(MEG_DIR)
        MEG_DIR_str = MEG_DIR_str.replace(SUBJECT, subject_name)
        return Path(MEG_DIR_str) / f"{subject_name}{session}_population_codes_{event_type}_500hz_masked_False.h5"
    else:
        return MEG_DIR / f"{SUBJECT}{session}_population_codes_{event_type}_500hz_masked_False.h5"

def get_meta_filepath(session: str, event_type: str, subject_name=None) -> Path:
    """Get the file path for metadata."""
    if subject_name:
        MEG_DIR_str = str(MEG_DIR)
        MEG_DIR_str = MEG_DIR_str.replace(SUBJECT, subject_name)
        return Path(MEG_DIR_str) / f"{subject_name}{session}_et_epochs_metadata_{event_type}.csv"
    else:
        if os.path.exists(MEG_DIR / f"{SUBJECT}{session}_et_epochs_metadata_{event_type}.csv"):
            return MEG_DIR / f"{SUBJECT}{session}_et_epochs_metadata_{event_type}.csv"
       
        else:
            print("File not found, trying to load from the sensor directory")
            meg_dir_sensor = Path(f"/share/klab/datasets/avs/population_codes/{SUBJECT}/sensor/erf/filter_0.2_200/ica/")
            return meg_dir_sensor / f"{SUBJECT}{session}_et_epochs_metadata_{event_type}.csv"
            

def load_meg_session_data(session: str, roi: str, event_type: str, channel_idx=None, subject_name=None) -> np.ndarray:
    """Load MEG data for a specific session and ROI."""
    # channel idx can be a list of indices or an array of indices as well as an integer or None
    # check if the channel_idx is a list or an array
    if isinstance(channel_idx, list) or isinstance(channel_idx, np.ndarray):
        channel_selection = "multiple"
        # Convert to numpy array if it's a list
        if isinstance(channel_idx, list):
            channel_idx = np.array(channel_idx)
    elif channel_idx == None:
        channel_selection = "all"
    else:
        channel_selection = "single"
    print("Channel selection", channel_selection)
    
    file_path = get_meg_filepath(session, event_type, subject_name)
    print("loading data from", file_path)
    
    with h5py.File(file_path, "r") as h5_file:
        if channel_selection == "all":
            data = h5_file[roi]["onset"][:]
        elif channel_selection == "single":
            data = h5_file[roi]["onset"][:, channel_idx]
        elif channel_selection == "multiple":
            # Create sorted indices for h5py indexing
            original_indices = np.arange(len(channel_idx))
            sorted_idx_pairs = sorted(zip(channel_idx, original_indices))
            sorted_channel_idx, restoration_order = zip(*sorted_idx_pairs)
            
            # Load data using sorted indices
            sorted_data = h5_file[roi]["onset"][:, sorted_channel_idx]
            
            # Restore original order of channels
            data = sorted_data[:, np.argsort(restoration_order)]
    
    print("data shape", data.shape)
    if channel_selection == "single":
        data = data[:, np.newaxis, :]
    return data

def check_available_channels(session: str, roi: str, event_type: str, subject_name=None) -> list:
    """Check the available channels for a given session."""
    file_path = get_meg_filepath(session, event_type, subject_name)
    # check the size of the data and use the second dimension as the number of channels
    with h5py.File(file_path, "r") as h5_file:
        shape_data = h5_file[roi]["onset"].shape
        return list(range(shape_data[1]))

def median_scale(data: np.ndarray, with_std=False, session=None) -> np.ndarray:
    """Session-wise median scaling of MEG data."""
    logging.info(f"Median scaling data per sensor of shape {data.shape}, with_std={with_std}, session={session}")
    scaler = mne.decoding.Scaler(scalings="median", with_std=with_std)
    return scaler.fit_transform(data)

def process_meg_data_for_roi(roi: str, event_type: str, sessions: list = SESSIONS, apply_median_scale=True, scale_with_std=False, channel_idx=None, subject_name=None, sensors_to_femto = True) -> np.ndarray:
    """Process MEG data for a given ROI across all sessions."""
    sessions_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sessions = [sessions_letters[i-1] for i in sessions]
    logging.info(f"Processing MEG data for ROI {roi} and event type {event_type} for sessions {sessions}")
    
    def load_and_scale(session):
        data = load_meg_session_data(session, roi, event_type, channel_idx=channel_idx, subject_name=subject_name)
        if DEBUG:
            return data
        if apply_median_scale:
            return median_scale(data, session=session, with_std=scale_with_std)
        return data
    
    transformed_sessions_data = Parallel(n_jobs=-1)(delayed(load_and_scale)(session) for session in sessions)
    concatenated_sessions_data = np.concatenate(transformed_sessions_data, axis=0)
    if sensors_to_femto:
        if roi in ["grad", "mag"]:
            print(f"Scaling {roi} data to femtoTesla")
            concatenated_sessions_data = scale_grad_or_mag_data(concatenated_sessions_data, roi)
    
    return concatenated_sessions_data

def scale_grad_or_mag_data(data: np.ndarray, grad_or_mag: str) -> np.ndarray:
    """Scale the data to femtoTesla."""
    scale_factor = 1e13 if grad_or_mag == "grad" else 1e15 if grad_or_mag == "mag" else None
    if scale_factor is None:
        raise ValueError("grad_or_mag must be 'grad' or 'mag'")
    return data * scale_factor

def merge_meta_df(event_type: str, sessions=SESSIONS, subject_name=None) -> pd.DataFrame:
    """Merge metadata for all sessions."""
    sessions_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sessions_to_read = [sessions_letters[i-1] for i in sessions] if sessions is not None else sessions_letters
    merged_df = pd.DataFrame()

    for session in sessions_to_read:
        file_path = get_meta_filepath(session, event_type, subject_name)
        print("loading metadata from", file_path)
        df = pd.read_csv(file_path, sep=";")
        merged_df = pd.concat([merged_df, df])

    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def read_hd5_timepoints(event_type: str) -> np.ndarray:
    """Get timepoints for MEG data."""
    file_path = get_meg_filepath("a", event_type)
    with h5py.File(file_path, "r") as h5_file:
        return h5_file.attrs["times"]

def adjust_meg_path_for_grand_average(subject: str) -> Path:
    """Adjusts the MEG directory path for a given subject to be used in grand average analysis."""
    MEG_DIR_str = str(MEG_DIR)
    MEG_DIR_str = MEG_DIR_str.replace(f"as{SUBJECT_ID:02d}", f"as{subject:02d}")
    return Path(MEG_DIR_str)

def get_grand_average_meg_data(event_type: str, roi: str, apply_median_scale=True, all_channels=False, ga_subjects=[1, 2, 3, 4, 5]) -> np.ndarray:
    """Load the data for all subjects and sessions. Return MEG data of shape (n_samples, n_sensors, n_timepoints)."""
    grand_average_data = []
    for subject in ga_subjects:
        subject_name = f"as{subject:02d}"
        logging.info(f"Loading data for subject {subject_name}")
        data = process_meg_data_for_roi(roi, event_type, apply_median_scale=apply_median_scale, all_channels=all_channels, subject_name=subject_name)
        grand_average_data.append(data)
    
    return np.concatenate(grand_average_data, axis=0)

def get_grand_average_metadata(event_type: str, sessions=SESSIONS, ga_subjects=[1, 2, 3, 4, 5]) -> pd.DataFrame:
    """Load the metadata for all subjects and sessions. Return metadata of shape (n_samples, n_sensors, n_timepoints)."""
    grand_average_metadata = []
    for subject in ga_subjects:
        subject_name = f"as{subject:02d}"
        logging.info(f"Loading metadata for subject {subject_name}")
        metadata = merge_meta_df(event_type, sessions, subject_name=subject_name)
        grand_average_metadata.append(metadata)
    
    return pd.concat(grand_average_metadata, axis=0)




def match_saccades_to_fixations(
    saccades_meta_df, fixations_meta_df, saccade_type="post-saccade" # pre-saccade
):
    """ 
    Match saccades to fixations based on the time difference between the end of a saccade and the start of a fixation.
    We also transfer the fixation sequence number and the start_time of the linked fixation event
    The saccade type can be either pre-saccade or post-saccade
    pre-saccade: the fixation is before the saccade
    post-saccade: the fixation is after the saccade
    """
    print(
        "Number of unique sceneIDs in saccades and fixations dataframes:",
        saccades_meta_df["sceneID"].nunique(),
        fixations_meta_df["sceneID"].nunique(),
    )
    combined_df = pd.concat([fixations_meta_df, saccades_meta_df], axis=0)

    sceneIDs_with_inconsistent_order = []
    time_differences = []
    saccades_followed_by_fixations = 0
    fixations_followed_by_saccades = 0
    num_events_with_0_time_difference = 0
    saccades_followed_by_saccades = 0
    fixations_followed_by_fixations = 0
    fixations_count = 0
    saccades_count = 0

    selected_saccades_rows = []

    unique_sceneIDs = saccades_meta_df["sceneID"].unique()

    for sceneID in unique_sceneIDs:
        scene_group = combined_df[combined_df["sceneID"] == sceneID]
        sorted_group = scene_group.sort_values(by="start_time")

        types = sorted_group["type"].values
        in_alternating_order = all(
            types[i] != types[i + 1] for i in range(len(types) - 1)
        )
        if not in_alternating_order:
            sceneIDs_with_inconsistent_order.append(sceneID)
        fixations_count += types[types == "fixation"].shape[0]
        saccades_count += types[types == "saccade"].shape[0]

        for i in range(len(types) - 1):
            if types[i] == "saccade" and types[i + 1] == "fixation":
                saccades_followed_by_fixations += 1

                if saccade_type == "pre-saccade":
                    saccade_end_time = sorted_group.iloc[i]["end_time"]
                    fixation_start_time = sorted_group.iloc[i + 1]["start_time"]

                    time_difference = fixation_start_time - saccade_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i].to_dict()
                        saccade_row_data["original_index"] = sorted_group.index[i]
                        following_fixation_sequence = sorted_group.iloc[i + 1][
                            "fix_sequence"
                        ]
                        following_start_time = sorted_group.iloc[i + 1]["start_time"]
                        saccade_row_data["associated_fix_sequence"] = (
                            following_fixation_sequence
                        )
                        saccade_row_data["associated_fix_start_time"] = following_start_time
                        
                        # add the duration of the fixation
                        saccade_row_data["associated_fixation_duration"] = sorted_group.iloc[i+1]["duration"]
                        selected_saccades_rows.append(saccade_row_data)

            if types[i] == "fixation" and types[i + 1] == "saccade":

                fixations_followed_by_saccades += 1

                if saccade_type == "post-saccade":
                    fixation_end_time = sorted_group.iloc[i]["end_time"]
                    saccade_start_time = sorted_group.iloc[i + 1]["start_time"]

                    time_difference = saccade_start_time - fixation_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i + 1].to_dict()
                        saccade_row_data["original_index"] = sorted_group.index[i + 1]
                        preceding_fixation_sequence = sorted_group.iloc[i][
                            "fix_sequence"
                        ]
                        saccade_row_data["associated_fix_sequence"] = (
                            preceding_fixation_sequence
                        )
                        preceding_start_time = sorted_group.iloc[i]["start_time"]
                        saccade_row_data["associated_fixation_duration"] = sorted_group.iloc[i]["duration"]
                        
                        saccade_row_data["associated_fix_start_time"] = preceding_start_time
                        selected_saccades_rows.append(saccade_row_data)
                        
                        

            if types[i] == "saccade" and types[i + 1] == "saccade":
                saccades_followed_by_saccades += 1
            if types[i] == "fixation" and types[i + 1] == "fixation":
                fixations_followed_by_fixations += 1

    selected_saccades_df = pd.DataFrame(selected_saccades_rows)
    selected_saccades_df.set_index("original_index", inplace=True)

    print("Total Fixations:", fixations_count)
    print("Total Saccades:", saccades_count)
    print("Saccades followed by fixations:", saccades_followed_by_fixations)
    print("Fixations followed by saccades:", fixations_followed_by_saccades)    
    #print("Time differences:", time_differences)
    print(
        "Events 0 time difference:",
        num_events_with_0_time_difference,
    )

    diff = (
        selected_saccades_df["associated_fix_sequence"]
        - selected_saccades_df["sac_sequence"]
    ).to_list()

    print("Saccade and fixation sequence number difference")
    print({value: diff.count(value) for value in set(diff)})

    print("Fixations followed by saccades:", fixations_followed_by_saccades)
    print("Saccades followed by saccades:", saccades_followed_by_saccades)
    print("Fixations followed by fixations:", fixations_followed_by_fixations)
    print("SceneIDs with inconsistent order:", len(sceneIDs_with_inconsistent_order))
    
    print("Selected saccades df shape", selected_saccades_df.shape)

    return selected_saccades_df


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
        timepoints = read_hd5_timepoints(event_type=EVENT_TYPE)
    
        
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
        timepoints = read_hd5_timepoints(event_type=EVENT_TYPE)
    fix_onset_idx = sac_onset_idx + int((saccade_duration * 1000) / (1000 / S_FREQ))
    return fix_onset_idx

def get_fix_shift_per_trial(meta_df, dur_col,EVENT_TYPE=None):
    """
    Calculate fixation shifts and onsets per trial.

    Parameters:
    meta_df : pandas.DataFrame
        DataFrame containing metadata for each trial. Each row represents a trial.
    dur_col : str
        Column name in `meta_df` that contains the duration values for each trial.

    Returns:
    numpy.ndarray: Array of fixation shifts for each trial.
    numpy.ndarray: Array of fixation onset indices for each trial.
    """
    timepoints = read_hd5_timepoints(event_type=EVENT_TYPE)
    sac_onset_idx = get_idx_saccade_onset(timepoints,EVENT_TYPE)
    
    fix_shifts = np.zeros((len(meta_df)))
    fix_onsets = np.zeros((len(meta_df)))
    for idx in meta_df.index:
        this_trial = meta_df.iloc[idx]
        fix_onset_idx = get_idx_fix_onset(sac_onset_idx, this_trial[dur_col], timepoints)
        diff_fix_from_sac = fix_onset_idx - sac_onset_idx
        fix_shifts[idx] = diff_fix_from_sac
        fix_onsets[idx] = fix_onset_idx
    return fix_shifts, fix_onsets


# create a function that source projects the loaded data per subject. It needs the data per session and can only operate on sensor data