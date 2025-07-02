# make imports
import os
import pandas as pd
import numpy as np

from avs_gazetime.config import (
    S_FREQ,
    SESSIONS,
    SUBJECT_ID,
    ET_DIR,
    PLOTS_DIR,
    PLOTS_DIR_BEHAV,
    MEG_DIR,
    PROJECT_DIR,)

def get_memorability_scores(metadata_cross_session, subject, targets, model_task, crop_size_pix = 164):
    """
    Retrieves memorability labels for the given metadata.

    Args:
        metadata_cross_session (pd.DataFrame): Metadata containing information about fixations.
        subject (int): Subject identifier.
        targets (list): List of target labels to retrieve.
        model_task (str): Model task, either "classification" or "regression".
        crop_size_pix (int, optional): Size of the crop in pixels. Defaults to 164.

    Returns:
        pd.DataFrame: Metadata with memorability labels added.

    Raises:
        None
    """
    
    # Use the same directory structure as memorability_analysis
    memscore_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_scores")
    
    # Load memorability data using same naming convention
    memscore_file = os.path.join(memscore_dir, f"as{str(subject).zfill(2)}_crops_metadata_with_memscore_{crop_size_pix}.csv")
    
    if not os.path.exists(memscore_file):
        raise FileNotFoundError(f"Memorability file not found: {memscore_file}")
    
    metadata_with_resmem = pd.read_csv(memscore_file, low_memory=False)
    print(f"Loaded memorability data from: {memscore_file}")
    print(f"Columns available: {metadata_with_resmem.columns.tolist()}")
    
    # merge the only the resmem score with the metadata from the current analysis
    # merge on duration, sceneID, trial, and subject, session, 
    print(f"Original metadata length: {len(metadata_cross_session)}")
    print(f"Memorability data length: {len(metadata_with_resmem)}")
    print(f"Metadata columns: {metadata_cross_session.columns.tolist()}")
    
    # Check for duplicates in metadata_cross_session
    merge_columns = ["start_time", "end_time", "duration", "sceneID", "trial", "subject", "session", "type", "recording", "block"]
    
    # Only use merge columns that exist in both dataframes
    available_merge_cols = [col for col in merge_columns if col in metadata_cross_session.columns and col in metadata_with_resmem.columns]
    print(f"Using merge columns: {available_merge_cols}")
    
    # Select only necessary columns from memorability data
    mem_columns = available_merge_cols + ["mem_score"]
    metadata_with_resmem_subset = metadata_with_resmem[mem_columns]
    
    metadata_cross_session = pd.merge(
        metadata_cross_session,
        metadata_with_resmem_subset, 
        on=available_merge_cols,
        how="left",
        validate="one_to_one"
    )
    
    print(f"After merge length: {len(metadata_cross_session)}")
    print(f"Missing memorability scores: {metadata_cross_session['mem_score'].isna().sum()}")

    if model_task == "classification":
        if "memorability" in targets:
            metadata_cross_session["memorability"] = pd.qcut(metadata_cross_session["mem_score"], 4, [1,2,3,4])
            # make the dtype integer
            #drop nan values
            # only take qartiles 1 and 4 set all other values to nan
            metadata_cross_session["memorability"] = metadata_cross_session["memorability"].astype(float)

            metadata_cross_session.loc[(metadata_cross_session["memorability"] != 1) & (metadata_cross_session["memorability"] != 4), "memorability"] = np.nan

            # how many fixations do we have with and without memorability scores
            print(metadata_cross_session["memorability"].value_counts())
            
        if "mem_relative" in targets:
            
            # this would be the z-score of the memorability score of the current fixation relative to all fixations on the same scene
            metadata_cross_session["mem_relative"] = metadata_cross_session.groupby(["subject", "sceneID"])["mem_score"].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
            # we binarize this. memorability score > 0 = 1, memorability score < 0 = 0
            metadata_cross_session["mem_relative"] = metadata_cross_session["mem_relative"].apply(lambda x: 1 if x > 0 else 4)
            
    elif model_task == "regression":
        metadata_cross_session["memorability"] = metadata_cross_session["mem_score"]
        # z-score the memorability score
        metadata_cross_session["memorability"] = (metadata_cross_session["memorability"] - np.nanmean(metadata_cross_session["memorability"])) / np.nanstd(metadata_cross_session["memorability"])
        print(f"Memorability stats: {metadata_cross_session['memorability'].describe()}")
        
        if "mem_pre" in targets:
            # this will add the memorability score of the previous fixation of a given sequence to the current fixation (for the first fixation of a sequence,  this would naturally be nan)
            for subject in metadata_cross_session["subject"].unique():
                for sceneID in metadata_cross_session["sceneID"].unique():
                    scene_metadata = metadata_cross_session[(metadata_cross_session["subject"] == subject) & (metadata_cross_session["sceneID"] == sceneID)]
                    # get the indices of the events
                    metadata_indices = scene_metadata.index
                    # add the previous memorability score to the metadata
                    metadata_cross_session.loc[metadata_indices, "mem_pre"] = scene_metadata["memorability"].shift(1)
                    
        if "mem_post" in targets:
            # this will add the memorability score of the next fixation of a given sequence to the current fixation (for the last fixation of a sequence,  this would naturally be nan)
            metadata_cross_session["mem_post"] = metadata_cross_session.groupby(["subject", "sceneID", "trial", "fix_sequence"])["memorability"].shift(-1)
            
        if "mem_relative" in targets:
            # this would be the z-score of the memorability score of the current fixation relative to all fixations on the same scene
            metadata_cross_session["mem_relative"] = metadata_cross_session.groupby(["subject", "sceneID"])["memorability"].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
            
        if "mem_scene" in targets:
            metadata_cross_session["mem_scene"] = metadata_cross_session.groupby(["subject", "sceneID"])["memorability"].transform(lambda x: np.nanmean(x))
    
    # describe the memorability scores
    print(f"Final memorability stats: {metadata_cross_session['memorability'].describe()}")
    
    return metadata_cross_session