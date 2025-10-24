"""
Tools for loading classification entropy scores from network activations.
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.special import softmax
from avs_gazetime.config import PLOTS_DIR_BEHAV


def classification_entropy(features, apply_softmax=True):
    """
    Compute classification entropy from network activations.

    Parameters:
    -----------
    features : np.ndarray
        Network activations (n_samples, n_features)
    apply_softmax : bool
        Whether to apply softmax normalization

    Returns:
    --------
    entropy : np.ndarray
        Classification entropy values
    """
    if apply_softmax:
        features = softmax(features, axis=1)

    # Avoid log(0) by adding small epsilon
    features = np.maximum(features, np.finfo(float).eps)
    return -np.sum(features * np.log(features), axis=1)


def load_network_activations(subject_id, crop_size_pix, model_name="resnet50_ecoset_crop", module_name="fc"):
    """
    Load network activations for a given subject and module.

    Parameters:
    -----------
    subject_id : int
        Subject ID (1-5)
    crop_size_pix : int
        Crop size in pixels
    model_name : str
        Name of the neural network model
    module_name : str
        Name of the module/layer

    Returns:
    --------
    features : np.ndarray
        Network features
    filenames_df : pd.DataFrame
        DataFrame with filenames and feature mappings
    """
    subject_str = f"as{subject_id:02d}"
    activations_dir = f"{PLOTS_DIR_BEHAV}/crop_activations/{crop_size_pix}px"
    activations_path = os.path.join(activations_dir, subject_str, model_name, module_name)

    # Load features from HDF5 file
    features_file = None
    for file in os.listdir(activations_path):
        if file.endswith('.hdf5'):
            features_file = os.path.join(activations_path, file)
            break

    if not features_file:
        raise FileNotFoundError(f"No HDF5 file found in {activations_path}")

    with h5py.File(features_file, "r") as f:
        features = f["features"][:]

    # Load filenames
    txt_fname = os.path.join(activations_dir, subject_str, model_name, "file_names.txt")
    if not os.path.exists(txt_fname):
        # Look for any text file in the activations directory
        txt_files = [f for f in os.listdir(activations_path) if f.endswith('.txt')]
        if txt_files:
            txt_fname = os.path.join(activations_path, txt_files[0])
        else:
            raise FileNotFoundError(f"No filename mapping found for {subject_str}")

    with open(txt_fname, "r") as f:
        filenames_full = [line.strip() for line in f.readlines()]

    # Parse filenames to extract metadata
    filenames = [os.path.splitext(os.path.basename(filename))[0].split("_")
                for filename in filenames_full]
    filenames = [[int(x) for x in filename] for filename in filenames]

    # Determine column names based on filename structure
    if len(filenames[0]) == 4:
        colnames = ["subject", "trial", "fix_sequence", "sceneID"]
    elif len(filenames[0]) == 5:
        colnames = ["subject", "trial", "fix_sequence", "start_time", "sceneID"]
    else:
        raise ValueError(f"Unexpected filename structure: {filenames[0]}")

    filenames_df = pd.DataFrame(filenames, columns=colnames)
    filenames_df["full_filename"] = filenames_full

    return features, filenames_df


def get_entropy_scores(metadata_df, subject_id, targets, crop_size_pix=100,
                       model_name="resnet50_ecoset_crop", module_name="fc"):
    """
    Add classification entropy scores to metadata dataframe.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Metadata dataframe with 'crop_filename' column
    subject_id : int
        Subject ID (1-5)
    targets : list
        List of target variables to add (e.g., ["entropy", "entropy_relative"])
    crop_size_pix : int
        Crop size in pixels
    model_name : str
        Name of the neural network model
    module_name : str
        Name of the module/layer

    Returns:
    --------
    metadata_df : pd.DataFrame
        Dataframe with entropy scores added
    """
    print(f"Loading entropy scores for subject {subject_id}...")

    # Load network activations
    features, filenames_df = load_network_activations(
        subject_id, crop_size_pix, model_name, module_name
    )

    # Compute classification entropy
    entropy_scores = classification_entropy(features)
    filenames_df["entropy_raw"] = entropy_scores

    # Merge with metadata using crop_filename
    metadata_df = pd.merge(
        metadata_df,
        filenames_df[["full_filename", "entropy_raw"]],
        how="left",
        left_on="crop_filename",
        right_on="full_filename"
    )

    # Add requested targets
    for target in targets:
        if target == "entropy":
            # Absolute entropy (z-scored per subject)
            metadata_df["entropy"] = (
                metadata_df["entropy_raw"] - metadata_df["entropy_raw"].mean()
            ) / metadata_df["entropy_raw"].std()

        elif target == "entropy_relative":
            # Scene-relative entropy (z-scored within scene)
            metadata_df["entropy_relative"] = metadata_df.groupby("sceneID")["entropy_raw"].transform(
                lambda x: (x - x.mean()) / x.std()
            )

    # Clean up temporary columns
    metadata_df = metadata_df.drop(columns=["full_filename"], errors="ignore")

    print(f"Added entropy scores. Available targets: {targets}")
    print(f"Missing entropy scores: {metadata_df['entropy_raw'].isna().sum()}/{len(metadata_df)}")

    return metadata_df
