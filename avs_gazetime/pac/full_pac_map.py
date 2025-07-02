#!/usr/bin/env python3
"""
Streamlined script for visualizing phase-amplitude coupling (PAC) matrices 
across subjects, channels, and ROIs. Creates 1x5 figure with individual subjects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from matplotlib.gridspec import GridSpec
from avs_gazetime.config import (
    S_FREQ,
    SESSIONS,
    SUBJECT_ID,
    ET_DIR,
    PLOTS_DIR,
    MEG_DIR,
    CH_TYPE)

from params_pac import (QUANTILES, EVENT_TYPE, DECIM,
                        MEM_SCORE_TYPE, PHASE_OR_POWER, remove_erfs, 
                        TIME_WINDOW, THETA_BAND, GAMMA_BAND, PAC_METHOD,
                        THETA_STEPS, GAMMA_STEPS, FF_BAND_PHASE, FF_BAND_AMPLITUDE, SURROGATE_STYLE)


def restrict_pac_matrix(pac_data, theta_freqs, gamma_freqs, phase_range=None, amplitude_range=None):
    """
    Restrict a PAC matrix to specific frequency ranges on phase and amplitude axes.
    
    Parameters:
    -----------
    pac_data : np.ndarray
        The PAC matrix with shape (len(theta_freqs), len(gamma_freqs))
    theta_freqs : np.ndarray
        Array of phase frequencies
    gamma_freqs : np.ndarray
        Array of amplitude frequencies
    phase_range : tuple or None
        (min, max) frequency range for phase axis (theta)
    amplitude_range : tuple or None
        (min, max) frequency range for amplitude axis (gamma)
        
    Returns:
    --------
    restricted_matrix : np.ndarray
        PAC matrix restricted to specified frequency ranges
    restricted_theta_freqs : np.ndarray
        Restricted array of phase frequencies
    restricted_gamma_freqs : np.ndarray
        Restricted array of amplitude frequencies
    """
    # Create masks for frequency ranges
    theta_mask = np.ones_like(theta_freqs, dtype=bool)
    gamma_mask = np.ones_like(gamma_freqs, dtype=bool)
    
    # Apply phase frequency range restriction
    if phase_range is not None:
        theta_mask = (theta_freqs >= phase_range[0]) & (theta_freqs <= phase_range[1])
        # Check if mask is empty
        if not np.any(theta_mask):
            raise ValueError(f"No phase frequencies found in range {phase_range}. Available range: {min(theta_freqs)}-{max(theta_freqs)} Hz")
    
    # Apply amplitude frequency range restriction
    if amplitude_range is not None:
        gamma_mask = (gamma_freqs >= amplitude_range[0]) & (gamma_freqs <= amplitude_range[1])
        # Check if mask is empty
        if not np.any(gamma_mask):
            raise ValueError(f"No amplitude frequencies found in range {amplitude_range}. Available range: {min(gamma_freqs)}-{max(gamma_freqs)} Hz")
    
    # Extract restricted frequency arrays
    restricted_theta_freqs = theta_freqs[theta_mask]
    restricted_gamma_freqs = gamma_freqs[gamma_mask]
    print(pac_data.shape)
    # Extract restricted PAC matrix
    restricted_matrix = pac_data[np.ix_(theta_mask, gamma_mask)]
    
    return restricted_matrix, restricted_theta_freqs, restricted_gamma_freqs


def plot_pac_heatmap(pac_data, theta_freqs, gamma_freqs, output_dir, title, 
                     filename, phase_range=None, amplitude_range=None,
                     cmap='magma', vmin=None, vmax=None, smooth=True):
    """
    Plot a PAC heatmap.
    
    Parameters:
    -----------
    pac_data : np.ndarray
        PAC matrix data (phase frequencies x amplitude frequencies)
    theta_freqs : np.ndarray
        Phase frequency range
    gamma_freqs : np.ndarray
        Amplitude frequency range
    output_dir : str
        Directory to save the output image
    title : str
        Plot title (can be None for no title)
    filename : str
        Filename for the saved plot
    phase_range : tuple or None
        (min, max) frequency range for phase axis (theta)
    amplitude_range : tuple or None
        (min, max) frequency range for amplitude axis (gamma)
    cmap : str, optional
        Colormap name
    vmin, vmax : float, optional
        Color limits
    smooth : bool, optional
        Whether to apply bicubic interpolation
    """
    sns.set_context("poster")
    
    # Restrict frequency ranges if specified
    if phase_range is not None or amplitude_range is not None:
        pac_data, theta_freqs, gamma_freqs = restrict_pac_matrix(
            pac_data, theta_freqs, gamma_freqs, phase_range, amplitude_range)
    
    # Auto-set color scale limits if not specified
    if vmax is None:
        vmax = np.percentile(pac_data, 98)
    if vmin is None:
        vmin = 0
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Plot the heatmap
    if smooth:
        cax = ax.imshow(pac_data.T, extent=(theta_freqs[0], theta_freqs[-1], gamma_freqs[0], gamma_freqs[-1]),
                    origin='lower', cmap=cmap, interpolation='bicubic', aspect='auto', vmin=vmin, vmax=vmax)
    else:
        cax = ax.imshow(pac_data.T, extent=(theta_freqs[0], theta_freqs[-1], gamma_freqs[0], gamma_freqs[-1]), 
                    origin='lower', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, label='PAC [z-score]')
    cbar.outline.set_visible(False)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add labels with frequency range information if specified
    if phase_range:
        phase_label = f"frequency for phase [{phase_range[0]}-{phase_range[1]} Hz]"
    else:
        phase_label = "frequency for phase [Hz]"
        
    if amplitude_range:
        amp_label = f"frequency for amplitude [{amplitude_range[0]}-{amplitude_range[1]} Hz]"
    else:
        amp_label = "frequency for amplitude [Hz]"
    
    ax.set_xlabel(phase_label)
    ax.set_ylabel(amp_label)
    
    # Add title with range information if specified
    if title == True:
        if phase_range or amplitude_range:
            range_info = ""
            if phase_range:
                range_info += f"Phase: {phase_range[0]}-{phase_range[1]}Hz"
            if amplitude_range:
                if range_info:
                    range_info += ", "
                range_info += f"Amplitude: {amplitude_range[0]}-{amplitude_range[1]}Hz"
            title = f"{title}\n{range_info}"
    
        ax.set_title(title)
    elif isinstance(title, str):
        ax.set_title(title)
    
    # Save figure
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add frequency range information to filename if specified
    if phase_range or amplitude_range:
        base_filename, ext = os.path.splitext(filename)
        if phase_range:
            base_filename += f"_phase_{phase_range[0]}-{phase_range[1]}"
        if amplitude_range:
            base_filename += f"_amp_{amplitude_range[0]}-{amplitude_range[1]}"
        filename = base_filename + ext
    
    fig.savefig(os.path.join(output_dir, filename), transparent=False)
    plt.close(fig)




def plot_roi_average(subjects, base_dir, output_dir, roi_names, hemisphere='both',
                    ch_type='stc', event_type=EVENT_TYPE, 
                    theta_band=FF_BAND_PHASE, gamma_band=FF_BAND_AMPLITUDE, 
                    time_window=TIME_WINDOW, theta_steps=THETA_STEPS, gamma_steps=GAMMA_STEPS,
                    phase_range=None, amplitude_range=None):
    """
    Plot average PAC across all vertices in specified ROIs.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs (integers)
    base_dir : str
        Base directory path
    output_dir : str
        Directory to save output plots
    roi_names : list
        List of ROI names to process
    hemisphere : str
        'lh', 'rh', or 'both'
    ch_type : str
        Channel type (typically 'stc' for source space)
    event_type : str
        Event type (saccade, fixation)
    theta_band : tuple
        (min, max) for phase frequency
    gamma_band : tuple
        (min, max) for amplitude frequency
    time_window : tuple
        (start, end) time window in seconds
    theta_steps, gamma_steps : int
        Step size for frequency bands
    phase_range : tuple or None
        Optional restriction for phase frequency range
    amplitude_range : tuple or None
        Optional restriction for amplitude frequency range
    """
    print(f"Plotting ROI average PAC for {len(subjects)} subjects")
    
    # Create frequency arrays
    theta_freqs = np.arange(theta_band[0], theta_band[1], theta_steps)
    gamma_freqs = np.arange(gamma_band[0], gamma_band[1], gamma_steps)
    
    # Create hemispheres list
    if hemisphere == 'both':
        hemispheres = ['lh', 'rh']
    else:
        hemispheres = [hemisphere]
    
    # Process each ROI
    for roi in roi_names:
        print(f"\nProcessing ROI: {roi}")
        
        for hemi in hemispheres:
            print(f"  Hemisphere: {hemi}")
            
            # Collect matrices across subjects and all vertices
            roi_matrices = []
            vertices_per_subject = {}
            
            for subject in subjects:
                sub_name = f"as{subject:02d}"
                
                # Set up paths
                if base_dir:
                    this_base_dir = base_dir.replace(f"as{SUBJECT_ID:02d}", sub_name)
                else:
                    # Use config paths
                    this_base_dir = PLOTS_DIR.replace(f"as{SUBJECT_ID:02d}", sub_name)
                
                # ROI path pattern
                roi_dir = f"{this_base_dir}/roi_ffpac_{subject}_{ch_type}_{event_type}_{theta_band[0]}-{theta_band[1]}_{gamma_band[0]}-{gamma_band[1]}_{time_window[0]}-{time_window[1]}_{SURROGATE_STYLE}"
                roi_pattern = f"{roi_dir}/{roi}_{hemi}/vertex_*_ffmap_{SURROGATE_STYLE}.npy"
                
                # Find all vertex files
                vertex_files = glob.glob(roi_pattern)
                vertices_per_subject[subject] = len(vertex_files)
                print(vertex_files)
                print(f"    Subject {subject}: Found {len(vertex_files)} vertices")
                
                # Load all vertex files
                for f in vertex_files:
                    # Load the PAC matrix
                    data = np.load(f, allow_pickle=True)
                    print(f"    Loading {f}")
                    #print(data)
                    # Check if the data is a dictionary
                    if isinstance(data, dict):
                        # get the key and extract the data
                        key = list(data.keys())[0]
                        data = data[key]
                        
                    # # Restrict matrix if necessary
                    # if phase_range is not None or amplitude_range is not None:
                    #     data, _, _ = restrict_pac_matrix(
                    #         data, theta_freqs, gamma_freqs, phase_range, amplitude_range)
                    
                    roi_matrices.append(data)
                    print(f"    Loaded matrix from {f}")
                    
            
            # Compute average if matrices were found
            if roi_matrices:
                print(f"  Computing average across {len(roi_matrices)} matrices for {roi}_{hemi}")
                # Report vertices per subject
                for subject, count in vertices_per_subject.items():
                    print(f"    Subject {subject}: {count} vertices")
                
                roi_matrices = np.array(roi_matrices)
                avg_roi_pac = np.median(roi_matrices, axis=0)
                
                # Plot
                for smooth in [True, False]:
                    filename = f"pac_{roi}_{hemi}_smooth_{smooth}.png"
                    title = SURROGATE_STYLE
                    
                    plot_pac_heatmap(
                        avg_roi_pac, theta_freqs, gamma_freqs, 
                        output_dir, title, filename,
                        phase_range=phase_range,
                        amplitude_range=amplitude_range,
                        smooth=smooth
                    )
            else:
                print(f"  No matrices found for {roi}_{hemi}")

def plot_pac_by_subject(subjects, base_dir, output_dir, roi_names, hemisphere='both',
                      ch_type='stc', event_type=EVENT_TYPE,
                      theta_band=FF_BAND_PHASE, gamma_band=FF_BAND_AMPLITUDE,
                      time_window=TIME_WINDOW, theta_steps=THETA_STEPS, gamma_steps=GAMMA_STEPS,
                      phase_range=None, amplitude_range=None, smooth=True, figsize=(20, 6),
                      cmap='magma', vmin=None, vmax=None):
    """
    Plot PAC matrices separately for each subject for specified ROIs.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs (integers)
    base_dir : str
        Base directory path
    output_dir : str
        Directory to save output plots
    roi_names : list
        List of ROI names to process
    hemisphere : str
        'lh', 'rh', or 'both'
    ch_type : str
        Channel type (typically 'stc' for source space)
    event_type : str
        Event type (saccade, fixation)
    theta_band : tuple
        (min, max) for phase frequency
    gamma_band : tuple
        (min, max) for amplitude frequency
    time_window : tuple
        (start, end) time window in seconds
    theta_steps, gamma_steps : int
        Step size for frequency bands
    phase_range : tuple or None
        Optional restriction for phase frequency range
    amplitude_range : tuple or None
        Optional restriction for amplitude frequency range
    smooth : bool
        Whether to apply bicubic interpolation in plots
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for heatmaps
    vmin, vmax : float or None
        Min and max values for colormap
    """
    print(f"Plotting PAC matrices for {len(subjects)} subjects and {len(roi_names)} ROIs")
    
    # Create frequency arrays
    theta_freqs = np.arange(theta_band[0], theta_band[1], theta_steps)
    gamma_freqs = np.arange(gamma_band[0], gamma_band[1], gamma_steps)
    
    # Create hemispheres list
    if hemisphere == 'both':
        hemispheres = ['lh', 'rh']
    else:
        hemispheres = [hemisphere]
    # set context for seaborn
    sns.set_context("poster")
    # Process each ROI and hemisphere
    for roi in roi_names:
        for hemi in hemispheres:
            print(f"\nProcessing {roi}_{hemi}")
            
            # Dictionary to store median PAC matrices per subject
            subject_matrices = {}
            
            # First pass: load all matrices to determine global color scaling
            all_matrices = []
            
            for subject in subjects:
                sub_name = f"as{subject:02d}"
                
                # Set up paths
                if base_dir:
                    this_base_dir = base_dir.replace(f"as{SUBJECT_ID:02d}", sub_name)
                else:
                    # Use config paths
                    this_base_dir = PLOTS_DIR.replace(f"as{SUBJECT_ID:02d}", sub_name)
                
                # ROI path pattern
                roi_dir = f"{this_base_dir}/roi_ffpac_{subject}_{ch_type}_{event_type}_{theta_band[0]}-{theta_band[1]}_{gamma_band[0]}-{gamma_band[1]}_{time_window[0]}-{time_window[1]}_{SURROGATE_STYLE}"
                roi_pattern = f"{roi_dir}/{roi}_{hemi}/vertex_*_ffmap_{SURROGATE_STYLE}.npy"
                
                # Find all vertex files
                vertex_files = glob.glob(roi_pattern)
                print(f"  Subject {subject}: Found {len(vertex_files)} vertices")
                
                # Skip if no vertices found
                if not vertex_files:
                    continue
                    
                # Load all vertex matrices for this subject
                subject_vertex_matrices = []
                for f in vertex_files:
                    data = np.load(f, allow_pickle=True)
                    
                    # Check if the data is a dictionary
                    if isinstance(data, dict):
                        key = list(data.keys())[0]
                        data = data[key]
                    
                    subject_vertex_matrices.append(data)
                
                # Compute median across vertices for this subject
                if subject_vertex_matrices:
                    subj_median = np.median(subject_vertex_matrices, axis=0)
                    subject_matrices[subject] = subj_median
                    all_matrices.append(subj_median)
            
            # Skip ROI if no data found
            if not all_matrices:
                print(f"  No data found for {roi}_{hemi}, skipping")
                continue
                
            # Calculate global vmin/vmax if not provided
            all_data = np.concatenate([m.flatten() for m in all_matrices])
            if vmax is None:
                vmax = np.percentile(all_data, 98)
            if vmin is None:
                vmin = np.max([0, np.percentile(all_data, 2)])
            
            print(f"  Color scale: {vmin:.2f} to {vmax:.2f}")
                
            # Create a figure with one subplot per subject
            fig = plt.figure(figsize=figsize, dpi=300)
            gs = GridSpec(1, len(subjects) + 1, width_ratios=[1]*len(subjects) + [0.05])
            
            # Plot each subject's matrix
            for i, subject in enumerate(subjects):
                if subject not in subject_matrices:
                    # Create empty subplot for missing subjects
                    ax = fig.add_subplot(gs[0, i])
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_title(f"Subject {subject}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Get the matrix
                matrix = subject_matrices[subject]
                
                # Restrict matrix if necessary
                if phase_range is not None or amplitude_range is not None:
                    matrix, theta_freqs_restricted, gamma_freqs_restricted = restrict_pac_matrix(
                        matrix, theta_freqs, gamma_freqs, phase_range, amplitude_range)
                else:
                    theta_freqs_restricted, gamma_freqs_restricted = theta_freqs, gamma_freqs
                
                # Create subplot
                ax = fig.add_subplot(gs[0, i])
                
                # Plot the heatmap
                if smooth:
                    im = ax.imshow(matrix.T, extent=(theta_freqs_restricted[0], theta_freqs_restricted[-1], 
                                                 gamma_freqs_restricted[0], gamma_freqs_restricted[-1]),
                              origin='lower', cmap=cmap, interpolation='bicubic', 
                              aspect='auto', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(matrix.T, extent=(theta_freqs_restricted[0], theta_freqs_restricted[-1],
                                                 gamma_freqs_restricted[0], gamma_freqs_restricted[-1]),
                              origin='lower', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                
                # Set title and labels
                ax.set_title(f"Subject {subject}")
                
                # Only add y-label on the first subplot
                if i == 0:
                    if amplitude_range:
                        amp_label = f"Amplitude freq. [{amplitude_range[0]}-{amplitude_range[1]} Hz]"
                    else:
                        amp_label = "Amplitude freq. [Hz]"
                    ax.set_ylabel(amp_label)
                else:
                    ax.set_yticks([])
                
                # Add x-label on all plots
                if phase_range:
                    phase_label = f"Phase freq. [{phase_range[0]}-{phase_range[1]} Hz]"
                else:
                    phase_label = "Phase freq. [Hz]"
                if i == len(subjects) - 1:
                    ax.set_xlabel(phase_label)
                #ax.set_xlabel(phase_label)
                
                # Clean up spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Add a colorbar
            cbar_ax = fig.add_subplot(gs[0, -1])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label('PAC [z-score]')
            
            # Add an overall title
            plt.suptitle(f"{roi}_{hemi} - PAC ({SURROGATE_STYLE})", y=0.95)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create filename with frequency restrictions if applicable
            filename = f"pac_by_subject_{roi}_{hemi}_smooth_{smooth}"
            if phase_range:
                filename += f"_phase_{phase_range[0]}-{phase_range[1]}"
            if amplitude_range:
                filename += f"_amp_{amplitude_range[0]}-{amplitude_range[1]}"
            filename += ".png"
            
            # Save the figure
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved figure to {os.path.join(output_dir, filename)}")
            
if __name__ == "__main__":
    # Example usage - modify as needed
    subjects = [1, 2, 3, 4, 5]
    output_dir = f"{PLOTS_DIR}/pac_visualization/{FF_BAND_PHASE[0]}-{FF_BAND_PHASE[1]}_{FF_BAND_AMPLITUDE[0]}-{FF_BAND_AMPLITUDE[1]}_{TIME_WINDOW[0]}-{TIME_WINDOW[1]}_{SURROGATE_STYLE}"
    
    # Optional frequency range restrictions
    phase_range = (1, 15)        # Restrict phase to theta band (4-8 Hz)
    amplitude_range = (30, 190) # Restrict amplitude to high gamma (30-190 Hz)
    
    # Process each ROI and create 1x5 figure with individual subjects
    roi_names = ["HC", "infFC", "dlPFC", "FEF", "OFC", "early"]
    

    
    # Also plot average PAC across subjects for each ROI
    plot_roi_average(
        subjects=subjects,
        base_dir=None,  # Use config paths
        output_dir=output_dir,
        roi_names=roi_names,
        hemisphere='both',  # Process both hemispheres
        phase_range=phase_range,
        amplitude_range=amplitude_range
    )
    
    # Plot PAC matrices for each subject
    plot_pac_by_subject(
        subjects=subjects,
        base_dir=None,  # Use config paths
        output_dir=output_dir,
        roi_names=roi_names,
        hemisphere='both',  # Process both hemispheres
        phase_range=phase_range,
        amplitude_range=amplitude_range
    )