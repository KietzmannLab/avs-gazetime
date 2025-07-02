# AVS-GazeTime: MEG + Eye-tracking Analysis Code

Why we linger: Memory encoding, rather than visual processing demand, drives fixation timing on natural scenes - evidence from a large-scale MEG dataset (Sulewski et al., in press).

Analysis code for investigating fixation duration control during natural scene viewing using MEG and eye-tracking data.


## Overview

This repository contains the Python analysis pipeline for our study examining how fixation durations vary during natural scene exploration. The code tests two competing hypotheses about fixation control: processing-demand vs memory-facilitation.

## Repository Structure

```
avs_gazetime/
├── config.py                          # Configuration and paths
├── utils/
│   ├── load_data.py                   # Core data loading functions
│   ├── sensors_mapping.py             # MEG sensor definitions
│   └── tools.py                       # Utility functions
├── ease-of-recognition/               # Visual complexity analysis
│   ├── avs_ease_of_recognition.py     # ResNet50 classification entropy
│   ├── plot_entropy_examples.py       # Visualisation scripts
│   ├── get_netwok_activations.py      # Neural network feature extraction
│   └── run_network_extraction.sh      # SLURM batch script
├── memorability/                      # ResMem memorability analysis
│   ├── memorability_analysis.py       # Memorability vs duration analysis
│   ├── compute_memorability_scores.py # ResMem score computation
│   └── run_compute_memorability_scores.sh
├── fix2cap/                           # Fixation-to-caption analysis
│   ├── fix2cap_quality_controls.py    # Quality control and statistics
│   ├── fix2cap_prepare_data_package.py # Data preparation
│   ├── setup.py                       # Package setup
│   └── README.txt                     # Data access instructions
├── pac/                               # Phase-amplitude coupling
│   ├── cross_frequency_pac.py         # Theta-gamma PAC computation
│   ├── full_pac_map.py               # Full frequency PAC mapping
│   └── run_pac_cross_freq.sh         # SLURM batch processing
├── decoding/                          # Multivariate pattern analysis
│   ├── memorability_decoder.py        # ResMem score decoding from MEG
│   ├── duration_decoder.py           # Duration prediction (not used)
│   ├── decoding_params.py            # Analysis parameters
│   └── run_memorability_decoder.sh   # SLURM script
└── duration_heatmaps/                # Visualisation tools
    └── plot_heatmaps.py              # MEG heatmap plotting
```

## Core Analysis Modules

### Configuration (`config.py`)
Central configuration managing:
- Subject IDs and session parameters
- MEG data paths (sensor vs source space)
- Eye-tracking data directories
- Analysis output directories

### Data Loading (`utils/load_data.py`)
Core functions for loading and preprocessing:
```python
# Load MEG data for specific ROI and event type
meg_data = process_meg_data_for_roi(roi="mag", event_type="fixation")

# Load corresponding metadata
metadata = merge_meta_df(event_type="fixation", sessions=SESSIONS)

# Load session-specific data
session_data = load_meg_session_data(session="a", roi="mag", event_type="fixation")
```

### Visual Complexity Analysis (`ease-of-recognition/`)
Tests processing-demand hypothesis using ResNet50 classification entropy:
- **`avs_ease_of_recognition.py`**: Main analysis computing entropy scores for fixated patches
- **`get_netwok_activations.py`**: Extracts neural network features from image crops
- **`run_network_extraction.sh`**: SLURM job for parallel processing across subjects

### Memorability Analysis (`memorability/`)
Tests memory-facilitation hypothesis using ResMem predictions:
- **`memorability_analysis.py`**: Correlates memorability scores with fixation durations
- **`compute_memorability_scores.py`**: Generates ResMem predictions for fixated patches
- Implements both absolute and relative (within-scene) memorability measures

### Fixation-to-Caption Matching (`fix2cap/`)
Analyses semantic relevance of fixated content:
- **`fix2cap_quality_controls.py`**: Statistical analysis of caption inclusion
- **`fix2cap_prepare_data_package.py`**: Prepares data for annotation interface
- Uses human ratings to classify fixations as mentioned/not mentioned in captions

### Phase-Amplitude Coupling (`pac/`)
Examines theta-gamma coupling during longer fixations:
- **`cross_frequency_pac.py`**: Computes PAC using single-cut surrogate method
- **`full_pac_map.py`**: Generates comprehensive frequency × frequency PAC maps
- Focuses on 3-7 Hz (theta) × 40-80 Hz (gamma) coupling

### Multivariate Decoding (`decoding/`)
Tests information content in MEG patterns:
- **`memorability_decoder.py`**: Decodes ResMem scores from MEG sensor patterns
- **`decoding_params.py`**: Ridge regression and cross-validation parameters
- Uses sliding window approach across time

## Key Dependencies

```python
# Core scientific computing
import numpy as np
import pandas as pd
import scipy

# MEG analysis
import mne
from mne.decoding import SlidingEstimator

# Machine learning
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

# Deep learning (for memorability)
import torch
import torchvision

# Statistical analysis
import statsmodels.formula.api as smf
```

## Usage Examples

### Basic Data Loading
```python
from avs_gazetime.config import configure_run
from avs_gazetime.utils.load_data import process_meg_data_for_roi, merge_meta_df

# Configure for subject 1, magnetometer data
config = configure_run(subject_id=1, ch_type="mag")

# Load MEG data and metadata
meg_data = process_meg_data_for_roi("mag", "fixation")
metadata = merge_meta_df("fixation")
```

### Running SLURM Jobs

All major analyses use SLURM array jobs for parallel processing:

```bash
# Compute ResMem memorability scores (subjects 1-5)
sbatch avs_gazetime/memorability/run_compute_memorability_scores.sh

# Extract neural network activations for classification entropy
sbatch avs_gazetime/ease-of-recognition/run_network_extraction.sh 100  # crop size

# Run theta-gamma PAC analysis
sbatch avs_gazetime/pac/run_pac_cross_freq.sh

# Decode memorability from MEG patterns
sbatch avs_gazetime/decoding/run_memorability_decoder.sh
```

**SLURM Environment Variables:**
- `SUBJECT_ID_GAZETIME`: Subject ID (1-5) set automatically by array jobs
- `CH_TYPE_GAZETIME`: Channel type ("mag", "grad", "stc")

**Local Analysis:**
```bash
# Run memorability analysis after computing scores
python avs_gazetime/memorability/memorability_analysis.py

# Quality control for fixation-caption matching
python avs_gazetime/fix2cap/fix2cap_quality_controls.py
```

## Data Structure

The code expects MEG data in HDF5 format:
```
{subject}{session}_population_codes_{event_type}_500hz_masked_False.h5
```

Metadata in CSV format:
```
{subject}{session}_et_epochs_metadata_{event_type}.csv
```

## Contact

For questions about the code:
- Philip Sulewski: psulewski{att}uos.de
