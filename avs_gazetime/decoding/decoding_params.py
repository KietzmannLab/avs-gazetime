"""
Parameters for memorability decoding analysis.
"""
import numpy as np

# Event and data parameters
EVENT_TYPE = "fixation"
DECIM = 1
APPLY_MEDIAN_SCALE = True

# Cross-validation parameters
N_FOLDS = 5
N_JOBS = -2

# Memorability parameters
MEMORABILITY_TARGETS = [ "mem_relative",] #"memorability"
CROP_SIZE_PIX = 100  # Use 100px memorability estimates
MODEL_TASK = "regression"  # Use regression for continuous memorability scores

# Hyperparameter settings
HYPERPARAMETER_SEARCH = True
ALPHAS = np.logspace(-3, 10, 7)
HP_CV_FOLDS = 3

# Data preprocessing
LOG_TRANSFORM = False
CLIP_OUTLIERS = True
CLIP_STD_THRESHOLD = 5
SCALER = "robust"
LOWPASS_FILTER = False # boxcar filter at 30Hz

# Temporal generalization settings
TEMPORAL_GENERALIZATION = False
TG_DECIM = 5  # Additional decimation for temporal generalization

# Optimization parameters
USE_PCA = False
PCA_VARIANCE_THRESHOLD = 0.95  # Retain 95% of variance

# Parameters for ElasticNetCV (not used in this script)
L1_RATIOS = [0.01, 0.25, 0.5]
ALPHAS = np.logspace(-3, 1, 10)
USE_SCENE_GROUPS = True