"""
Parameter definitions for neural dynamics analysis.
"""

# MEG data parameters
EVENT_TYPE = "fixation"
TIME_WINDOW = (-0.200, 0.500)  # in seconds
# DECIM = 2
remove_erfs = False#["saccade", "fixation"]

# Analysis parameters
CHANGE_METRIC = "correlation"  # "correlation", "cosine", "euclidean", "mahalanobis"
DELTA_T = 30  # time difference in ms
STRIDE = 1
N_JOBS = -2  # -1 for all cores, -2 for all but one
RANDOM_STATE = 42

# ROI definition
ROI_GROUPS = {
    #"midventral": ["midventral"],
    "early": ["early"],
    "(mid)lateral": ['midlateral', "lateral"],
    "(mid)parietal": ["midparietal", "parietal"],
    "(mid)ventral": ["midventral", "ventral"],
    # "HC": ["H"],
    # "FEF": ["FEF"],
    # "dlPFC": ["8C", "8Av", "i6-8", "s6-8", "SFL", "8BL", "9p", "9a", "8Ad", "p9-46v", "a9-46v", "46", "9-46d"],
    # "OFC": ["47s", "47m", "a47r", "11l", "13l", "a10p", "p10p", "10pp", "10d", "OFC", "pOFC"],
}
HEMI = "both"  # hemisphere, "lh" for left, "rh" for right, or "both"
# you might whant to run the dynamics analysis for both hemis separately, in that case set HEMI to "lh" or "rh". 
# To integrate over the hemispheres for plotting, set HEMI to "both" after the analysis is done.