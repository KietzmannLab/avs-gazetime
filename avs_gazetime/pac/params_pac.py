QUANTILES = 80 # 80
EVENT_TYPE = "saccade"
TIME_WINDOW =(0.150, 0.400) # in seconds
THETA_BAND =(3, 8)
GAMMA_BAND =(40, 140)
PAC_METHOD = "modulation_index" #"pac_ozkurt", #"circular_linear_correlation"
PHASE_OR_POWER = "power"
DECIM = 1
remove_erfs = ["saccade", "fixation","saccade_post", "fixation_post"]
# surrogate data generation method
SURROGATE_STYLE = "single_cut"  # Options: "phase_shuffle", "session_aware", "single_cut"
N_BOOTSTRAPS = 200

#cross-frequency PAC params
FF_BAND_AMPLITUDE = (40, 160)
FF_BAND_PHASE = (1, 15)
THETA_STEPS = 1
GAMMA_STEPS = 15

# Memorability-based epoch splitting
MEM_SPLIT = "40/40"  # Options: None, "50/50", "25/25", "33/33", etc. Format: "bottom_percent/top_percent"
MEM_CROP_SIZE = 100  # Crop size in pixels for memorability analysis (must match precomputed scores)

# Duration-based epoch splitting
DURATION_SPLIT = None  # Threshold in ms (e.g., 350, 400) or None to disable
DURATION_BALANCE = True  # Balance sample sizes between short/long groups
OFFSET_LOCKED = False  # If True, time-lock to fixation offset (end) instead of onset
