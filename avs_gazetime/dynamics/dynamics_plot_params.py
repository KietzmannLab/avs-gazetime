"""
Parameters for dynamics plotting and analysis.
"""

# Analysis settings
ANALYSIS_TYPE = "halfway"  # "halfway", "peak_latency", or "peak_amplitude"
FILTER_DYNAMICS = 50  # Set to a cutoff frequency (e.g., 50) to enable lowpass filtering
Z_SCORE_PER_SUBJECT = False  # Whether to z-score dynamics per subject
DURATION_QUANTILES = ["very short", "short", "medium", "long"]  # Quantile labels
N_BOOT = 100  # Number of bootstrap iterations for significance testing
# Plotting parameters
PLOT_TIME_WINDOW = (-0.100, 0.350)  # Time window for dynamics plots (in seconds)
PEAK_WINDOW = (0.050, 0.250)  # Time window for peak detection (in seconds)
MAX_HALFWAY_WINDOW = 0.100  # Maximum time after peak to search for halfway point
MAX_POST_FIXATION = 0.150  # Maximum time after fixation end for valid analysis points

# Subject settings
SUBJECTS = ["01", "02", "03", "04", "05",]
