# Parameters for memorability score analysis
import numpy as np

# Analysis parameters
SUBJECTS = [1, 2, 3, 4, 5]
CROP_SIZE_PIX = 100
LOG_DURATION = False
Z_SCORE_DURATION = False
Z_SORE_MEMSCORE = True  # z-score memorability scores


# Plot control flags
PLOT_RELATIONSHIPS = True
PLOT_MEMSCORE_STUFF = True

# Image parameters for scatter plot with crops
NUM_IMAGES = 50
SEEDS = [42, 43, 44, 45, 46]

# Quartile parameters
NUM_QUARTILES = 6
Q_LABELS = np.arange(NUM_QUARTILES)

# Percentile cutoffs for duration and memorability score filtering
DURATION_PERCENTILE = 98
MEMSCORE_PERCENTILE = 2