import datetime
import numpy as np

# Epoch date
EPOCH_DATE = datetime.date(1970, 1, 1)

# Benford distribution
BENFORD_DIST = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])

# Quantile constants
QUANTILE_25_VALUE = 0.25
QUANTILE_75_VALUE = 0.75

# Autocorrelation constants
AUTOCORRELATION_LAG = 1

# FFT coefficient constants
FFT_COEFFICIENT_0 = 0
FFT_COEFFICIENT_1 = 1

# Number crossing constants
NUMBER_CROSSING_VALUE = 0

# Permutation entropy constants
PERMUTATION_ENTROPY_DEFAULT_ORDER = 3
PERMUTATION_ENTROPY_DEFAULT_DELAY = 1

# Energy ratio by chunks constants
ENERGY_RATIO_DEFAULT_NUM_SEGMENTS = 10
ENERGY_RATIO_DEFAULT_SEGMENT_FOCUS = 0

# CatBoost best hyperparameters
BEST_PARAMS = {
    'depth': None,
    'learning_rate': None,
    'l2_leaf_reg': None,
    'random_strength': None,
    'bagging_temperature': None,
    'od_type': None,
    'od_wait': None
}

MODEL_PATH = 'models/'
