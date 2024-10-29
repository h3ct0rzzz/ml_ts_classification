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
    'learning_rate': 0.05549981700358849,
    'l2_leaf_reg': 17,
    'colsample_bylevel': 0.7096296254119971,
    'auto_class_weights': 'Balanced',
    'depth': 7,
    'bootstrap_type': 'MVS',
    'boosting_type': 'Plain',
    'random_strength': 2.7716379366222275,
    'iterations': 1000,
    'loss_function': 'Logloss'
}

# List of features
LIST_OF_FEATURES = [
    'sum_values', 'linear_trend', 'mean', 'last_location_of_minimum',
    'autocorrelation_lag_1', 'mean_abs_change',
    'percentage_of_reoccurring_datapoints',
    'percentage_of_reoccurring_values', 'fft_coefficient_1',
    'last_location_of_maximum', 'ratio_unique_values', 'abs_energy',
    'sum_of_reoccurring_values', 'quantile_75', 'mean_change',
    'phase_duration', 'variation_coefficient', 'time_series_entropy',
    'root_mean_square', 'duration', 'first_location_of_maximum',
    'first_location_of_minimum', 'absolute_sum_of_changes',
    'fft_coefficient_0', 'skewness', 'end_date', 'variance',
    'mean_second_derivative_central', 'standard_deviation',
    'start_date', 'maximum', 'median', 'number_crossing_0',
    'benford_correlation', 'permutation_entropy',
    'energy_ratio_by_chunks', 'kurtosis', 'quantile_25',
    'count_above_mean'
]
