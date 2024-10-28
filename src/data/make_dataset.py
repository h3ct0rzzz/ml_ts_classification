import pandas as pd
import numpy as np
import logging
from src.features.build_features import *
from src.constants import (
    QUANTILE_25_VALUE,
    QUANTILE_75_VALUE,
    AUTOCORRELATION_LAG,
    FFT_COEFFICIENT_0,
    FFT_COEFFICIENT_1,
    NUMBER_CROSSING_VALUE
)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns:
        nan_mask = df[col_name].isna()
        if nan_mask.any():
            valid_values = df[~nan_mask][col_name].values
            for idx in df[nan_mask].index:
                df.at[idx, col_name] = np.random.choice(valid_values)
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df['values'].isna().any():
        raise ValueError("NaN values detected in 'values' column.")
    feature_dict = {
        'id': df['id'],
        'start_date': df['dates'].apply(get_start_date),
        'end_date': df['dates'].apply(get_end_date),
        'duration': df['dates'].apply(calculate_duration)
    }

    for name, func in globals().items():
        if hasattr(func, 'is_feature'):
            try:
                if name == 'quantile':
                    feature_dict['quantile_25'] = df['values'].apply(
                        lambda values: func(values, QUANTILE_25_VALUE)
                    )
                    feature_dict['quantile_75'] = df['values'].apply(
                        lambda values: func(values, QUANTILE_75_VALUE)
                    )
                elif name == 'autocorrelation':
                    feature_dict['autocorrelation_lag_1'] = df['values'].apply(
                        lambda values: func(values, AUTOCORRELATION_LAG)
                    )
                elif name == 'fft_coefficient':
                    feature_dict['fft_coefficient_0'] = df['values'].apply(
                        lambda values: np.abs(func(values, FFT_COEFFICIENT_0))
                    )
                    feature_dict['fft_coefficient_1'] = df['values'].apply(
                        lambda values: np.abs(func(values, FFT_COEFFICIENT_1))
                    )
                elif name == 'number_crossing_m':
                    feature_dict['number_crossing_0'] = df['values'].apply(
                        lambda values: func(values, NUMBER_CROSSING_VALUE)
                    )
                else:
                    feature_dict[name] = df['values'].apply(
                        lambda values: func(values) if not np.isnan(values).any() else np.nan
                    )
            except Exception as e:
                logging.warning(f"Error calculating feature {name}: {str(e)}")
                continue

    processed_df = pd.DataFrame(feature_dict)

    if 'label' in df.columns:
        processed_df['target'] = df['label']

    return processed_df


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ["object", "category", "datetime64[ns, UTC]"]:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing_values(df)
    df = generate_features(df)
    df = reduce_mem_usage(df)
    df = df.fillna(df.mean())
    return df
