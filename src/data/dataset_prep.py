import os
import pandas as pd
from typing import Tuple
import warnings
from src.data.make_dataset import make_dataset


warnings.filterwarnings("ignore")

TRAIN_FILE = os.path.join(os.path.dirname(__file__), '../../data/raw/train.parquet')
TEST_FILE = os.path.join(os.path.dirname(__file__), '../../data/raw/test.parquet')

PROCESSED_TRAIN_FILE = os.path.join(os.path.dirname(__file__), '../../data/processed/train.parquet')
PROCESSED_TEST_FILE = os.path.join(os.path.dirname(__file__), '../../data/processed/test.parquet')


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df = pd.read_parquet(TRAIN_FILE, engine='auto')
    except FileNotFoundError:
        print(f"Error: {TRAIN_FILE} not found.")
        return None, None

    try:
        test_df = pd.read_parquet(TEST_FILE, engine='auto')
    except FileNotFoundError:
        print(f"Error: {TEST_FILE} not found.")
        return None, None

    return train_df, test_df


def process_datasets(train_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Processing train dataset...")
    train_df = make_dataset(train_df)
    print(f"Train dataset processed. Shape: {train_df.shape}")

    print("Processing test dataset...")
    test_df = make_dataset(test_df)
    print(f"Test dataset processed. Shape: {test_df.shape}")

    print("Done!")
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> None:
    train_df = train_df.dropna()
    train_df.to_parquet(PROCESSED_TRAIN_FILE, engine='auto')
    test_df.to_parquet(PROCESSED_TEST_FILE, engine='auto')


def main() -> None:
    train_df, test_df = load_data()
    if train_df is not None and test_df is not None:
        train_df, test_df = process_datasets(train_df, test_df)
        save_processed_data(train_df, test_df)


if __name__ == "__main__":
    main()
