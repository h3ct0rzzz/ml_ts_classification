import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold
from src.constants import BEST_PARAMS, MODEL_PATH


def train_model(df: pd.DataFrame) -> None:
    pass


def evaluate_model(model: CatBoostRegressor, df: pd.DataFrame) -> None:
    pass


def save_model(model: CatBoostRegressor) -> None:
    pass
