import os
import logging
import pandas as pd
from typing import Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from src.constants import BEST_PARAMS, LIST_OF_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_FILE = os.path.join(os.path.dirname(__file__), '../../data/processed/train.parquet')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')

RANDOM_STATE = 42
TEST_SIZE = 0.1


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_parquet(TRAIN_FILE, engine='auto')
        if df.empty:
            raise ValueError("Dataset is empty.")

        required_columns = set(LIST_OF_FEATURES + ['target'])
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise e


def split_data(df: pd.DataFrame) -> Tuple[Pool, Pool]:
    try:
        df = df[LIST_OF_FEATURES + ['target']]

        X_train, X_val, y_train, y_val = train_test_split(
            df.drop(['target'], axis=1),
            df['target'],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        train_pool = Pool(data=X_train, label=y_train)
        val_pool = Pool(data=X_val, label=y_val)

        return train_pool, val_pool
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise e


def train_model(train_pool: Pool, val_pool: Pool) -> CatBoostClassifier:
    try:
        model = CatBoostClassifier(
            **BEST_PARAMS,
            eval_metric='AUC',
            verbose=100,
            early_stopping_rounds=100
        )
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise e


def evaluate_model(model: CatBoostClassifier, val_pool: Pool) -> float:
    try:
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        roc_auc = roc_auc_score(val_pool.get_label(), y_pred_proba)
        return roc_auc
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise e


def save_model(model: CatBoostClassifier, roc_auc: float) -> None:
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cbm_auc_{roc_auc:.4f}_{timestamp}.cbm"
        model_path = os.path.join(MODEL_DIR, model_name)

        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise e


def main() -> None:
    try:
        logger.info("Training pipeline:")

        df = load_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        train_pool, val_pool = split_data(df)
        logger.info("Data split completed.")

        model = train_model(train_pool, val_pool)
        logger.info("Model training completed.")

        roc_auc = evaluate_model(model, val_pool)
        logger.info(f"Model ROC-AUC score: {roc_auc:.4f}")

        save_model(model, roc_auc)
        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
