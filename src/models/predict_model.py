import os
import logging
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from src.constants import LIST_OF_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.path.join(os.path.dirname(__file__), '../../models'))
TEST_FILE = os.path.join(os.path.dirname(__file__), '../../data/processed/test.parquet')
SUBMISSION_FILE = os.path.join(os.path.dirname(__file__), '../../data/predict/submission.csv')
PREDICT_DIR = Path(os.path.dirname(SUBMISSION_FILE))


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_parquet(TEST_FILE)
        if df.empty:
            raise ValueError("Dataset is empty")

        missing_cols = set(LIST_OF_FEATURES) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
    except Exception as e:
        logger.exception("Failed to load test data")
        raise e


def get_best_model_path() -> Path:
    try:
        model_files = list(MODEL_DIR.glob('*.cbm'))
        if not model_files:
            raise FileNotFoundError("No model files found in models directory")

        return max(
            model_files,
            key=lambda x: float(x.stem.split('_')[2])
        )
    except Exception as e:
        logger.exception("Failed to get best model path")
        raise e


def load_model() -> CatBoostClassifier:
    try:
        model_path = get_best_model_path()
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logger.exception("Failed to load model")
        raise e


def make_predictions(df: pd.DataFrame, model: CatBoostClassifier) -> pd.DataFrame:
    try:
        predictions = model.predict_proba(df[LIST_OF_FEATURES])[:, 1]
        return pd.DataFrame({
            'id': df['id'],
            'score': predictions
        })
    except Exception as e:
        logger.exception("Failed to generate predictions")
        raise e


def save_predictions(predictions: pd.DataFrame) -> None:
    try:
        PREDICT_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(
            SUBMISSION_FILE,
            index=False,
            float_format='%.6f'
        )
        logger.info("Predictions saved to %s", SUBMISSION_FILE)
    except Exception as e:
        logger.exception("Failed to save predictions")
        raise e


def run_inference_pipeline() -> None:
    try:
        logger.info("Starting prediction pipeline")

        df = load_data()
        logger.info("Test data loaded successfully. Shape: %s", df.shape)

        model = load_model()
        predictions_df = make_predictions(df, model)
        save_predictions(predictions_df)

        logger.info("Prediction pipeline completed successfully")
    except Exception as e:
        logger.error("Prediction pipeline failed")
        raise e


if __name__ == "__main__":
    run_inference_pipeline()
