"""Load all 6 saved models from disk
Run inference for each model → get 6 sets of predictions
Average them equally → one final prediction"""

import logging
import numpy as np
import polars as pl
from typing import Dict, List, Tuple

from config import CFG
from evaluate import weighted_r2, evaluate_by_date
from inference import run_inference_single, load_trained_models

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("ensemble")


def collect_predictions(
    all_models: Dict,
    df_test: pl.DataFrame,
    feature_cols: List[str],
) -> Dict[str, np.ndarray]:

    logger.info("=" * 50)
    logger.info(f"Collecting predictions from {len(all_models)} models")
    logger.info("=" * 50)

    all_predictions = {}

    for key, model in all_models.items():
        logger.info(f"Running inference — Model {key}")
        prediction, r2 = run_inference_single(
            model=model,
            df_test=df_test,
            feature_cols=feature_cols
        )

        all_predictions[key] = prediction
        logger.info(f"Model {key} — Individual R²: {r2:.6f}")

    return all_predictions


def average_predictions(
    all_predictions: Dict[str, np.ndarray]
) -> np.ndarray:
    prediction_matrix = np.stack(list(all_predictions.values()), axis=0)
    logger.info(f"Predictions matrix shape: {prediction_matrix.shape}")
    logger.info(f"  {prediction_matrix.shape[0]} models × {prediction_matrix.shape[1]:,} rows")
    ensemble_predictions = prediction_matrix.mean(axis=0)
    logger.info(f"Ensemble predictions shape: {ensemble_predictions.shape}")

    return ensemble_predictions


def score_ensemble(
    ensemble_predictions: np.ndarray,
    df_test: pl.DataFrame,
) -> float:

    y_true  = df_test[CFG.data.target]
    weights = df_test[CFG.data.weight_col]
    score   = weighted_r2(y_true, ensemble_predictions, weights)
    logger.info(f"Ensemble final R²: {score:.6f}")

    return score


def compare_models(
    all_predictions      : Dict[str, np.ndarray],
    ensemble_predictions : np.ndarray,
    df_test              : pl.DataFrame,
) -> None:

    y_true  = df_test[CFG.data.target]
    weights = df_test[CFG.data.weight_col]

    logger.info("=" * 50)
    logger.info("Model comparison:")
    logger.info("=" * 50)

    scores = {}

    for key, predictions in all_predictions.items():
        score = weighted_r2(y_true, predictions, weights)
        scores[key] = score

    ensemble_score = weighted_r2(y_true, ensemble_predictions, weights)
    scores["ensemble"] = ensemble_score

    best_key   = max(scores, key=scores.get)
    best_score = scores[best_key]
    if best_key == "ensemble":
        logger.info("Ensemble beats all individual models ✓")
    else:
        logger.info(
            f"Best single model: {best_key} ({best_score:.6f}) "
            f"vs ensemble ({ensemble_score:.6f})"
        )



def run_ensemble(
    df_test: pl.DataFrame,
    feature_cols: List[str],
    all_models: Dict = None
) -> Tuple[np.ndarray, float]:

    if all_models is None:
        logger.info("Loading trained models from disk...")
        all_models = load_trained_models(feature_cols)

    all_predictions      = collect_predictions(all_models, df_test, feature_cols)
    ensemble_predictions = average_predictions(all_predictions)
    ensemble_score       = score_ensemble(ensemble_predictions, df_test)
    compare_models(all_predictions, ensemble_predictions, df_test)
    pred_path = CFG.paths.predictions_dir / "ensemble_predictions.npy"
    np.save(pred_path, ensemble_predictions)
    logger.info(f"Predictions saved to: {pred_path}")

    return ensemble_predictions, ensemble_score

if __name__ == "__main__":
    print("=" * 55)
    print("ensemble.py — Sanity Check")
    print("=" * 55)
 
    # Test average_predictions with dummy data
    dummy_predictions = {
        "A_42"  : np.array([0.1, 0.2, 0.3]),
        "A_123" : np.array([0.2, 0.3, 0.4]),
        "A_2024": np.array([0.3, 0.4, 0.5]),
        "B_42"  : np.array([0.4, 0.5, 0.6]),
        "B_123" : np.array([0.5, 0.6, 0.7]),
        "B_2024": np.array([0.6, 0.7, 0.8]),
    }
 
    ensemble = average_predictions(dummy_predictions)
 
    # Expected average: (0.1+0.2+0.3+0.4+0.5+0.6)/6 = 0.35 for row 0
    expected_row0 = np.mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert abs(ensemble[0] - expected_row0) < 1e-6, \
        f"Wrong average: {ensemble[0]} vs {expected_row0}"
    print(f"Average predictions ✓ — row 0: {ensemble[0]:.4f} (expected {expected_row0:.4f})")
 
    # Test shape
    assert ensemble.shape == (3,), f"Wrong shape: {ensemble.shape}"
    print(f"Shape ✓ — {ensemble.shape}")
 
    # Test that ensemble is between min and max of individual predictions
    all_preds_matrix = np.stack(list(dummy_predictions.values()))
    assert np.all(ensemble >= all_preds_matrix.min(axis=0)), \
        "Ensemble below minimum"
    assert np.all(ensemble <= all_preds_matrix.max(axis=0)), \
        "Ensemble above maximum"
    print("Ensemble within bounds ✓")
 
    print("\n" + "=" * 55)
    print("ensemble.py — All checks passed.")
    print("=" * 55)