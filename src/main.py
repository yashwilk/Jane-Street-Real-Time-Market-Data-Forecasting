"""
    1. Load and split data
    2. Build features
    3. Train all 6 models
    4. Run ensemble inference on test set
    5. Print final score   
"""

import logging
import argparse
import time
import numpy

from config import CFG
from data_loader import load_all
from feature import build_features
from train import train_all_model
from ensemble import run_ensemble
from inference import load_trained_models

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("main")


def main (skip_training: bool = False) -> None:
    start_time=time.time()
    logger.info("=" * 60)
    logger.info("Jane Street — Full Pipeline")
    logger.info("=" * 60)

    logger.info("\nStep 1: Loading data...")

    df_train,df_val,df_test,feature_cols=load_all()
     
    logger.info(f"Train : {len(df_train):,} rows")
    logger.info(f"Val   : {len(df_val):,} rows")
    logger.info(f"Test  : {len(df_test):,} rows")


    df_train, df_val, df_test, all_feature_cols = build_features(df_train, df_val, df_test, feature_cols)
    logger.info(f"Total features: {len(all_feature_cols)}")
    #now test and val have # 76 original features +16 market averages+ 16 rolling means+16 rolling stds+1 time_id_norm
    #train has all of the above + 2 auxiliary targets

    if skip_training:
        logger.info("\nStep 3: Loading saved models from disk...")
        all_models = load_trained_models(all_feature_cols)
#load model dictionary with 6 models and its weights
    else:
        logger.info("\nStep 3: Training all 6 models...")
        all_models, all_histories = train_all_model(
            df_train     = df_train,
            df_val       = df_val,
            feature_cols = all_feature_cols,
        )
        # Log best val scores for each model
        logger.info("\nTraining complete. Best validation scores:")
        for key, history in all_histories.items():
            best_score = max(history["val_score"])
            best_epoch = history["val_score"].index(best_score) + 1
            logger.info(f"  Model {key:8s} : R² = {best_score:.6f} (epoch {best_epoch})")

    logger.info("\nStep 4: Running ensemble inference on test set...")
    ensemble_predictions, ensemble_score = run_ensemble(
        df_test      = df_test,
        feature_cols = all_feature_cols,
        all_models   = all_models,
    )









    total_time = time.time() - start_time
 
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Ensemble Test R²  : {ensemble_score:.6f}")
    logger.info(f"Total time        : {total_time/60:.1f} minutes")
    logger.info(f"Predictions saved : {CFG.paths.predictions_dir / 'ensemble_predictions.npy'}")
    logger.info("=" * 60)
 
    # ── Context for the score ─────────────────────────────────────
    logger.info("\nScore context:")
    logger.info("  R² = 0.000  → baseline (predicting zero)")
    logger.info("  R² = 0.005  → decent model")
    logger.info("  R² = 0.010  → good model")
    logger.info("  R² = 0.0112 → 8th place solution target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jane Street Pipeline")
    parser.add_argument(
        "--skip-training",
        action  = "store_true",
        default = False,
        help    = "Skip training and load saved models from disk"
    )
    args = parser.parse_args()

    main(skip_training=args.skip_training)