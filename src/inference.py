"""Day 1499 → model predicts → save hidden state h1
Day 1500 → model starts with h1 → predicts → save h2
During inference we only go forward once — day 1499 → 1500 → 1501 — strictly in order. That's when carrying hidden state makes sense.
"""

import logging
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from config import CFG
from model import get_device
from evaluate import weighted_r2, evaluate_by_date
from train import prepare_day_batch


logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("inference")

# 1.  ONLINE LEARNING STEP
def online_update(model: nn.Module,
                  x: torch.Tensor,
                  y_true: torch.Tensor,
                  weights: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
    """One forward pass + one backward pass + one weight update.
    return Online learning loss for this day."""

    model.train()#dropout ON
    predictions, _ = model(x)
    y_pred_r6 = predictions[:, :, 0].flatten()  # only use responder_6
    y_true_r6 = y_true[:, :, 0].flatten()
    w = weights.flatten()

    numerator = (w * (y_true_r6 - y_pred_r6) ** 2).sum()
    denominator = (w * y_true_r6 ** 2).sum()

    if denominator > 1e-8:
        r2 = 1 - numerator / denominator
        loss = -r2
    else:
        loss = torch.tensor(0.0, device=x.device)#compute loss

    optimizer.zero_grad()
    loss.backward()#gradients computed
    optimizer.step()

    return loss.item()



def predict_one_day(
    model: nn.Module,
    df_day: pl.DataFrame,
    features: List[str],
    hidden: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    model.eval() #dropout OFF
    x, y_true, weights = prepare_day_batch(df_day, features, [CFG.data.target], device)

    with torch.no_grad():#no gradients computed
        predictions, hidden = model(x, hidden)
        predictions_np = predictions[:, :, 0].flatten().cpu().numpy()
    return predictions_np, x, y_true, weights, hidden



def run_inference_single(
    model: nn.Module,
    df_test: pl.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, float]:

    """       For each day in test set:
            1. Feed features to model → get predictions
            2. Reveal true labels
            3. Perform one online learning weight update
            4. Move to next day

    Hidden state is carried forward across all 200 test days."""

    device = get_device()
    model = model.to(device)
    all_predictions = []
    all_true        = []
    all_weights     = []

    online_optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.online.learning_rate
    )
    date_ids = sorted(df_test["date_id"].unique().to_list())

    hidden = None

    for i, date_id in enumerate(date_ids):
        df_day = df_test.filter(pl.col("date_id") == date_id)
        predictions_np, x, y_true, weights, hidden = predict_one_day(model, df_day, feature_cols, hidden, device)

        true_np = y_true[:, :, 0].flatten().cpu().numpy()
        weights_np = weights.flatten().cpu().numpy()


        all_predictions.append(predictions_np)
        all_true.append(true_np)
        all_weights.append(weights_np)

        online_loss = online_update(
            model, x, y_true, weights, online_optimizer
        )

        if (i+1) % 50 == 0:
            running_pred    = np.concatenate(all_predictions)
            running_true    = np.concatenate(all_true)
            running_weights = np.concatenate(all_weights)




            running_r2      = weighted_r2(running_true, running_pred, running_weights)

            logger.info(
                f"Day {i+1}/{len(date_ids)} | "
                f"date_id={date_id} | "
                f"Running R²={running_r2:.6f} | "
                f"Online loss={online_loss:.6f}"
            )

    all_preds = np.concatenate(all_predictions)
    all_true_np = np.concatenate(all_true)
    all_weights_np = np.concatenate(all_weights)
    final_r2 = weighted_r2(all_true_np, all_preds, all_weights_np)
    return all_preds, final_r2



# 4.  RUN INFERENCE FOR ALL 6 MODELS
def run_inference(
    all_models   : Dict,
    df_test      : pl.DataFrame,
    feature_cols : List[str],
) -> Tuple[np.ndarray, float]:
    all_model_predictions = []
    all_r2 = []
    logger.info("=" * 50)
    logger.info(f"Running inference for {len(all_models)} models")
    logger.info("=" * 50)

    for key, model in all_models.items():
        predictions, r2 = run_inference_single(model, df_test, feature_cols)
        all_model_predictions.append(predictions)
        logger.info(f"Model {key} — Test R²: {r2:.6f}")
 
    # ── Ensemble: simple average ──────────────────────────────────
    ensemble_predictions = np.mean(all_model_predictions, axis=0)
 
    # ── Final ensemble score ──────────────────────────────────────
    df_test_pd  = df_test.select([CFG.data.target, CFG.data.weight_col]).to_pandas()
    ensemble_r2 = weighted_r2(
        df_test_pd[CFG.data.target].values,
        ensemble_predictions,
        df_test_pd[CFG.data.weight_col].values,
    )
 
    logger.info("=" * 50)
    logger.info(f"Ensemble Test R²: {ensemble_r2:.6f}")
    logger.info("=" * 50)
 
    return ensemble_predictions, ensemble_r2


def load_trained_models(
    feature_cols : List[str],
) -> Dict:
    from model import build_model

    device=get_device()

    all_models={}
    for arcitecture in["A","B"]:
        for seed in CFG.train.seed:
            key        = f"{arcitecture}_{seed}"
            model_path = CFG.paths.model_dir / f"model_{arcitecture}_seed{seed}.pt"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Run train.py first to train and save the models."
                )
            model = build_model(arcitecture, input_size=len(feature_cols), seed=seed)

            # Load saved weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()

            all_models[key] = model
            logger.info(f"Loaded model {key} from {model_path}")

    logger.info(f"Loaded {len(all_models)} models")
    return all_models


if __name__ == "__main__":
    print("=" * 55)
    print("inference.py — Sanity Check")
    print("=" * 55)
 
    from data_loader import load_all
    from features import build_features
    from train import train_model
 
    # Load data
    df_train, df_val, df_test, feature_cols = load_all()
 
    # Use small subset for sanity check
    df_train_small = df_train.filter(pl.col("date_id") <= 704)
    df_val_small   = df_val.filter(pl.col("date_id") <= 1303)
    df_test_small  = df_test.filter(pl.col("date_id") <= 1503)
 
    # Build features
    df_train_small, df_val_small, df_test_small, all_feature_cols = build_features(
        df_train_small, df_val_small, df_test_small, feature_cols
    )
 
    # Train one small model
    CFG.train.max_epochs = 2
    model, _ = train_model(
        architecture     = "B",
        seed             = 42,
        df_train         = df_train_small,
        df_val           = df_val_small,
        feature_cols     = all_feature_cols,
        all_feature_cols = all_feature_cols,
    )
 
    # Run inference on small test set
    predictions, r2 = run_inference_single(
        model        = model,
        df_test      = df_test_small,
        feature_cols = all_feature_cols,
    )
 
    print(f"\nPredictions shape : {predictions.shape}")
    print(f"Test R²           : {r2:.6f}")
    print(f"Expected           : small positive or negative number close to 0")
 
    assert predictions.shape[0] > 0, "No predictions generated"
    print("\nAll assertions passed ✓")
 
    print("\n" + "=" * 55)
    print("inference.py — Sanity Check passed.")
    print("=" * 55)