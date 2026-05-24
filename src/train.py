"""Implements:
    - Day-based batching (one batch = one full trading day)
    - Multi-task loss (sum of weighted R² across 5 targets)
    - Early stopping (stop if val score doesn't improve for N epochs)
    - Model checkpointing (save best weights)
    - Training all 6 models (2 architectures × 3 seeds)
 """


import logging
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Dict
import polars as pl

from config import CFG
from model import build_model, get_device
from evaluate import weighted_r2


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)
logger=logging.getLogger("train")


# 1.  PREPARE DAY BATCH
def prepare_day_batch(
         df_day          : pl.DataFrame,
        feature_cols:List[str],
        target_cols: List[str],
        device: torch.device,
)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:


    #The GRU expects input shape: (seq_len, batch_size, input_size)
    #One day has 968 time_ids × n_symbols rows.We treat each symbol as a separate batch item.
    """
    trianing-date_id 700 → 1298 = 599 days = 599 batches per epoch

    validation-date_id 1299 → 1498 = 200 days = 200 batches

    testing-date_id 1499 → 1698 = 200 days = 200 batches



    each batch is

    968 time steps × ~39 symbols × 125 features

    = shape (968, 39, 125)

    = ~37,752 rows
    """

    # Sort by time_id then symbol_id — critical for correct sequence orde
    df_day=df_day.sort(["time_id","symbol_id"])#37,752 rows per day
    n_time_ids=df_day["time_id"].n_unique()
    n_symbols=df_day["symbol_id"].n_unique()

    X=torch.tensor(
       df_day.select(feature_cols).to_numpy(), dtype=torch.float32
    ).reshape(n_time_ids,n_symbols,len(feature_cols)).to(device)

    available_targets=[t for t in target_cols if t in df_day.columns]

    y=torch.tensor(df_day.select(available_targets).fill_null(0).fill_nan(0).to_numpy(),
                   dtype=torch.float32).reshape(n_time_ids,n_symbols,len(available_targets)).to(device)


    weights=torch.tensor(
       df_day[CFG.data.weight_col].fill_null(0).fill_nan(0).to_numpy(),
                dtype=torch.float32
    ).reshape(n_time_ids, n_symbols).to(device)

    return X, y, weights



def compute_multitask_loss(predictions, y, weights):#retuens as tensor to be use in next loop same calcualtion as weightedr2
    n_targets = y.shape[-1]
    total_loss = torch.tensor(0.0, device=predictions.device)
    w = weights.flatten()
    w_sum = w.sum()
    for t in range(n_targets):
        pred_t = predictions[:, :, t].flatten()
        true_t = y[:, :, t].flatten()
        mean_t = (w * true_t).sum() / w_sum
        ss_res = (w * (true_t - pred_t) ** 2).sum()
        ss_tot = (w * (true_t - mean_t) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        total_loss = total_loss - r2
    return total_loss / n_targets




def train_one_epoch(model, df_train, optimizer, feature_cols, target_cols, device):

    model.train()
    date_ids=df_train["date_id"].unique().sort().to_list()
    total_loss = 0.0
    n_days     = len(date_ids)
    for date_id in date_ids:
        df_day=df_train.filter(pl.col("date_id")==date_id)
        x, y, weights = prepare_day_batch(df_day, feature_cols, target_cols, device)
        predictions, _ = model(x)

        loss = compute_multitask_loss(predictions, y, weights)
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        total_loss+=loss.item()
    avg_loss = total_loss / n_days
    return avg_loss


# 4.  EVALUATE ONE EPOCH
def evaluate_one_epoch(
    model:nn.Module,
    df_val:pl.DataFrame,
    feature_cols:List[str],
    device:torch.device
)->float:#Only predicts responder_6

    model.eval()

    date_ids=sorted(df_val["date_id"].unique().to_list())
    all_preds=[]
    all_targets=[]
    all_weights=[]
    with torch.no_grad():
        for date_id in date_ids:
            df_day=df_val.filter(pl.col("date_id")==date_id)
            X,y,weights=prepare_day_batch(df_day,feature_cols,[CFG.data.target],device)
            prediction,_=model(X)

            pred_r6=prediction[:,:,0].flatten().cpu().numpy()
            true_r6=y[:,:,0].flatten().cpu().numpy()
            w= weights.flatten().cpu().numpy()
            all_preds.append(pred_r6)
            all_targets.append(true_r6)
            all_weights.append(w)

        all_preds=np.concatenate(all_preds)
        all_targets=np.concatenate(all_targets)
        all_weights=np.concatenate(all_weights)

        score=weighted_r2(all_targets,all_preds,all_weights)
    return score




def train_model(
        arcitecture:str,
        seed:int,
        df_train:pl.DataFrame,
        df_val:pl.DataFrame,
        feature_cols    : List[str],
        all_feature_cols: List[str],
)-> Tuple[nn.Module, Dict]:

    logger.info(f"Training Model {arcitecture} — seed {seed}")

    device=get_device()
# ── Build model
    model=build_model(
        arcitecture, len(feature_cols), seed=seed
    )
    optimizer=torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate)

    target_cols = [CFG.data.target] + CFG.data.aux_targets

    best_val_score=-np.inf
    best_weights=None
    epoch_no_improve=0


    history={
        "train_loss":[],
        "val_score":[]
    }
# ── Training loop
    for epoch in range(CFG.train.max_epochs):
        start_time=time.time()
        train_loss=train_one_epoch(model, df_train, optimizer, feature_cols, target_cols, device)
        val_score = evaluate_one_epoch(model, df_val, feature_cols, device)
        elapsed=time.time()-start_time

        logger.info(
            f"Epoch {epoch+1:3d}/{CFG.train.max_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val R²: {val_score:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_score"].append(val_score)
# ── Save best model
        if val_score>best_val_score:
            best_val_score=val_score
            best_weights={k:v.clone() for k,v in model.state_dict().items()}
            epoch_no_improve=0
            logger.info(f"New best val R²: {best_val_score:.6f} ✓")
        else:
            epoch_no_improve+=1
            logger.info(
                f"No improvement for {epoch_no_improve}/{CFG.train.early_stop} epochs"
            )

# ── Early stopping

        if epoch_no_improve>=CFG.train.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break


    model.load_state_dict(best_weights)
    logger.info(f"Loaded best weights — Val R²: {best_val_score:.6f}")
    model_path = CFG.paths.models_dir / f"model_{arcitecture}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")

    return model, history




def train_all_model(
  df_train:pl.DataFrame,
  df_val:    pl.DataFrame,
  feat_cols: List[str],
)->dict:


    all_models={}
    all_histories={}

    for arcitecture in["A","B"]:
        for seed in CFG.train.seed:
            key=f"{arcitecture}_{seed}"
            model,history = train_model(
                                arcitecture      = arcitecture,
                    seed             = seed,
                    df_train         = df_train,
                    df_val           = df_val,
                    feature_cols     = feat_cols,
                    all_feature_cols = feat_cols,
            )
            all_models[key]    = model
            all_histories[key] = history

            logger.info(f"Model {key} complete. Best val R²: {max(history['val_score']):.6f}")

    logger.info("=" * 50)
    logger.info("All 6 models trained.")
    logger.info("=" * 50)

    return all_models, all_histories



