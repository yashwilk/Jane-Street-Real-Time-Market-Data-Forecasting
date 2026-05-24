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



def compute_multitask_loss():
    pass

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
