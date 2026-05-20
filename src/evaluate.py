#  Sample-weighted zero-mean R²

 
import logging
import numpy as np
import polars as pl
import pandas as pd
from typing import Union
 
from config import CFG
 

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("evaluate")


# Arrays can be numpy arrays, polars Series, or pandas Series
ArrayLike = Union[np.ndarray, pl.Series, pd.Series]
 
 

def weighted_r2(
    y_true      : ArrayLike,
    y_pred      : ArrayLike,
    weights     : ArrayLike,
) -> float:
    """
    Compute the sample-weighted zero-mean R² score.
 
    This is the official Jane Street competition metric.
 
    Formula:
        R² = 1 - ( Σ wᵢ(yᵢ - ŷᵢ)² ) / ( Σ wᵢ · yᵢ² )
 
    Key differences from standard R²:
        1. Weighted   — each row is weighted by Jane Street's confidence
        2. Zero-mean  — denominator uses yᵢ² not (yᵢ - ȳ)²
                        This means predicting all zeros gives R²=0,
                        not a negative score.
 
"""
    y_true  = _to_numpy(y_true)
    y_pred  = _to_numpy(y_pred)
    weights = _to_numpy(weights)
 
    # ── Input validation ─────────────────────────────────────────
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if len(y_true) != len(weights):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, weights={len(weights)}"
        )
    if weights.sum() == 0:
        raise ValueError("All weights are zero — cannot compute R²")
 
 
    valid_mask = (
        ~np.isnan(y_true) &
        ~np.isnan(y_pred) &
        ~np.isnan(weights) &
        (weights > 0)
    )
 
    if valid_mask.sum() == 0:
        raise ValueError("No valid rows after removing NaNs")
 
    y_true  = y_true[valid_mask]
    y_pred  = y_pred[valid_mask]
    weights = weights[valid_mask]
 
    # Numerator   : weighted sum of squared errors
    numerator   = np.sum(weights * (y_true - y_pred) ** 2)
 
    # Denominator : weighted sum of squared true values (zero-mean baseline)
    denominator = np.sum(weights * y_true ** 2)
 
    if denominator == 0:
        logger.warning("Denominator is zero (all y_true = 0). Returning 0.0")
        return 0.0
 
    r2 = 1.0 - (numerator / denominator)
 
    return float(r2)


def evaluate_predictions(
    df      : pl.DataFrame,
    y_pred  : ArrayLike,
    split   : str = "unknown",
) -> float:
    y_true=df[CFG.data.target]
    weights=df[CFG.data.weight_col]
    score=weighted_r2(y_true,y_pred,weights)
    logger.info(f"Weighted R² [{split}]: {score:.6f}")
 
    return score



def _to_numpy(arr: ArrayLike) -> np.ndarray:

    if isinstance(arr, pl.Series):
        return arr.to_numpy().astype(np.float64)
    elif isinstance(arr, pd.Series):
        return arr.values.astype(np.float64)
    elif isinstance(arr, np.ndarray):
        return arr.astype(np.float64)
    else:
        return np.array(arr, dtype=np.float64)
    


# 4.  BASELINE SCORE (PREDICT ZERO)
def baseline_score(df: pl.DataFrame, split: str = "unknown") -> float:
    n     = len(df)
    zeros = np.zeros(n)
    score = evaluate_predictions(df, zeros, split=f"{split}_baseline")

    logger.info(f"Baseline (predict zero) score [{split}]: {score:.6f}")

    return score


# 3.  EVALUATE ACROSS DATE RANGES
"""This is useful for spotting temporal degradation —
    if the score drops significantly in later date_ids,
    the model is not adapting well to market drift."""
def evaluate_by_date(    df      : pl.DataFrame,
    y_pred  : np.ndarray,
    n_bins  : int = 10,
) -> pd.DataFrame:
    

    df_pd=df.select(["date_id",CFG.data.target,CFG.data.weight_col]).to_pandas()
    df_pd["y_pred"]=_to_numpy(y_pred)


    date_min = df_pd["date_id"].min()
    date_max = df_pd["date_id"].max()
    bins=np.linspace(date_min,date_max+1,n_bins+1,dtype=int)#include last bin and last date #np.linspace(start, stop, number_of_points)
    results = []
    for i in range(len(bins) - 1):
        mask = (df_pd["date_id"] >= bins[i]) & (df_pd["date_id"] < bins[i + 1])
        chunk = df_pd[mask]

        if len(chunk) == 0:
            continue

        score = weighted_r2(
            chunk[CFG.data.target].values,
            chunk["y_pred"].values,
            chunk[CFG.data.weight_col].values,
        )

        results.append({
            "date_start"  : bins[i],
            "date_end"    : bins[i + 1] - 1,
            "n_rows"      : len(chunk),
            "weighted_r2" : round(score, 6),
        })
 
    results_df = pd.DataFrame(results)
    logger.info(f"Score by date range:\n{results_df.to_string(index=False)}")
 
    return results_df




 