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


"""weighted_r2-takes in y true,ypred,weights 
|
call_numpy_ function to convert all 3 into np.array
|
checks len ytrue and  y_pred
|
checks length of y true and weights
|
check if sum of weight is not 0
|
valid mask check all non empty rows and weight>0
|
calc weighted sum of squared errors -N
|
weighted sum of squared true
|
return r2 as float
"""


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

"""
evaluate_predictions
|
takes in df to extract ytrue,takes in y_pred,split
|
computes weighted_r2
"""

def _to_numpy(arr: ArrayLike) -> np.ndarray:

    if isinstance(arr, pl.Series):
        return arr.to_numpy().astype(np.float64)
    elif isinstance(arr, pd.Series):
        return arr.values.astype(np.float64)
    elif isinstance(arr, np.ndarray):
        return arr.astype(np.float64)
    else:
        return np.array(arr, dtype=np.float64)

"""_to_numpy-takes pl.series,pd,series,np,Array
|
converts all input to np.array"""


# 4.  BASELINE SCORE (PREDICT ZERO)
def baseline_score(df: pl.DataFrame, split: str = "unknown") -> float:
    n     = len(df)
    zeros = np.zeros(n)
    score = evaluate_predictions(df, zeros, split=f"{split}_baseline")

    logger.info(f"Baseline (predict zero) score [{split}]: {score:.6f}")

    return score

"""
baseline_score
|
takes df_val
|
creates zeros same length as df
|
computes baseling score
"""


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


"""
evaluate_by_date-taked df,ypred and bin count
|
selects dateid,target cols and weightcals form polars df can convert to pandas df
|
to dhis df add ypred 
|
find the min date and max date
|
create an array of date base
|
loop through all the index of the bin Array
|
split the df based on the bins
|
check if the length of df is ==0
|
calculate weighted r2 for this chuck of df
|
collect all the chunks
"""





if __name__ == "__main__":
    print("=" * 55)
    print("evaluate.py — Sanity Check")
    print("=" * 55)
 
    y_true  = np.array([1.0,  2.0, -1.0,  0.5])
    weights = np.array([1.0,  1.0,  1.0,  1.0])
    score   = weighted_r2(y_true, y_true, weights)
    assert abs(score - 1.0) < 1e-9, f"Perfect prediction should be 1.0, got {score}"
    print(f"Test 1 — Perfect predictions     : R² = {score:.6f} ✓")
 
    y_pred_zero = np.zeros(4)
    score       = weighted_r2(y_true, y_pred_zero, weights)
    assert abs(score - 0.0) < 1e-9, f"Zero prediction should be 0.0, got {score}"
    print(f"Test 2 — Predict zero (baseline) : R² = {score:.6f} ✓")
 
    y_pred_bad = -y_true  
    score      = weighted_r2(y_true, y_pred_bad, weights)
    assert score < 0.0, f"Bad predictions should give R² < 0, got {score}"
    print(f"Test 3 — Predict wrong direction : R² = {score:.6f} ✓ (negative)")

    y_true2   = np.array([1.0, 1.0])
    y_pred2   = np.array([1.0, 0.0])   # first row perfect, second row wrong
    w_high    = np.array([10.0, 1.0])  # first row weighted heavily
    w_low     = np.array([1.0, 10.0])  # second row weighted heavily
    score_high = weighted_r2(y_true2, y_pred2, w_high)
    score_low  = weighted_r2(y_true2, y_pred2, w_low)
    assert score_high > score_low, "Higher weight on good prediction should give higher R²"
    print(f"Test 4 — Weighted scoring        : high_w={score_high:.4f} > low_w={score_low:.4f} ✓")
 
    y_true3   = np.array([1.0, np.nan, 2.0])
    y_pred3   = np.array([1.0, 0.5,   2.0])
    weights3  = np.array([1.0, 1.0,   1.0])
    score     = weighted_r2(y_true3, y_pred3, weights3)
    print(f"Test 5 — NaN handling            : R² = {score:.6f} ✓ (NaN row excluded)")
 
    print("\nLoading real data for baseline check...")
    from data_loader import load_all
    _, df_val, _, _ = load_all()
 
    baseline = baseline_score(df_val, split="val")
    assert abs(baseline) < 1e-6, f"Baseline should be ~0.0, got {baseline}"
    print(f"Test 6 — Real data baseline      : R² = {baseline:.6f} ✓")
 
    print("\n" + "=" * 55)
    print("evaluate.py — All checks passed.")
    print("=" * 55)
 