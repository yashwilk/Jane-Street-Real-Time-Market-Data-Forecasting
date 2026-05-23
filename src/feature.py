# 1. Market averages  — mean of top-16 features per (date_id, time_id) 2. Rolling stats    — rolling mean + std per symbol over last 1000 time_ids
import logging
import numpy as np
import polars as pl
import pandas as pd
from typing import List, Tuple
 
from config import CFG


logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("features")


def impute_nulls(df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
    null_counts = df.select(feature_cols).null_count().to_pandas().sum(axis=1)[0]
    logger.info(f"Imputing {null_counts:,} null values with 0")

    df = df.with_columns([
        pl.col(c).fill_null(0.0).fill_nan(0.0)   #df[feature_cols] = (df[feature_cols].fillna(0.0))
        for c in feature_cols
    ])
 
    return df


def select_top_features(df:pl.DataFrame,feature_cols:List[str],n:int=CFG.features.n_top_features)->List[str]:
        logger.info(f"Selecting top {n} features by correlation with {CFG.data.target}")

        # Fill nulls before computing correlations
        df_filled=df.select(feature_cols+[CFG.data.target]).fill_null(0).fill_nan(0)

        target_vals=df_filled[CFG.data.target].to_numpy().astype(np.float64)

        correlation={}
        for col in feature_cols:
            feat_vals = df_filled[col].to_numpy().astype(np.float64)
            corr=np.corrcoef(feat_vals,target_vals)[0,1]
            correlation[col]=abs(corr) if not np.isnan(corr) else 0.0
        top_features=sorted(correlation,key=correlation.get,reverse=True)[:n]
        logger.info(f"Top {n} features selected: {top_features[:5]} ...")
        logger.info(
            f"Correlation range: "
            f"{correlation[top_features[0]]:.4f} → {correlation[top_features[-1]]:.4f}"
        )

        return top_features


# 3.  MARKET AVERAGES
def add_market_averages(
    df          : pl.DataFrame,
    top_features: List[str],
) -> pl.DataFrame:
     
    logger.info(f"Adding market averages for {len(top_features)} features")

    market_avg_cols=[pl.col(feat).mean().over(["date_id", "time_id"]).alias(f"{feat}_market_avg") for feat in top_features ]   
    """
    pandas equivalent 

    for feat in top_features:
        df[f"{feat}_market_avg"] = (
            df.groupby(["date_id", "time_id"])[feat]
              .transform("mean")
        )

    """
    df = df.with_columns(market_avg_cols)

    new_cols = [f"{feat}_market_avg" for feat in top_features]
    logger.info(f"Added {len(new_cols)} market average columns")
 
    return df
 

#   date_id=800, time_id=5, symbols=[0,1,2,3]
#   feature_00 values = [0.3, 0.2, 0.1, 0.4] → mean = 0.25
#   All 4 rows get feature_00_market_avg = 0.25


# 4.  ROLLING STATISTICS PER SYMBOL
def add_rolling_stats(
    df          : pl.DataFrame,
    top_features: List[str],
    window      : int = CFG.features.rolling_window,
) -> pl.DataFrame:
    df = df.sort(["symbol_id", "date_id", "time_id"])

    rolling_cols = []

    for feat in top_features:
        rolling_cols.append(pl.col(feat).shift(1).rolling_mean(window_size=window, min_periods=1).over("symbol_id").alias(f"{feat}_roll_mean_{window}"))
        rolling_cols.append(pl.col(feat).shift(1).rolling_std(window_size=window, min_periods=2).over("symbol_id").fill_null(1.0)   # fill early NaNs with 1 (neutral std)
            .alias(f"{feat}_roll_std_{window}")
        )
    df = df.with_columns(rolling_cols)
    new_cols = len(top_features) * 2
    logger.info(f"Added {new_cols} rolling stat columns")

    return df

"""  
pandas equivalent
  df[f"{feat}_roll_std_{window}"] = (
        df.groupby("symbol_id")[feat]
          .shift(1)
          .rolling(window, min_periods=2)
          .std()
          .fillna(1.0)
          .reset_index(level=0, drop=True)
    )
    
"""


def standardize(
    df_train    : pl.DataFrame,
    df_val      : pl.DataFrame,
    df_test     : pl.DataFrame,
    feature_cols: List[str],    
)->tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
     
# Standardize features to zero mean and unit variance.
#fit on train only, apply to all splits.

    logger.info(f"Standardizing {len(feature_cols)} feature columns")
    logger.info("Computing mean/std on train only — applying to all splits")

    #[compute mean ad std for ever column as store as array]1 single value for each column

    stats = df_train.select(feature_cols).select([
        pl.all().mean().name.prefix("mean_"),
        pl.all().std().name.prefix("std_"),
    ])
    

    # PANDAS
    """
    mean_stats = df_train[feature_cols].mean().add_prefix("mean_")
    std_stats  = df_train[feature_cols].std().add_prefix("std_")

    stats = pd.concat([mean_stats, std_stats])

    """

    means = {col: stats[f"mean_{col}"][0] for col in feature_cols}
    stds  = {col: stats[f"std_{col}"][0]  for col in feature_cols}

    def _apply_standardization(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ((pl.col(c) - means[c]) / (stds[c] if stds[c] > 1e-8 else 1.0))
            .alias(c)
            for c in feature_cols
        ])

    df_train = _apply_standardization(df_train)
    df_val   = _apply_standardization(df_val)
    df_test  = _apply_standardization(df_test)
 
    logger.info("Standardization complete")
 
    return df_train, df_val, df_test


#All 125 columns get standardized in one go.


def add_auxiliary_targets(df: pl.DataFrame) -> pl.DataFrame:
     
    """We engineer two more:
        responder_9  ≈ 8-day  rolling avg = responder_8 + responder_8 shifted -4
        responder_10 ≈ 60-day rolling avg = responder_6 + r6 shifted -20 + r6 shifted -40
"""
    logger.info("Adding engineered auxiliary targets: responder_9, responder_10")

    logger.info("Adding engineered auxiliary targets: responder_9, responder_10")
 
    df = df.with_columns([
        # responder_9 ≈ 8-day rolling avg
        (
            pl.col("responder_8")
            + pl.col("responder_8").shift(-4).over("symbol_id")
        ).fill_null(0.0).alias("responder_9"),
 
        # responder_10 ≈ 60-day rolling avg
        (
            pl.col("responder_6")
            + pl.col("responder_6").shift(-20).over("symbol_id")
            + pl.col("responder_6").shift(-40).over("symbol_id")
        ).fill_null(0.0).alias("responder_10"),
    ])
 
    logger.info("Auxiliary targets added: responder_9, responder_10")
 
 #responder_9   = 8-day   ← short term
 #responder_10  = 60-day  ← longer term

    return df


# 7.  ADD TIME_ID AS FEATURE
def add_time_id_feature(df: pl.DataFrame) -> pl.DataFrame:
    if not CFG.features.add_time_id_feature:
        return df
 
    max_time_id = CFG.data.n_time_ids - 1  # 967
 
    df = df.with_columns(
        (pl.col("time_id") / max_time_id).alias("time_id_norm")
    )
 
    logger.info("Added time_id_norm feature (normalized to [0, 1])")
 
    return df



def build_features(
    df_train    : pl.DataFrame,
    df_val      : pl.DataFrame,
    df_test     : pl.DataFrame,
    feature_cols: List[str]
)-> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:

    """Steps (in order):
        1. Impute NaNs with 0
        2. Select top-16 features by correlation (on train only)
        3. Add market averages
        4. Add rolling statistics
        5. Add time_id as normalized feature
        6. Add auxiliary targets (train only — not needed for val/test features)
        7. Standardize all features (fit on train, apply to all)
        8. Return updated dataframes and final feature column list"""

    logger.info("Starting feature engineering pipeline")

 # Step 1: Impute NaNs
    df_train = impute_nulls(df_train, feature_cols)
    df_val   = impute_nulls(df_val,   feature_cols)
    df_test  = impute_nulls(df_test,  feature_cols)
#Step 2: Select top features (train only)
    top_features = select_top_features(df_train, feature_cols)
    CFG.features.top_features = top_features

#Step 3: Market averages
    df_train = add_market_averages(df_train, top_features)
    df_val   = add_market_averages(df_val,   top_features)
    df_test  = add_market_averages(df_test,  top_features)
#Step 4: Rolling statistics
    df_train = add_rolling_stats(df_train, top_features)
    df_val   = add_rolling_stats(df_val,   top_features)
    df_test  = add_rolling_stats(df_test,  top_features)

#Step 5: Time ID feature
    df_train = add_time_id_feature(df_train)
    df_val   = add_time_id_feature(df_val)
    df_test  = add_time_id_feature(df_test)

# Step 6: Auxiliary targets (train only)
    df_train = add_auxiliary_targets(df_train)


    market_avg_cols  = [f"{f}_market_avg"                    for f in top_features]
    rolling_mean_cols= [f"{f}_roll_mean_{CFG.features.rolling_window}" for f in top_features]
    rolling_std_cols = [f"{f}_roll_std_{CFG.features.rolling_window}"  for f in top_features]
    time_id_cols     = ["time_id_norm"] if CFG.features.add_time_id_feature else []
 
    all_feature_cols = (
        feature_cols        +   # 76 original features
        market_avg_cols     +   # 16 market averages
        rolling_mean_cols   +   # 16 rolling means
        rolling_std_cols    +   # 16 rolling stds
        time_id_cols            # 1 time_id_norm
    )


#Step 8: Standardize
    df_train, df_val, df_test = standardize(
        df_train, df_val, df_test, all_feature_cols
    )

    logger.info("=" * 50)
    logger.info(f"Feature engineering complete")
    logger.info(f"Total features: {len(all_feature_cols)}")
    logger.info(f"  Original  : {len(feature_cols)}")
    logger.info(f"  Market avg: {len(market_avg_cols)}")
    logger.info(f"  Roll mean : {len(rolling_mean_cols)}")
    logger.info(f"  Roll std  : {len(rolling_std_cols)}")
    logger.info(f"  Time ID   : {len(time_id_cols)}")
    logger.info("=" * 50)
 
    return df_train, df_val, df_test, all_feature_cols
 



if __name__ == "__main__":
    print("=" * 55)
    print("features.py — Sanity Check")
    print("=" * 55)
 
    from data_loader import load_all
    from evaluate import baseline_score
 
    # Load data
    df_train, df_val, df_test, feature_cols = load_all()
 
    # Run feature pipeline
    df_train, df_val, df_test, all_feature_cols = build_features(
        df_train, df_val, df_test, feature_cols
    )
 
    print(f"\nTrain shape after features : {df_train.shape}")
    print(f"Val   shape after features : {df_val.shape}")
    print(f"Test  shape after features : {df_test.shape}")
    print(f"Total feature columns      : {len(all_feature_cols)}")
 
    # Check auxiliary targets were added to train
    assert "responder_9"  in df_train.columns, "responder_9 missing from train"
    assert "responder_10" in df_train.columns, "responder_10 missing from train"
    print(f"\nAuxiliary targets ✓ found in train")
 
    # Check auxiliary targets NOT in val/test (not needed there)
    assert "responder_9"  not in df_val.columns,  "responder_9 should not be in val"
    assert "responder_10" not in df_test.columns, "responder_10 should not be in test"
    print(f"Auxiliary targets ✓ correctly absent from val/test")
 
    # Check no NaNs in feature columns after imputation
    null_count = df_train.select(all_feature_cols).null_count().to_pandas().sum(axis=1)[0]
    assert null_count == 0, f"NaNs found after imputation: {null_count}"
    print(f"NaN check ✓ — zero nulls in feature columns after imputation")
 
    # Check top features saved to config
    assert len(CFG.features.top_features) == CFG.features.n_top_features
    print(f"Top features ✓ saved to CFG: {CFG.features.top_features[:3]} ...")
 
    # Check time_id_norm is between 0 and 1
    time_min = df_train["time_id_norm"].min()
    time_max = df_train["time_id_norm"].max()
    assert 0.0 <= time_min <= time_max <= 1.0, "time_id_norm out of range"
    print(f"time_id_norm ✓ range: [{time_min:.4f}, {time_max:.4f}]")
 
    print("\n" + "=" * 55)
    print("features.py — All checks passed.")
    print("=" * 55)