#Responsible for one thing only: loading and splitting the raw data.

import logging  #same as print but can be recorded
import pandas as pd
import polars as pl #same as pandas but used for bigger file
from pathlib import Path
from typing import Tuple
from config import CFG


logging.basicConfig(level=logging.INFO,
                    format="[%(name)s] %(message)s")

logger=logging.getLogger("data_loader")

# Format: [data_loader] Loaded 5.7M rows  <- you always know the source

# 1.  LOAD RAW TRAINING DATA

def load_train_data(path:Path=CFG.paths.train_path)->pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"training data not found at {path}")

    logger.info(f"loading training data from {path}")

    df=pl.read_parquet(path)

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"date_id range: {df['date_id'].min()} - {df['date_id'].max()}")

    return df

"""
load_train_data
|
if path doesnt exist KeyError
|
read file using ploas(conver to pl df)
"""



# 2.  FILTER TO STABLE PERIOD
def filter_stable_period(df :pl.DataFrame)->pl.DataFrame:
    # Remove data before date_id 700.
    rows_before=len(df)
    df=df.filter(pl.col("date_id")>=CFG.data.date_start)
    rows_after=len(df)

    rows_dropped=rows_before-rows_after
    logger.info(
        f"Filtered to date_id >= {CFG.data.date_start}: "
        f"kept {rows_after:,} rows, dropped {rows_dropped:,} rows"
    )

    return df


"""
filter_stable_period - take the pl dataframe
|
remove first 700 rows usning pl.filter
pandas version
df = df[df["date_id"] > CFG.data.date_start]
"""



def split_data(df: pl.DataFrame)->Tuple[pl.DataFrame,pl.DataFrame,pl.DataFrame]:
    #Split the stable-period data into train, validation, and test sets.
    df_train=df.filter(pl.col('date_id')<=CFG.data.train_end)
    df_val = df.filter(
        (pl.col("date_id") >= CFG.data.val_start) &
        (pl.col("date_id") <= CFG.data.val_end)
    )

    df_test = df.filter(
        pl.col("date_id") >= CFG.data.test_start
    )

    assert len(df_train)>0,"Training set is empty — check date boundaries"  #assert is a tru or fals check if its false the returns the error statement
    assert len(df_test)>0, "Test set is empty — check date boundaries"
    assert len(df_val)>0,"Validation set is empty — check date boundaries"

    assert df_train['date_id'].max()<df_val['date_id'].min(),"DATA LEAKAGE: train and val date_ids overlap"
    assert df_val["date_id"].max() < df_test["date_id"].min(), \
        "DATA LEAKAGE: val and test date_ids overlap"

    logger.info("Data split complete:")
    logger.info(
        f"  Train : date_id {df_train['date_id'].min()} → "
        f"{df_train['date_id'].max()} | {len(df_train):,} rows"
    )
    logger.info(
        f"  Val   : date_id {df_val['date_id'].min()} → "
        f"{df_val['date_id'].max()} | {len(df_val):,} rows"
    )
    logger.info(
        f"  Test  : date_id {df_test['date_id'].min()} → "
        f"{df_test['date_id'].max()} | {len(df_test):,} rows"
    )

    return df_train, df_val, df_test


"""
split_data -takes the filtered df- return 3 dfs 
|
filter the main dfs by date ids into 3 df
|
retun 3 df
"""






# 4.  GET FEATURE AND RESPONDER COLUMN NAMES
def get_feature_cols(df:pl.DataFrame)-> list:
    #   Return all feature column names, excluding the 3 categorical ones.
    all_features=[c for c in df.columns if c.startswith("feature_")]
    feature_cols=[f for f in all_features if f not in CFG.data.drop_features]

    logger.info(
        f"Feature columns: {len(feature_cols)} used, "
        f"{len(CFG.data.drop_features)} dropped {CFG.data.drop_features}"
    )
 
    return feature_cols


"""
get_feature_cols- take the main df
|
return all get_feature_cols
|
drop feature  which are not needed
|
return the final feature columns as list
"""



def get_responder_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("responder_")]

"""get_responder_cols takes df and returns list of responders"""



def load_feature_meta()->pd.DataFrame:
    path = CFG.paths.feature_csv
 
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found at: {path}")
 
    df_meta = pd.read_csv(path)
    logger.info(f"Loaded features metadata: {len(df_meta)} rows")
 
    return df_meta



def load_responders_meta() -> pd.DataFrame:

    path = CFG.paths.responder_csv
 
    if not path.exists():
        raise FileNotFoundError(f"responders.csv not found at: {path}")
 
    df_meta = pd.read_csv(path)
    logger.info(f"Loaded responders metadata: {len(df_meta)} rows")
 
    return df_meta




def load_lags(path: Path = CFG.paths.lags_path) -> pl.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"lags.parquet not found at: {path}")
 
    logger.info(f"Loading lags data from: {path}")
    df_lags = pl.read_parquet(path)
    logger.info(f"Lags shape: {df_lags.shape}")
 
    return df_lags



def load_all() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list]:
    logger.info("Starting full data load pipeline...")

    df           = load_train_data()
    df           = filter_stable_period(df)
    df_train, df_val, df_test = split_data(df)
    feature_cols = get_feature_cols(df_train)
 
    logger.info("Data load pipeline complete.")
 
    return df_train, df_val, df_test, feature_cols


if __name__=="__main__":
    print("data_loader.py — Sanity Check")
    df_train, df_val, df_test, feature_cols = load_all()
    print(f"\nTrain shape   : {df_train.shape}")
    print(f"Val   shape   : {df_val.shape}")
    print(f"Test  shape   : {df_test.shape}")
    print(f"Feature cols  : {len(feature_cols)}")
    print(f"First 5 feats : {feature_cols[:5]}")
    print(f"Last  5 feats : {feature_cols[-5:]}")


    # Check target column exists
    assert CFG.data.target in df_train.columns, \
        f"Target column '{CFG.data.target}' not found"
    print(f"\nTarget column '{CFG.data.target}' found OK")

    # Check weight column exists
    assert CFG.data.weight_col in df_train.columns, \
        f"Weight column '{CFG.data.weight_col}' not found"
    print(f"Weight column '{CFG.data.weight_col}' found OK")
 
    # Load metadata
    features_meta   = load_feature_meta()
    responders_meta = load_responders_meta()
    print(f"\nFeatures  metadata : {features_meta.shape}")
    print(f"Responders metadata: {responders_meta.shape}")
