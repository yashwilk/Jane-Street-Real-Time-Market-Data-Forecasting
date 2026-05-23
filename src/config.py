from pathlib import Path

from dataclasses import dataclass, field
from typing import List

ROOT_DIR=Path(__file__).parent.parent #goes to main folder, same as '../

@dataclass
class Config:
    data_dir:Path=ROOT_DIR/'data'
    train_path: Path=ROOT_DIR/'data'/'train.parquet'
    lags_path: Path=ROOT_DIR/'data'/'lags.parquet'
    feature_csv: Path=ROOT_DIR/'data'/'features.csv'
    responder_csv: Path=ROOT_DIR/'data'/'responders.csv'
    output_dir: Path=ROOT_DIR/'outputs'
    model_dir: Path=ROOT_DIR/'outputs'/'models'
    prediction_dir: Path=ROOT_DIR/'outputs'/'predictions'
    plots_dir: Path=ROOT_DIR/'outputs'/'plots'

    def create_dirs(self)-> None:
        for path in[self.output_dir,self.model_dir,self.prediction_dir,self.plots_dir]:
            path.mkdir(exist_ok=True,parents=True)

@dataclass
class DataConfig:
    date_start: int=700
    train_end: int=1298
    val_start: int=1299
    val_end: int=1498
    test_start: int=1499
    id_col: List[str]=field(default_factory=lambda: ["date_id","time_id","symbol_id"])
    target: str="responder_6"
    drop_features:List[str]=field(default_factory=lambda:["feature_09", "feature_10", "feature_11"])
    all_responders: list[str]=field(default_factory=lambda:[f"responder_{i}"for i in range(9)])
    aux_targets: list[str]=field(default_factory=lambda:["responder_7","responder_8","responder_9","responder_10"]) #9 and 10  engineered: ~8-day  # engineered: ~60-day
    weight_col:str="weight"
    n_time_ids:int=968    # Number of time_ids per day after date_id 700


@dataclass
class TrainingConfig:
    learning_rate: float=0.0005
    max_epochs: int=50
    early_stop: int=5
    seed:list[int]=field(default_factory=lambda:[42, 2023, 1234])
    device: str ="cpu"


@dataclass
class OnlineConfig:
    """After each trading day, when true labels become available,
    we perform one gradient update step to adapt to market drif"""
    learning_rate: float=0.0003 #lower than training
    target: str='responder_6'
    update_every: int=1


@dataclass
class FeaturesConfig:
    n_top_features      : int       = 16
    rolling_window      : int       = 1000
    add_time_id_feature : bool      = True
    top_features        : List[str] = field(default_factory=list)


@dataclass
class LGBMConfig:
    n_estimators    : int   = 1000
    learning_rate   : float = 0.05
    num_leaves      : int   = 31
    max_depth       : int   = -1        # -1 = no limit
    min_child_samples: int  = 20
    subsample       : float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha       : float = 0.1       # L1 regularisation
    reg_lambda      : float = 0.1       # L2 regularisation
    random_state    : int   = 42
    n_jobs          : int   = -1        # use all CPU cores
    verbose         : int   = -1


@dataclass
class ModelConfig:
    hidden_size      : int   = 64
    n_heads          : int   = 5    # responder_6 through responder_10
    dropout          : float = 0.1
    gru_a_num_layers : int   = 3    # Model A: 3-layer GRU


@dataclass
class Configs:
    paths   : Config        = field(default_factory=Config)
    data    : DataConfig    = field(default_factory=DataConfig)
    train   : TrainingConfig= field(default_factory=TrainingConfig)
    online  : OnlineConfig  = field(default_factory=OnlineConfig)
    lgbm    : LGBMConfig    = field(default_factory=LGBMConfig)
    features: FeaturesConfig= field(default_factory=FeaturesConfig)
    model   : ModelConfig   = field(default_factory=ModelConfig)

    def __post_init__(self):
        """Automatically create output directories when config is instantiated."""
        self.paths.create_dirs()


CFG = Configs()

if __name__ == "__main__":
    print("=" * 50)
    print("Jane Street — Config Sanity Check")
    print("=" * 50)
    print(f"Project root      : {ROOT_DIR}")
    print(f"Train data path   : {CFG.paths.train_path}")
    print(f"Data start        : date_id {CFG.data.date_start}")
    print(f"Train end         : date_id {CFG.data.train_end}")
    print(f"Val   range       : date_id {CFG.data.val_start} -> {CFG.data.val_end}")
    print(f"Test  range       : date_id {CFG.data.test_start} -> end")
    print(f"Target            : {CFG.data.target}")
    print(f"Aux targets       : {CFG.data.aux_targets}")
    print(f"Training seeds    : {CFG.train.seed}")
    print(f"Online LR         : {CFG.online.learning_rate}")
    print(f"Output dir        : {CFG.paths.output_dir}")
    print("=" * 50)
    print("Config loaded successfully.")
