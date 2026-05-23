"""
(968, 1, 125)
 │    │   └── 125 total features (76 original + 49 engineered)
 │    └─────  1 day at a time (batch size = 1)
 └──────────  968 time steps within that day

 
Model A — 3 layer GRU:
 Input (968, 1, 125)
→ GRU layer 1  (125 → 64 hidden)
→ GRU layer 2  (64 → 64 hidden)
→ GRU layer 3  (64 → 64 hidden)
→ Final hidden state (1, 64)
→ Linear layer (64 → 5 outputs)  ← one per target

Model B — 1 layer GRU + 2 linear:
Input (968, 1, 125)
→ GRU layer 1  (125 → 64 hidden)
→ Final hidden state (1, 64)
→ Linear layer 1 (64 → 64) + ReLU + Dropout
→ Linear layer 2 (64 → 64) + ReLU + Dropout
→ Linear layer 3 (64 → 5 outputs)  ← one per target

output[0] = responder_6   ← primary target
output[1] = responder_7
output[2] = responder_8
output[3] = responder_9
output[4] = responder_10


Both models:
    - Take input shape  : (sequence_length, batch_size, input_size)
    - Output shape      : (sequence_length, batch_size, n_targets)
    - Have 5 output heads: responder_6, 7, 8, 9, 10
"""


import logging
import torch
import torch.nn as nn
from typing import Tuple 
from config import CFG


logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("model")
 


 #MODEL A — 3-LAYER GRU
 #Stacks 3 GRU layers on top of each other.Each layer learns increasingly abstract temporal patterns.
class GRUModelA(nn.Module):
    def __init__(
        self,
        input_size  : int   = None,
        hidden_size : int   = CFG.model.hidden_size,
        n_targets   : int   = CFG.model.n_heads,
        dropout     : float = CFG.model.dropout,
        n_layers    : int   = CFG.model.gru_a_num_layers,
    ):
        super(GRUModelA, self).__init__()

        if input_size is None:
            raise ValueError("input_size must be provided")

        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = False,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_targets)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        # predictions shape: (seq_len, batch_size, n_targets)
        predictions = self.output_layer(gru_out)

        return predictions, hidden
    


#MODEL B — 1-LAYER GRU + 2 LINEAR
class GRUModelB(nn.Module):
    def __init__(
            self,
            input_size  : int   = None,
            hidden_size : int   = CFG.model.hidden_size,
            linear_dim  : int   = CFG.model.gru_b_linear_dim,
            n_targets   : int   = CFG.model.n_heads,
            dropout     : float = CFG.model.dropout,
            n_layers    : int   = CFG.model.gru_a_num_layers,
    ):
        super(GRUModelB, self).__init__()

        if input_size is None:
            raise ValueError("input_size must be provided")

        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = self.n_layers,
            batch_first = False,
            dropout     = 0.0,
        )

        self.linear1  = nn.Linear(hidden_size, linear_dim)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2  = nn.Linear(linear_dim, linear_dim)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.output_layer = nn.Linear(linear_dim, n_targets)

    def forward(
        self,
        x  : torch.Tensor,
        h0 : torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        gru_out, hidden = self.gru(x)
        # Applied to every time step independently
        out = self.linear1(gru_out)     # (seq_len, batch, linear_dim)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.linear2(out)         # (seq_len, batch, linear_dim)
        out = self.relu2(out)
        out = self.dropout2(out)
        predictions = self.output_layer(out)
        return predictions, hidden



def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
 
    return device



def build_model(
        architecture: str,
        input_size: int,
        seed: int = 42) -> nn.Module:

    torch.manual_seed(seed)

    if architecture == "A":
        model = GRUModelA(input_size=input_size)
        logger.info(
            f"Built GRUModelA — "
            f"input={input_size}, hidden={CFG.model.hidden_size}, "
            f"layers={CFG.model.gru_a_num_layers}, seed={seed}"
        )

    elif architecture == "B":
        model = GRUModelB(input_size=input_size)
        logger.info(
            f"Built GRUModelB — "
            f"input={input_size}, hidden={CFG.model.hidden_size}, "
            f"linear={CFG.model.gru_b_linear_dim}, seed={seed}"
        )
    else:
        raise ValueError(f"Unknown architecture: '{architecture}'. Use 'A' or 'B'.")

    device = get_device()
    model = model.to(device)
    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    return model