"""Key idea:
    Instead of processing one time step at a time (like GRU),
    PatchTST divides the sequence into patches and processes
    them ALL IN PARALLEL using a Transformer encoder with
    multi-head self-attention"""


import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from model import get_device
from config import CFG

# LOGGER
# ─────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("patchtst")





class PatchEmbeddings(nn.Module):
    """ sequence_length = 968 time steps
        patch_size      = 16 time steps per patch
        stride          = 8  (overlapping patches)
        n_patches       = (968 - 16) / 8 + 1 = 120 patches"""
    def __init__(self,input_size:int,patch_size:int,stride:int,d_model:int,dropout:float=0.1):
        super().__init__()

        self.patch_size=patch_size
        self.stride=stride

        self.projection=nn.Linear(patch_size*input_size,d_model)
        self.dropout=nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and embed them.

        #input (max_len, 1, input_size(n features))
        #output (n_patches,1,d_model)


        """
        # Transpose to (batch_size, seq_len, input_size) for unfolding
        x=x.permute(1,0,2)
        batch_size  = x.shape[0]
        seq_len     = x.shape[1]
        input_size  = x.shape[2]
        # Result shape: (batch, n_patches, patch_size, input_size)
        patches = x.unfold(1, self.patch_size, self.stride)
        n_patches = patches.shape[1]

        # Flatten each patch: (batch, n_patches, patch_size × input_size)
        patches = patches.contiguous().view(batch_size, n_patches, -1)

        # Project to d_model: (batch, n_patches, d_model)
        embedded = self.projection(patches)
        embedded = self.dropout(embedded)

        # Transpose back to (n_patches, batch, d_model) for Transformer
        embedded = embedded.permute(1, 0, 2)

        return embedded
"""(968, 1, 125)
|
(1, 968, 125)
|
(1, 120, 125, 16)
|
(1, 120, 2000)
|
(1, 120, 64)
|
(120, 1, 64)"""




# 2.  POSITIONAL ENCODING
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model  : int,
        dropout  : float = 0.1,
        max_len  : int   = 500,
    ):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        # Shape: (max_len, 1, d_model) — broadcast over batch dimension
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)   # not a parameter, but moves with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #torch.Tensor: Same shape with positional info added.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)





    # ─────────────────────────────────────────────
# 3.  PATCHTST MODEL
# ─────────────────────────────────────────────

class PatchTST(nn.Module):
    def __init__(
        self,
        input_size : int   = None,
        patch_size : int   = 16,
        stride     : int   = 8,
        d_model    : int   = 64,
        n_heads    : int   = 4,
        n_layers   : int   = 2,
        d_ff       : int   = 256,
        n_targets  : int   = CFG.model.n_heads,
        dropout    : float = 0.1,
        seq_len    : int   = CFG.data.n_time_ids,
    ):
        super().__init__()

        if input_size is None:
            raise ValueError(
                "input_size must be provided — "
                "set it to len(all_feature_cols) after feature engineering"
            )

        self.patch_size = patch_size
        self.stride     = stride
        self.d_model    = d_model
        # Compute number of patches
        self.n_patches = (seq_len - patch_size) // stride + 1

        self.patch_embedding=PatchEmbeddings(
            input_size = input_size,
            patch_size = patch_size,
            stride     = stride,
            d_model    = d_model,
            dropout    = dropout,
        )

        self.pos_encoding = PositionalEncoding(
            d_model = d_model,
            dropout = dropout,
            max_len = self.n_patches + 10,   # +10 for safety margin
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = False,   # (seq, batch, features)
            norm_first      = True,    # Pre-LN: more stable training
        )

        self.transformer=nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = n_layers,
        )

        # Flatten all patch embeddings → predict n_targets
        self.flatten     = nn.Flatten(start_dim=0, end_dim=0)

        self.output_layer = nn.Linear(d_model, n_targets)


    def forward(
        self,
        x  : torch.Tensor,
        h0 : torch.Tensor = None
    ) -> Tuple[torch.Tensor, None]:

        seq_len = x.size(0)

        # (seq_len, batch, input_size) → (n_patches, batch, d_model)
        patches = self.patch_embedding(x)

        # ── Positional encoding ───────────────────────────────────
        patches = self.pos_encoding(patches)

        # ── Transformer encoder ───────────────────────────────────
        # (n_patches, batch, d_model) → (n_patches, batch, d_model)
        encoded = self.transformer(patches)
        """GRU output shape    : (968, 1, 64)  ← one output per time step
PatchTST output shape: (120, 1, 64)  ← one output per patch"""
        # encoded shape: (n_patches, batch, d_model)
        # Transpose for interpolation: (batch, d_model, n_patches)
        encoded_t = encoded.permute(1, 2, 0)

        # Interpolate from n_patches back to seq_len
        # (batch, d_model, n_patches) → (batch, d_model, seq_len)
        upsampled = F.interpolate(
            encoded_t,
            size   = seq_len,
            mode   = "linear",
            align_corners = False,
        )
        # Transpose back: (seq_len, batch, d_model)
        upsampled = upsampled.permute(2, 0, 1)
        predictions = self.output_layer(upsampled)

        return predictions, None   # None = no hidden state

"""(120, 1, 64)
|
(1, 64, 120)
|
(1, 64, 968)
|
(968, 1, 64)
|
(968, 1, 5)"""

# 4.  BUILD PATCHTST — FACTORY FUNCTION
# ─────────────────────────────────────────────

def build_patchtst(
    input_size : int,
    seed       : int = 42,
) -> nn.Module:
    """
    Factory function — builds and returns a PatchTST model."""

    torch.manual_seed(seed)
    model = PatchTST(input_size=input_size)
    device   = get_device()
    model    = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Built PatchTST — "
        f"input={input_size}, d_model={model.d_model}, "
        f"n_patches={model.n_patches}, seed={seed}"
    )
    logger.info(f"PatchTST parameters: {n_params:,}")

    return model



# 5.  SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("patchtst.py — Sanity Check")
    print("=" * 55)

    INPUT_SIZE = 125
    SEQ_LEN    = 968
    BATCH_SIZE = 1
    N_TARGETS  = 5

    device = get_device()

    # Build model
    model = build_patchtst(input_size=INPUT_SIZE, seed=42)

    print(f"\nModel architecture:")
    print(f"  patch_size : {model.patch_size}")
    print(f"  stride     : {model.stride}")
    print(f"  n_patches  : {model.n_patches}")
    print(f"  d_model    : {model.d_model}")

    # Dummy input
    x = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE).to(device)

    # Forward pass
    preds, hidden = model(x)

    print(f"\nInput shape      : {x.shape}")
    print(f"Output shape     : {preds.shape}")
    print(f"Hidden state     : {hidden}")   # should be None

    # Assertions
    assert preds.shape == (SEQ_LEN, BATCH_SIZE, N_TARGETS), \
        f"Wrong output shape: {preds.shape}"
    assert hidden is None, "PatchTST should return None for hidden state"
    print("\nShape assertions ✓")
    print("Hidden state is None ✓ (no recurrent state in Transformer)")

    # Test GRU API compatibility
    # Passing h0 should be accepted and ignored
    h0    = torch.zeros(1, BATCH_SIZE, 64).to(device)
    preds2, _ = model(x, h0)
    assert preds2.shape == (SEQ_LEN, BATCH_SIZE, N_TARGETS)
    print("GRU API compatibility ✓ (h0 accepted and ignored)")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPatchTST parameters : {n_params:,}")

    print("\n" + "=" * 55)
    print("patchtst.py — All checks passed.")
    print("=" * 55)
