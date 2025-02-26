"""
Model Config

Specifies the hyperparameters for the Pico model/model architecture.
"""

from dataclasses import dataclass
from typing import Optional

from ._constants import VOCAB_SIZE, BATCH_SIZE, MAX_SEQ_LEN


@dataclass
class ModelConfig:
    model_type: str = "pico_decoder"

    # Pico Decoder Defaults

    d_model: int = 768
    n_layers: int = 12

    vocab_size: int = VOCAB_SIZE
    batch_size: int = BATCH_SIZE
    max_seq_len: int = MAX_SEQ_LEN

    attention_n_heads: int = 12
    attention_n_kv_heads: Optional[int] = 4

    activation_hidden_dim: int = 3072

    norm_eps: float = 1e-6

    position_emb_theta: float = 10000.0

    # Can be one of "spectral_weight", "frobenius_loss", "orthogonality_loss"
    rank_normalization_strategy: str = "none"

    # Hyperparameter for the rank normalization loss
    rank_normalization_loss_weight: float = 0.0
