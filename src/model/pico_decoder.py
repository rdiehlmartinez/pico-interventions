"""
Pico Decoder: A Lightweight Causal Transformer Language Model

Pico Decoder uses a simple LLAMA-style transformer architecture, written for clarity and educational purposes.

Everything is written with a modular design for easy modification and experimentation.

Key features:
- RMSNorm for layer normalization
- Rotary Positional Embeddings (RoPE)
- Multi-head attention with KV-cache support
- SwiGLU activation function
- Residual connections throughout

- KV-cache for faster autoregressive generation

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - LLAMA: https://arxiv.org/abs/2302.13971

Adapted from:
    - OLMO: https://github.com/allenai/OLMo
    - LLAMA: https://github.com/meta/llama
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import spectral_norm

from torch.nn.attention import sdpa_kernel, SDPBackend

from dataclasses import asdict

import os

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutput

# typing imports
from typing import Union, Tuple, Optional, TYPE_CHECKING, Dict, Any

from safetensors import safe_open

try:
    if TYPE_CHECKING:
        # We need to do this to avoid importing these when creating the HF-compatible models
        from src.config import ModelConfig
except ImportError:
    pass

########################################################
#
# Layer Normalization
#
########################################################


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    A variant of Layer Normalization that uses RMS statistics instead of mean/variance,
    resulting in improved stability and performance.

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Configuration object containing normalization parameters
            - config.norm_eps: Small constant for numerical stability
            - config.d_model: Model dimension for the weight parameter

    References:
        https://arxiv.org/abs/1910.07467
    """

    def __init__(self, config: Union["ModelConfig", "PicoDecoderHFConfig"]):
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor by its RMS value.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor and scales it by the weight parameter.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


########################################################
#
# Positional Embedding
#
########################################################


class RoPE(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Implements position-dependent rotation of keys and queries in attention mechanism,
    allowing better modeling of relative positions in sequences. Uses complex number
    operations for efficient rotation.

    Args:
        config (Union[ModelConfig, PicoHFConfig]): Model configuration containing:
            - config.position_emb_theta: Base for frequency computation
            - config.d_model: Model dimension
            - config.attention_n_heads: Number of attention heads
            - config.max_seq_len: Maximum sequence length

    References:
        https://arxiv.org/abs/2104.09864
    """

    _freqs_cis_tensor: torch.Tensor | None = None

    def __init__(self, config: Union["ModelConfig", "PicoDecoderHFConfig"]):
        super().__init__()

        self.theta = config.position_emb_theta
        self.dim = config.d_model // config.attention_n_heads

        max_seq_len = config.max_seq_len

        # only gets set once, and then reused for all RoPE instances
        if RoPE._freqs_cis_tensor is None:
            RoPE._freqs_cis_tensor = self._setup_freqs_cis(
                max_seq_len, self.theta, self.dim
            )

        # register _freqs_cis buffer
        # can be easily recomputed so persistent=False
        self.register_buffer("_freqs_cis", self._freqs_cis_tensor, persistent=False)

    @classmethod
    def _setup_freqs_cis(cls, seq_len: int, theta: float, dim: int) -> torch.Tensor:
        """Setup Frequency Tensor for RoPE Embeddings

        Initializes the complex frequency tensor that is used to compute the RoPE embeddings.

        Note other implementations will use cos and sin directly, but using the complex
        number representation is (probably?) more efficient:

            e^(theta * i * t) = cos(theta * t) + i * sin(theta * t) [Euler's formula]
        """
        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, _freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def get_freqs_cis(
        self, input_shape: torch.Size, start_pos: int, end_pos: int
    ) -> torch.Tensor:
        """Reshape Frequency Tensor for RoPE Embeddings

        Makes the frequency tensor broadcastable with the input tensor.
        """
        _freqs_cis = self._freqs_cis[start_pos:end_pos]
        ndim = len(input_shape)
        assert 0 <= 1 < ndim
        assert _freqs_cis.shape == (input_shape[1], input_shape[-1])

        # TODO: Check whether this is correct (might be able to remove this)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(input_shape)]
        return _freqs_cis.view(*shape)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE Embeddings to Queries and Keys

        Applies the rotary positional embeddings to the input tensors via complex num multiplication

        NOTE: The start_pos is used if we want to use the kv_cache in the attention mechanism.
        """
        queries_ = torch.view_as_complex(
            queries.float().reshape(*queries.shape[:-1], -1, 2)
        )
        keys_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))

        input_shape = (
            queries_.shape
        )  # same as keys: (batch_size, seq_len, n_heads, head_dim/2)
        freqs_start_pos = start_pos
        freqs_end_pos = freqs_start_pos + queries_.shape[1]

        freqs_cis = self.get_freqs_cis(input_shape, freqs_start_pos, freqs_end_pos)

        queries_rotated = torch.view_as_real(queries_ * freqs_cis).flatten(3)
        keys_rotated = torch.view_as_real(keys_ * freqs_cis).flatten(3)
        return queries_rotated.type_as(queries), keys_rotated.type_as(keys)


########################################################
#
# Attention
#
########################################################


class Attention(nn.Module):
    """Multi-head Attention with Group Query Attention support.

    Implements scaled dot-product attention and supports:
    - Grouped Query Attention (GQA)
    - Key-Value caching for efficient inference
    - RoPE integration

    Args:
        config (Union[ModelConfig, PretrainedConfig]): Configuration containing:
            - config.attention_n_heads: Number of attention heads
            - config.attention_n_kv_heads: Number of key/value heads
            - config.d_model: Model dimension
            - config.batch_size: Maximum batch size
            - config.max_seq_len: Maximum sequence length

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        config: Union["ModelConfig", "PicoDecoderHFConfig"],
    ):
        super().__init__()

        self.normalization_strategy = config.rank_normalization_strategy

        self.n_heads = config.attention_n_heads
        self.n_kv_heads = config.attention_n_kv_heads

        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len

        d_model = config.d_model
        self.head_dim = d_model // self.n_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        _v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        _o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=False)

        if self.normalization_strategy == "spectral_weight":
            self.v_proj = spectral_norm(_v_proj)
            self.o_proj = spectral_norm(_o_proj)
        else:
            self.v_proj = _v_proj
            self.o_proj = _o_proj

        self.rope = RoPE(config)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for the attention mechanism.

        Computes queries, keys, and values for the attention mechanism. Applies rotary positional
        embeddings to the queries and keys, and then computes attention scores and outputs.

        For an introduction to the attention mechanism, see:
        https://arxiv.org/abs/1706.03762

        A few things to note:
        - The past_key_values is used to implement the KV cache, which is used to speed up
          generation by caching the KV pairs from previous forward passes. This is useful when doing
          tasks that require generating multiple tokens conditioned on previous tokens (e.g. language
          modeling, text generation, etc.). The way the KV cache is implemented is that each layer has
          its own KV cache - this KV cache is implemented as a tuple.
        """
        bsz, seq_len, _ = input.shape
        _queries, _keys, _values = (
            self.q_proj(input),
            self.k_proj(input),
            self.v_proj(input),
        )

        # Reshaping for multi-head attention
        queries = _queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = _keys.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values = _values.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # The start position is used to apply the RoPE embeddings to only the new tokens
        # when using the kv_cache in the attention mechanism.
        # We want to start from the last position in the cache.
        start_pos = past_key_values[0].shape[1] if past_key_values is not None else 0

        # apply rotary positional embeddings
        queries, keys = self.rope(queries, keys, start_pos)

        if past_key_values is not None:
            keys = torch.cat([past_key_values[0], keys], dim=1)
            values = torch.cat([past_key_values[1], values], dim=1)

        if use_cache:
            cached_keys = keys
            cached_values = values
        else:
            cached_keys = None
            cached_values = None

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        apply_gqa = self.n_rep > 1
        if apply_gqa and queries.device.type == "mps":
            # NOTE: MPS does not support GQA in the SDPA kernel, but we can repeat the keys and values
            # outside of the kernel to get the same effect.
            # See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            keys = keys.repeat_interleave(self.n_rep, dim=-3)
            values = values.repeat_interleave(self.n_rep, dim=-3)
            apply_gqa = False

        backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]

        with sdpa_kernel(backends=backends):
            attn_output = F.scaled_dot_product_attention(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                attn_mask=mask.to(queries.dtype),
                enable_gqa=apply_gqa,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, (cached_keys, cached_values)


########################################################
#
# SwiGLU (Combines MLP and Activation)
#
########################################################


class SwiGLU(nn.Module):
    """SwiGLU Activation Function with Linear Projections.

    Implements the SwiGLU activation function combined with linear transformations,
    serving as the feed-forward network in transformer blocks.

    Args:
        config (Union[ModelConfig, PicoDecoderHFConfig]): Configuration containing:
            - config.d_model: Model dimension
            - config.activation_hidden_dim: Hidden dimension (typically 4 * d_model)

    References:
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: Union["ModelConfig", "PicoDecoderHFConfig"]):
        super().__init__()

        model_dim = config.d_model
        act_hidden_dim = config.activation_hidden_dim  # usually 4 * d_model

        self.w_0 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_1 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        _w_2 = nn.Linear(act_hidden_dim, model_dim, bias=False)

        if config.rank_normalization_strategy == "spectral_weight":
            self.w_2 = spectral_norm(_w_2)
        else:
            self.w_2 = _w_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


########################################################
#
# PicoDecoderBlock
#
########################################################


class PicoDecoderBlock(nn.Module):
    """Single Transformer Block with Attention and Feed-forward layers.

    Implements a standard transformer block with:
    - Multi-head attention with normalization and residual connection
    - SwiGLU feed-forward network with normalization and residual connection

    Args:
        config (Union[ModelConfig, PicoDecoderHFConfig]): Model configuration; either a dataclass or
            a HuggingFace PicoDecoderHFConfig
    """

    def __init__(
        self,
        config: Union["ModelConfig", "PicoDecoderHFConfig"],
    ):
        super().__init__()

        self.attention = Attention(config)
        self.swiglu = SwiGLU(config)
        self.attention_norm = RMSNorm(config)
        self.swiglu_norm = RMSNorm(config)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attention_output, cached_key_values = self.attention(
            self.attention_norm(input),
            mask=mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        # NOTE: cached_key_values is None if use_cache is False

        h = input + attention_output
        out = h + self.swiglu(self.swiglu_norm(h))
        return out, cached_key_values


########################################################
#
# Pico Decoder (Causal Transformer Model)
#
########################################################


class PicoDecoder(nn.Module):
    """
    Pico Decoder: combines the embedding, causal decoder blocks, and output projection into a
    single autoregressive model.

    For more information on the model, see the classes for the modules that make up the model.
    """

    TARGET_MODULES = [
        "attention.v_proj",
        "attention.o_proj",
        "swiglu.w_2",
    ]

    def __init__(
        self,
        model_config: Union["ModelConfig", "PicoDecoderHFConfig"],
    ):
        super().__init__()
        self.config = model_config

        self.embedding_proj = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList(
            [PicoDecoderBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.output_norm = RMSNorm(self.config)
        self.de_embedding_proj = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )

    def convert_to_hf_model(self) -> "PicoDecoderHF":
        """Convert the Lightning model to a HuggingFace model."""
        # Create HF config without fabric-specific settings
        hf_config = PicoDecoderHFConfig.from_dataclass(self.config)

        # Create new HF model
        hf_model = PicoDecoderHF(hf_config)

        # Copy state dict, excluding fabric-specific keys
        hf_model.load_state_dict(self.state_dict(prefix="pico_decoder."))

        return hf_model

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Get the orthogonality loss for the model."""

        # compute the orthogonality loss for the model
        total_loss = torch.tensor(0.0)
        for name, module in self.named_modules():
            if any(target_module in name for target_module in self.TARGET_MODULES):
                _weight = module.weight
                _gram_matrix = _weight.T @ _weight

                _orthogonality_loss = torch.norm(
                    _gram_matrix - torch.eye(len(_gram_matrix), device=_weight.device),
                    p="fro",
                )
                if _weight.device != total_loss.device:
                    total_loss = total_loss.to(_weight.device)
                total_loss += _orthogonality_loss

        return total_loss

    def get_frobenius_loss(self) -> torch.Tensor:
        """Get the frobenius loss for the model."""

        total_loss = torch.tensor(0.0)
        # compute the frobenius loss for the model
        for name, module in self.named_modules():
            if any(target_module in name for target_module in self.TARGET_MODULES):
                _weight = module.weight

                _weight_norm = torch.norm(_weight, p="fro")

                if _weight.device != total_loss.device:
                    total_loss = total_loss.to(_weight.device)
                total_loss += _weight_norm

        return total_loss

    def get_normalization_loss(self) -> torch.Tensor:
        """Get the normalization loss for the model."""
        if self.normalization_strategy == "orthogonality_loss":
            return self.get_orthogonality_loss()
        elif self.normalization_strategy == "frobenius_loss":
            return self.get_frobenius_loss()
        else:
            raise NotImplementedError(
                f"Normalization strategy {self.normalization_strategy} not implemented"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        This is the forward pass for the entire Pico model. It boils down to:
        - Embedding the input ids
        - Creating a causal mask
        - Processing through the pico layers
        - Projecting the output to logits

        NOTE: One feature that might be confusing is the KV cache. The KV cache is used to speed up
        generation by caching the KV pairs from previous forward passes. This is useful when doing
        tasks that require generating multiple tokens conditioned on previous tokens (e.g. language
        modeling, text generation, etc.). The way the KV cache is implemented is that each layer has
        its own KV cache which is stored as a tuple. The whole model then stores a tuple of these
        KV caches (so a tuple of tuples).
        """

        seq_len = input_ids.shape[-1]
        h = self.embedding_proj(input_ids)

        # Calculate start position from past cached KV pairs. Remember that each layer has its
        # own KV Cache. So when we index past_key_values, we need to index into the KV pairs for the
        # correct layer and then for either the keys or values.
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]

        # Create causal mask for current sequence
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)

            # If using KV cache, extend mask to cover cached sequence length
            if past_key_values is not None:
                # Add zeros for cached tokens (we can attend to all of them)
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask])

            mask = mask.to(h.device)

        # NOTE: If we are using the cache, we need to store the cached KV pairs for each layer
        #       in a tuple. Each layer will have its own cached KV pair which we aggregate in a tuple.
        cached_key_values = () if use_cache else None

        # Process through transformer blocks
        for idx, layer in enumerate(self.layers):
            layer_past_key_values = (
                past_key_values[idx] if past_key_values is not None else None
            )

            h, layer_cached_key_values = layer(
                h, mask=mask, past_key_values=layer_past_key_values, use_cache=use_cache
            )

            if use_cache:
                cached_key_values += (layer_cached_key_values,)

        # Final norm and projection
        h = self.output_norm(h)
        logits = self.de_embedding_proj(h).float()

        return logits, cached_key_values


########################################################
#
# HuggingFace Wrapper
#
########################################################

"""
HuggingFace wrapper for the Pico model.

Many evaluation frameworks require a model be setup as a HuggingFace model, so we provide a simple
wrapper that does just that. When we save checkpoints of the Pico model, we save both the normal
Pico model as well as the model wrapped in this HuggingFace class.

This also lets you do cool things like: 

`model = AutoModelForCausalLM.from_pretrained("path/to/checkpoint")`
"""


class PicoDecoderHFConfig(PretrainedConfig):
    """HuggingFace config for Pico model."""

    model_type = "pico_decoder"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PicoDecoderHFConfig":
        # NOTE The typical from_dict method doesn't actually set the attributes unless they are
        # defined in the constructor.

        pico_config = cls(**kwargs)

        # Because this class is just a wrapper around the ModelConfig dataclass, we need to do
        # a little extra work to ensure that the attributes are actually set.
        for key, value in config_dict.items():
            setattr(pico_config, key, value)

        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        unused_kwargs = {
            key: value for key, value in kwargs.items() if not hasattr(pico_config, key)
        }

        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config

    @classmethod
    def from_dataclass(cls, model_config: "ModelConfig"):
        return cls.from_dict(asdict(model_config))


class PicoDecoderHF(PreTrainedModel):
    """HuggingFace wrapper for Pico model."""

    config_class = PicoDecoderHFConfig
    _no_split_modules = ["PicoBlock", "Attention", "SwiGLU", "RMSNorm"]

    def __init__(self, config: PicoDecoderHFConfig):
        super().__init__(config)
        self.pico_decoder = PicoDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[CausalLMOutput, CausalLMOutputWithPast]:
        """HuggingFace forward pass wrapper.

        Forwards pass for the HuggingFace version of the Pico Model. Basic wrapper around the
        Pico model's forward pass, and returns the output as a HuggingFace CausalLMOutput.
        """
        logits, past_key_values = self.pico_decoder(
            input_ids, past_key_values, use_cache
        )
        if use_cache:
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
            )
        else:
            return CausalLMOutput(
                logits=logits,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = PicoDecoderHFConfig.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        safetensor_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"Model file not found at {safetensor_path}")

        # load the checkpoint
        checkpoint = safe_open(safetensor_path, framework="pt")
        _state_dict = {}
        for key in checkpoint.keys():
            _state_dict[key] = checkpoint.get_tensor(key)

        model = cls(config)
        model.load_state_dict(_state_dict, strict=False)
        return model


# Register for auto classes
PicoDecoderHFConfig.register_for_auto_class()
PicoDecoderHF.register_for_auto_class("AutoModel")
PicoDecoderHF.register_for_auto_class("AutoModelForCausalLM")
