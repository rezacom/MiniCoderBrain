"""بلوک ترنسفورمر با نرمال‌سازی پیش از هر شاخه و دو اتصال باقیمانده."""

import math

from torch import nn

from minicoder.config import (
    DEVICE, EMBEDDING_DIM, HIDDEN_DIM, LAYER_NORM_EPS, MAX_CONTEXT_LENGTH, NUM_HEADS,
)
from .attention import _validate_attention_input
from .feedforward import FeedForward
from .multihead_attention import MultiHeadCausalAttention


class TransformerBlock(nn.Module):
    def __init__(
        self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM,
        max_context_length=MAX_CONTEXT_LENGTH, layer_norm_eps=LAYER_NORM_EPS,
    ):
        super().__init__()
        if (isinstance(layer_norm_eps, bool) or not isinstance(layer_norm_eps, (int, float))
                or not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0):
            raise ValueError("layer_norm_eps must be finite and positive")
        self.attention = MultiHeadCausalAttention(embedding_dim, num_heads, max_context_length)
        self.feedforward = FeedForward(embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps, device=DEVICE)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps, device=DEVICE)
        self.embedding_dim = embedding_dim
        self.max_context_length = max_context_length

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, inputs):
        _validate_attention_input(
            inputs, self.embedding_dim, self.max_context_length, self.norm1.weight.dtype
        )
        hidden = inputs + self.attention(self.norm1(inputs))
        return hidden + self.feedforward(self.norm2(hidden))
