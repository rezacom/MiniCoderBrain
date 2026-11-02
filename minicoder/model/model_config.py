"""قرارداد تغییرناپذیر معماری GPT؛ اندازهٔ واژگان از توکن‌ساز خوانده می‌شود."""

from dataclasses import dataclass, field
import math

from minicoder.config import (
    EMBEDDING_DIM, HIDDEN_DIM, LAYER_NORM_EPS, MAX_CONTEXT_LENGTH, NUM_HEADS, NUM_LAYERS,
)
from minicoder.tokenizer import PAD_ID, ByteTokenizer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = field(default_factory=lambda: ByteTokenizer().vocab_size)
    embedding_dim: int = EMBEDDING_DIM
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    hidden_dim: int = HIDDEN_DIM
    max_context_length: int = MAX_CONTEXT_LENGTH
    layer_norm_eps: float = LAYER_NORM_EPS
    ignore_index: int = PAD_ID
    tie_weights: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "embedding_dim", "num_heads", "num_layers", "hidden_dim", "max_context_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if (isinstance(self.layer_norm_eps, bool) or not isinstance(self.layer_norm_eps, (float, int))
                or not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0):
            raise ValueError("layer_norm_eps must be finite and positive")
        if type(self.ignore_index) is not int:
            raise TypeError("ignore_index must be an integer")
        if type(self.tie_weights) is not bool:
            raise TypeError("tie_weights must be a bool")
