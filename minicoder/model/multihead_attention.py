"""چندسر علّی: تفکیک بردارها به سرها و ترکیب دوباره با تصویر خروجی."""

import torch
from torch import nn

from minicoder.config import DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LENGTH, NUM_HEADS
from .attention import _causal_attention, _validate_attention_input
from .feedforward import ManualLinear, _positive_dimension


class MultiHeadCausalAttention(nn.Module):
    def __init__(
        self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
        max_context_length=MAX_CONTEXT_LENGTH,
    ):
        super().__init__()
        _positive_dimension("embedding_dim", embedding_dim)
        _positive_dimension("num_heads", num_heads)
        _positive_dimension("max_context_length", max_context_length)
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_context_length = max_context_length
        self.query = ManualLinear(embedding_dim, embedding_dim)
        self.key = ManualLinear(embedding_dim, embedding_dim)
        self.value = ManualLinear(embedding_dim, embedding_dim)
        self.output_projection = ManualLinear(embedding_dim, embedding_dim)
        self.register_buffer(
            "causal_mask",
            torch.ones(max_context_length, max_context_length, dtype=torch.bool, device=DEVICE).tril(),
            persistent=False,
        )

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def project_qkv(self, inputs):
        _validate_attention_input(
            inputs, self.embedding_dim, self.max_context_length, self.query.weight.dtype
        )
        return self.query(inputs), self.key(inputs), self.value(inputs)

    def split_heads(self, tensor):
        _validate_attention_input(
            tensor, self.embedding_dim, self.max_context_length, self.query.weight.dtype
        )
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("heads must be a torch.Tensor")
        if tensor.ndim != 4 or tensor.shape[1] != self.num_heads or tensor.shape[-1] != self.head_dim:
            raise ValueError("heads must have shape [batch_size, num_heads, sequence_length, head_dim]")
        batch, _, length, _ = tensor.shape
        merged = tensor.transpose(1, 2).contiguous().reshape(batch, length, self.embedding_dim)
        _validate_attention_input(
            merged, self.embedding_dim, self.max_context_length, self.query.weight.dtype
        )
        return merged

    def forward(self, inputs, *, return_attention=False):
        query, key, value = (self.split_heads(tensor) for tensor in self.project_qkv(inputs))
        length = inputs.shape[1]
        heads, weights = _causal_attention(query, key, value, self.causal_mask[:length, :length])
        output = self.output_projection(self.merge_heads(heads))
        return (output, weights) if return_attention else output
