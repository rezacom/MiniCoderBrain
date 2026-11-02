"""توجه علّی تک‌سر با ضرب ماتریسی، ماسک صریح و softmax پایدار."""

import math

import torch
from torch import nn

from minicoder.config import DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LENGTH
from .feedforward import ManualLinear, _positive_dimension


def _validate_attention_input(inputs, embedding_dim, max_context_length, dtype):
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.ndim != 3 or inputs.shape[-1] != embedding_dim:
        raise ValueError(f"inputs must have shape [batch_size, sequence_length, {embedding_dim}]")
    if inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("batch size and sequence length must be positive")
    if inputs.shape[1] > max_context_length:
        raise ValueError(f"sequence length exceeds max_context_length={max_context_length}")
    if not torch.is_floating_point(inputs) or inputs.dtype != dtype:
        raise TypeError("inputs must have the same floating-point dtype as the weights")
    if inputs.device.type != DEVICE:
        raise ValueError("inputs must be on CPU")
    if not torch.isfinite(inputs).all():
        raise ValueError("inputs must contain only finite values")


def _causal_attention(query, key, value, allowed_mask):
    """هستهٔ مشترک: ابعاد پیشرو می‌توانند دسته یا دسته و سر باشند."""
    if not all(torch.isfinite(tensor).all() for tensor in (query, key, value)):
        raise ValueError("Q, K and V must be finite; projection overflow detected")
    # با توجه به سقف کوچک زمینه روی CPU، محاسبهٔ داخلی دقیق‌تر کم‌هزینه است.
    # این تبدیل مانع سرریز ضرب QK در float32 برای ورودی‌های بزرگ می‌شود.
    q, k, v = (tensor.to(torch.float64) for tensor in (query, key, value))
    scores = torch.matmul(q / math.sqrt(query.shape[-1]), k.transpose(-2, -1))
    if not torch.isfinite(scores).all():
        raise ValueError("attention scores overflowed; input magnitudes are too large")
    # هر سطر دست‌کم خانهٔ قطری خود را دارد؛ هیچ سطر کاملاً مسدود نیست.
    scores = scores.masked_fill(~allowed_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output.to(value.dtype), weights.to(value.dtype)


class CausalSelfAttention(nn.Module):
    """هر توکن فقط Valueهای خودش و توکن‌های پیشین را ترکیب می‌کند."""

    def __init__(self, embedding_dim=EMBEDDING_DIM, max_context_length=MAX_CONTEXT_LENGTH):
        super().__init__()
        _positive_dimension("embedding_dim", embedding_dim)
        _positive_dimension("max_context_length", max_context_length)
        self.embedding_dim = embedding_dim
        self.max_context_length = max_context_length
        self.query = ManualLinear(embedding_dim, embedding_dim)
        self.key = ManualLinear(embedding_dim, embedding_dim)
        self.value = ManualLinear(embedding_dim, embedding_dim)
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

    def forward(self, inputs, *, return_attention=False):
        query, key, value = self.project_qkv(inputs)
        length = inputs.shape[1]
        output, weights = _causal_attention(query, key, value, self.causal_mask[:length, :length])
        return (output, weights) if return_attention else output
