"""لایهٔ خطی دستی و شبکهٔ پیش‌خورِ مستقل برای هر جایگاه دنباله."""

import math

import torch
from torch import nn

from minicoder.config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM
from .activations import GELU


def _positive_dimension(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class ManualLinear(nn.Module):
    """تبدیل Y = XW + b؛ شکل وزن برابر ورودی × خروجی است."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        _positive_dimension("in_features", in_features)
        _positive_dimension("out_features", out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features, device=DEVICE))
        self.bias = nn.Parameter(torch.empty(out_features, device=DEVICE))
        # محدود کردن اندازهٔ اولیهٔ وزن با توجه به تعداد ورودی‌ها.
        bound = 1.0 / math.sqrt(in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim not in (2, 3):
            raise ValueError("inputs must be 2D or 3D with features on the last axis")
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"last input dimension must equal in_features={self.in_features}")
        if not torch.is_floating_point(inputs) or inputs.dtype != self.weight.dtype:
            raise TypeError("inputs must have the same floating-point dtype as the weights")
        if inputs.device.type != DEVICE:
            raise ValueError("inputs must be on CPU")
        return torch.matmul(inputs, self.weight) + self.bias


class FeedForward(nn.Module):
    """گسترش بُعد، فعال‌سازی GELU و بازگرداندن بُعد برای هر توکن."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.linear1 = ManualLinear(embedding_dim, hidden_dim)
        self.activation = GELU()
        self.linear2 = ManualLinear(hidden_dim, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self.linear1.in_features

    @property
    def hidden_dim(self) -> int:
        return self.linear1.out_features

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch_size, sequence_length, embedding_dim]")
        return self.linear2(self.activation(self.linear1(inputs)))
