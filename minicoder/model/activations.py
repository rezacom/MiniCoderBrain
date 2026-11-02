"""توابع فعال‌سازی بدون پارامتر، با عملیات تنسوری و مشتق‌گیری خودکار."""

import math

import torch
from torch import nn

from minicoder.config import DEVICE


def _validate_activation_input(inputs: torch.Tensor) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if not torch.is_floating_point(inputs):
        raise TypeError("inputs must have a floating-point dtype")
    if inputs.device.type != DEVICE:
        raise ValueError("inputs must be on CPU")


class ReLU(nn.Module):
    """مقدارهای منفی را صفر می‌کند؛ مشتق در صفر را نیز صفر انتخاب می‌کنیم."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_activation_input(inputs)
        return torch.where(inputs > 0, inputs, torch.zeros_like(inputs))


class GELU(nn.Module):
    """نسخهٔ دقیق GELU: ضرب ورودی در تابع توزیع تجمعی نرمال استاندارد."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_activation_input(inputs)
        return 0.5 * inputs * (1.0 + torch.erf(inputs / math.sqrt(2.0)))
