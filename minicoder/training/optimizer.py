"""AdamW با نرخ قابل تنظیم و بدون واپاشی وزن برای بایاس و LayerNorm."""

import math

import torch


def create_optimizer(model, *, learning_rate=3e-4, weight_decay=0.01):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("نرخ یادگیری باید مثبت و متناهی باشد.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("واپاشی وزن باید نامنفی و متناهی باشد.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups = [
        {"params": [parameter for parameter in parameters if parameter.ndim >= 2], "weight_decay": weight_decay},
        {"params": [parameter for parameter in parameters if parameter.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))
