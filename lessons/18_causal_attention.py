"""درس ۱۸: بررسی عملیِ جلوگیری از دسترسی به آینده."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import CausalSelfAttention


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    attention = CausalSelfAttention(embedding_dim=4, max_context_length=4)
    inputs = torch.randn(1, 4, 4, device=DEVICE, requires_grad=True)
    changed = inputs.detach().clone()
    changed[:, 2:] += 100
    output, weights = attention(inputs, return_attention=True)
    changed_output = attention(changed)
    print("ماسک؛ ۱ یعنی دسترسی مجاز:\n", attention.causal_mask.int())
    print("احتمال‌های توجه:\n", weights[0].detach())
    print("بزرگ‌ترین اختلاف خروجی دو توکن اول پس از تغییر آینده:",
          (output[:, :2] - changed_output[:, :2]).abs().max().item())
    output[:, :2].square().sum().backward()
    print("بزرگ‌ترین گرادیان ورودی آینده از خطای گذشته:", inputs.grad[:, 2:].abs().max().item())
    print("ماسک پیش از softmax اعمال می‌شود تا آینده دقیقاً احتمال صفر بگیرد.")


if __name__ == "__main__":
    main()
