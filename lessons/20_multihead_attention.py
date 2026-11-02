"""درس ۲۰: توجه مستقل هر سر و تصویر نهاییِ سرهای متصل."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, EMBEDDING_DIM, SEED
from minicoder.model import MultiHeadCausalAttention


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    attention = MultiHeadCausalAttention()
    inputs = torch.randn(2, 4, EMBEDDING_DIM, device=DEVICE)
    output, weights = attention(inputs, return_attention=True)
    print("شکل ورودی و خروجی:", tuple(inputs.shape), tuple(output.shape))
    print("شکل احتمال‌ها:", tuple(weights.shape))
    print("احتمال‌های سر صفر:\n", weights[0, 0].detach())
    print("احتمال‌های سر یک:\n", weights[0, 1].detach())
    print("تعداد پارامترهای چهار تصویر:", attention.trainable_parameter_count)
    output.square().mean().backward()
    print("اندازهٔ گرادیان تصویر خروجی:", attention.output_projection.weight.grad.norm().item())
    print("هر سر الگوی توجه خودش را دارد؛ تصویر خروجی اطلاعات سرها را ترکیب می‌کند.")


if __name__ == "__main__":
    main()
