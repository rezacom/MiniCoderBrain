"""درس ۲۱: نرمال‌سازی ویژگی‌های هر توکن مستقل از توکن‌های دیگر."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, LAYER_NORM_EPS


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    inputs = torch.tensor([[[1., 2., 4., 8.], [10., 20., 40., 80.]]], device=DEVICE)
    norm = torch.nn.LayerNorm(4, eps=LAYER_NORM_EPS, device=DEVICE)
    output = norm(inputs)
    manual = (inputs - inputs.mean(-1, keepdim=True)) / torch.sqrt(
        inputs.var(-1, unbiased=False, keepdim=True) + norm.eps
    )
    print("ورودی:\n", inputs)
    print("میانگین هر جایگاه:", inputs.mean(-1))
    print("خروجی نرمال‌شده:\n", output.detach())
    print("میانگین خروجی:", output.mean(-1).detach())
    print("واریانس خروجی (تقریباً یک):", output.var(-1, unbiased=False).detach())
    print("برابری با فرمول دستی؟", torch.allclose(output, manual))
    print("ضریب مقیاس قابل یادگیری:", norm.weight.detach())
    print("بایاس قابل یادگیری:", norm.bias.detach())
    print("میانگین و واریانس روی محور ویژگی محاسبه می‌شوند؛ نه روی محور زمان یا دسته.")


if __name__ == "__main__":
    main()
