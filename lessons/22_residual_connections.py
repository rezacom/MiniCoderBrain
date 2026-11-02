"""درس ۲۲: مسیر مستقیم ورودی در جمع باقیمانده و مشتق آن."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    inputs = torch.tensor([-2., 0., 3.], device=DEVICE, requires_grad=True)
    branch = 0.1 * inputs.square()
    output = inputs + branch
    output.sum().backward()
    print("ورودی:", inputs.detach())
    print("شاخهٔ آموختنیِ فرضی f(x)=0.1x²:", branch.detach())
    print("خروجی y=x+f(x):", output.detach())
    print("گرادیان dy/dx:", inputs.grad)
    print("مشتق مورد انتظار 1+0.2x:", 1 + 0.2 * inputs.detach())
    print("اگر شاخهٔ f صفر باشد، خود ورودی و مسیر گرادیان آن حفظ می‌شوند.")
    print("این مسیر مستقیم به آموزش شبکه‌های عمیق کمک می‌کند، ولی تضمین آموزش موفق نیست.")


if __name__ == "__main__":
    main()
