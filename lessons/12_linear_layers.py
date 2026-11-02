"""درس ۱۲: تبدیل خطیِ دارای بایاس با وزن‌های قابل یادگیری."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import ManualLinear


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    layer = ManualLinear(in_features=2, out_features=3)
    # برای دنبال کردن محاسبه با دست، وزن‌ها را به مقدارهای معلوم تغییر می‌دهیم.
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 0., -1.], [2., 1., 1.]], device=DEVICE))
        layer.bias.copy_(torch.tensor([0.5, 0., 1.], device=DEVICE))
    inputs = torch.tensor([[1., 2.], [3., 4.]], device=DEVICE)
    output = layer(inputs)
    print("رابطهٔ لایه: Y = X @ W + b")
    print("ورودی:\n", inputs)
    print("وزن با شکل ورودی × خروجی:\n", layer.weight.detach())
    print("بایاس:", layer.bias.detach())
    print("خروجی:\n", output.detach())
    print("اولین مؤلفه = ۱×۱ + ۲×۲ + ۰٫۵ =", output[0, 0].item())
    print("شکل ورودی و خروجی دوبعدی:", tuple(inputs.shape), tuple(output.shape))
    sequence = inputs.unsqueeze(0)
    print("شکل ورودی و خروجی سه‌بعدی:", tuple(sequence.shape), tuple(layer(sequence).shape))
    print("تعداد پارامترهای قابل یادگیری:", layer.trainable_parameter_count)
    print("وزن‌ها اثر ویژگی‌ها را ترکیب می‌کنند و بایاس خروجی را جابه‌جا می‌کند.")


if __name__ == "__main__":
    main()
