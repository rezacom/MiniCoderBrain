"""درس ۱۳: چرا بین لایه‌های خطی به تابع غیرخطی نیاز داریم؟"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import GELU, ReLU, ManualLinear


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    inputs = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]], device=DEVICE)
    first = ManualLinear(1, 2)
    second = ManualLinear(2, 1)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1., -1.]], device=DEVICE))
        first.bias.zero_()
        second.weight.fill_(1.)
        second.bias.zero_()

    linear_stack = second(first(inputs))
    nonlinear_stack = second(ReLU()(first(inputs)))
    collapsed_weight = first.weight @ second.weight
    collapsed_bias = first.bias @ second.weight + second.bias
    collapsed = inputs @ collapsed_weight + collapsed_bias

    print("ورودی | ReLU | GELU:\n", torch.cat([inputs, ReLU()(inputs), GELU()(inputs)], dim=1))
    print("دو لایهٔ خطی، بدون فعال‌سازی:\n", linear_stack.detach())
    print("برابری با یک لایهٔ خطی واحد؟", torch.allclose(linear_stack, collapsed))
    print("همان وزن‌ها با ReLU بین دو لایه:\n", nonlinear_stack.detach())
    print("بدون فعال‌سازی، خروجی x + (-x) برابر صفر است.")
    print("با ReLU، خروجی max(0,x) + max(0,-x) برابر قدرمطلق x می‌شود.")
    print("این تابع در صفر خم دارد و یک تبدیل خطی واحد نمی‌تواند آن را روی این ورودی‌ها بسازد.")
    print("GELU ورودی را به‌صورت نرم و با تابع توزیع تجمعی نرمال وزن‌دهی می‌کند.")


if __name__ == "__main__":
    main()
