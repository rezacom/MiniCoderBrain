"""درس ۱۵: آموزش MLP روی دادهٔ ساختگی با به‌روزرسانی دستی پارامترها."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, EMBEDDING_DIM, LEARNING_RATE, SEED
from minicoder.model import FeedForward


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # برای این محاسبات کوچک روی CPU، دو رشتهٔ پردازشی کافی است.
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    inputs = torch.randn(2, 4, EMBEDDING_DIM, device=DEVICE)
    targets = torch.tanh(inputs)
    network = FeedForward()
    initial_weights = network.linear1.weight[0, :5].detach().clone()
    print("داده‌ها در حافظه ساخته شدند؛ هدف ساختگی: tanh(x)")
    print("پنج وزن اولیهٔ لایهٔ اول:", initial_weights)

    for step in range(80):
        network.zero_grad()
        prediction = network(inputs)
        loss = (prediction - targets).square().mean()
        loss.backward()
        if step == 0:
            initial_loss = loss.item()
            print("شکل ورودی و خروجی:", tuple(inputs.shape), tuple(prediction.shape))
            print("خطای اولیه (میانگین مربع فاصله):", initial_loss)
            print("پس‌انتشار مشتق خطا را از خروجی تا هر دو لایه می‌رساند.")
            for name, parameter in network.named_parameters():
                print(f"اندازهٔ گرادیان {name}: {parameter.grad.norm().item():.6f}")
            print("پنج گرادیان وزن لایهٔ اول:", network.linear1.weight.grad[0, :5])

        with torch.no_grad():
            for parameter in network.parameters():
                parameter -= LEARNING_RATE * parameter.grad

    network.zero_grad()
    with torch.no_grad():
        final_loss = (network(inputs) - targets).square().mean().item()
    print("به‌روزرسانی دستی: پارامتر ← پارامتر - نرخ یادگیری × گرادیان")
    print("پنج وزن لایهٔ اول پس از آموزش:", network.linear1.weight[0, :5].detach())
    print("خطا پس از ۸۰ گام:", final_loss)
    print("کاهش خطا:", initial_loss - final_loss)
    print("این نتیجه فقط برازش دادهٔ ساختگی حاضر را نشان می‌دهد و سنجش تعمیم نیست.")


if __name__ == "__main__":
    main()
