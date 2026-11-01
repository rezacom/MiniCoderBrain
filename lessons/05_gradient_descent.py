"""درس ۵: آموزش رگرسیون خطی y = ۲x + ۱ با گرادیان‌کاهشی دستی."""

from pathlib import Path
import sys

import torch

# این درس با دستور python lessons/05_gradient_descent.py نیز قابل اجرا است.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, EPOCHS, LEARNING_RATE, NUM_SAMPLES, SEED


def make_data():
    """نمونه‌های ساختگی را در همین رایانه و بدون دانلود تولید می‌کند."""
    x = torch.linspace(-1.0, 1.0, NUM_SAMPLES, device=DEVICE).reshape(-1, 1)
    y = 2.0 * x + 1.0
    return x, y


def make_parameters(seed=SEED):
    """وزن و بایاس را از ابتدا و به‌شکل قابل مشتق‌گیری می‌سازد."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = torch.randn((1, 1), generator=generator, device=DEVICE, requires_grad=True)
    bias = torch.randn((1,), generator=generator, device=DEVICE, requires_grad=True)
    return weight, bias


def predict(x, weight, bias):
    """گذر رو به جلو: برای هر ورودی، y_hat = xw + b."""
    return x @ weight + bias


def mean_squared_error(prediction, target):
    """میانگین مربع فاصلهٔ پیش‌بینی و مقدار واقعی."""
    return ((prediction - target) ** 2).mean()


def update_parameters(parameters, learning_rate):
    """یک گام دستیِ گرادیان‌کاهشی و پاک‌سازی گرادیان‌های قدیمی."""
    with torch.no_grad():
        for parameter in parameters:
            parameter -= learning_rate * parameter.grad
            parameter.grad.zero_()


def train(epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=SEED):
    """مدل خطی را آموزش می‌دهد و وزن، بایاس و تاریخچهٔ خطا را برمی‌گرداند."""
    x, y = make_data()
    weight, bias = make_parameters(seed)
    losses = []

    for _ in range(epochs):
        prediction = predict(x, weight, bias)
        loss = mean_squared_error(prediction, y)
        loss.backward()
        update_parameters((weight, bias), learning_rate)
        losses.append(loss.item())

    return weight.detach(), bias.detach(), losses


def main():
    weight, bias, losses = train()
    print("داده‌ها: y = ۲x + ۱؛ ۱۰۱ نمونهٔ محلی روی CPU")
    print("گذر رو به جلو: x @ weight + bias")
    print("خطا: میانگین مربع فاصله؛ backward مشتق‌ها را محاسبه می‌کند.")
    print("به‌روزرسانی دستی: پارامتر ← پارامتر - نرخ یادگیری × گرادیان")
    print(f"خطای شروع: {losses[0]:.6f} | خطای پایان: {losses[-1]:.6f}")
    print(f"وزن یادگرفته‌شده: {weight.item():.6f} (هدف: ۲)")
    print(f"بایاس یادگرفته‌شده: {bias.item():.6f} (هدف: ۱)")


if __name__ == "__main__":
    main()
