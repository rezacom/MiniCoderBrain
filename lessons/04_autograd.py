"""درس ۴: مشتق‌گیری خودکار و انباشته شدن گرادیان."""

import torch


def main():
    x = torch.tensor(3.0, requires_grad=True)
    y = x * x + 2 * x + 1  # y = x² + ۲x + ۱
    print("تابع: y = x² + ۲x + ۱؛ در x=۳ مقدار y برابر است با", y.item())
    print("گرهٔ محاسبهٔ خروجی در گراف:", type(y.grad_fn).__name__)

    y.backward()  # dy/dx = ۲x + ۲؛ در x=۳ برابر ۸ است.
    print("گرادیان پس از backward اول (انتظار: ۸):", x.grad.item())

    # فراخوانی دوم گرادیان را جمع می‌کند؛ برای هر گذر یک گراف تازه می‌سازیم.
    (x * x + 2 * x + 1).backward()
    print("گرادیان انباشته پس از گذر دوم (انتظار: ۱۶):", x.grad.item())

    x.grad.zero_()
    print("گرادیان پس از صفر کردن:", x.grad.item())
    print("گراف محاسباتی مسیر عملیات را نگه می‌دارد تا backward مشتق را از خروجی به ورودی برگرداند.")


if __name__ == "__main__":
    main()
