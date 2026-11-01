"""درس ۳: وزن‌های قابل یادگیری و شکل آن‌ها."""

import torch


def main():
    torch.manual_seed(42)
    # هر ستونِ وزن به یک خروجی مربوط است؛ اینجا ۳ ورودی و ۲ خروجی داریم.
    weight = torch.nn.Parameter(torch.randn(3, 2) * 0.1)
    bias = torch.nn.Parameter(torch.zeros(2))
    inputs = torch.tensor([[1.0, 2.0, 3.0]])  # یک نمونه با سه ویژگی
    outputs = inputs @ weight + bias

    print("وزن اولیهٔ تصادفی:\n", weight)
    print("شکل وزن:", weight.shape, "| شکل بایاس:", bias.shape)
    print("آیا وزن قابل یادگیری است؟", weight.requires_grad)
    print("آیا بایاس قابل یادگیری است؟", bias.requires_grad)
    print("شکل خروجی:", outputs.shape, "| خروجی:", outputs)
    print("هر وزن شدت و جهت اثر یک ورودی بر یک خروجی را تعیین می‌کند.")
    print("در زمان آموزش، گرادیان‌ها نشان می‌دهند وزن‌ها چگونه تغییر کنند.")


if __name__ == "__main__":
    main()
