"""درس ۱۷: سه تصویر قابل یادگیری از یک ورودی مشترک."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, EMBEDDING_DIM, SEED
from minicoder.model import CausalSelfAttention


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    inputs = torch.randn(1, 3, EMBEDDING_DIM, device=DEVICE)
    attention = CausalSelfAttention()
    query, key, value = attention.project_qkv(inputs)
    for name, tensor in (("Query", query), ("Key", key), ("Value", value)):
        print(name, "شکل:", tuple(tensor.shape), "پنج مؤلفه:", tensor[0, 0, :5].detach())
    output = attention(inputs)
    loss = output.square().mean()
    loss.backward()
    print("خطای ساختگی:", loss.item())
    for name, projection in (("Query", attention.query), ("Key", attention.key), ("Value", attention.value)):
        print("اندازهٔ گرادیان وزن", name, ":", projection.weight.grad.norm().item())
    print("تعداد پارامترها:", attention.trainable_parameter_count)
    print("Query پرسش جاری، Key نمایهٔ تطبیق و Value محتوای قابل ترکیب را می‌سازد.")


if __name__ == "__main__":
    main()
