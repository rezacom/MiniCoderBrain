"""درس ۱۶: امتیاز شباهت، softmax و میانگین وزن‌دار Valueها."""

from pathlib import Path
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    query = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64, device=DEVICE)
    key = query.clone()
    value = torch.tensor([[[10., 0.], [0., 20.], [10., 20.]]], dtype=torch.float64, device=DEVICE)
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    allowed = torch.ones(3, 3, dtype=torch.bool, device=DEVICE).tril()
    masked = scores.masked_fill(~allowed, float("-inf"))
    weights = masked.softmax(dim=-1)
    output = weights @ value
    print("امتیازها؛ سطر = Query و ستون = Key:\n", scores[0])
    print("امتیاز پس از مسدود کردن آینده:\n", masked[0])
    print("وزن‌ها پس از softmax:\n", weights[0])
    print("جمع هر سطر:", weights.sum(-1))
    print("میانگین وزن‌دار Valueها:\n", output[0])
    print("اولین توکن فقط خودش را می‌بیند؛ توکن سوم می‌تواند هر سه Value را ترکیب کند.")


if __name__ == "__main__":
    main()
