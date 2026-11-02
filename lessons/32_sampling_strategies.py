"""درس ۳۲: اثر دما، Top-K و Top-P روی یک توزیع معلوم."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import SEED
from minicoder.inference import filtered_logits, sample_token


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logits = torch.tensor([2., 1., 0., -1.])
    print("امتیازهای چهار توکن فرضی:", logits)
    print("انتخاب حریصانه:", sample_token(logits))
    for temperature in (0.5, 1.0, 2.0):
        scores = filtered_logits(logits, temperature=temperature)
        print("دما:", temperature, "| احتمال‌ها:", scores.softmax(-1))
    print("احتمال‌های Top-K با K=2:", filtered_logits(logits, top_k=2).softmax(-1))
    print("احتمال‌های Top-P با P=0.8:", filtered_logits(logits, top_p=0.8).softmax(-1))
    first = torch.Generator(device="cpu").manual_seed(SEED)
    second = torch.Generator(device="cpu").manual_seed(SEED)
    sequence = [sample_token(logits, strategy="sample", temperature=1.5, generator=first) for _ in range(12)]
    repeated = [sample_token(logits, strategy="sample", temperature=1.5, generator=second) for _ in range(12)]
    print("انتخاب‌های نمونه‌گیری:", sequence)
    print("تکرار با بذر یکسان:", sequence == repeated)
    print("دما و فیلترها انتخاب را تغییر می‌دهند؛ دانش تازه‌ای به وزن‌های مدل اضافه نمی‌کنند.")


if __name__ == "__main__":
    main()
