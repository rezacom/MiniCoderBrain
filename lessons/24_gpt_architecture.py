"""درس ۲۴: کنار هم قرار دادن اجزا و شمارش دقیق پارامترهای GPT."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import SEED
from minicoder.model import MiniCoderGPT, ModelConfig


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    config = ModelConfig()
    model = MiniCoderGPT(config)
    print("تنظیمات:", asdict(config))
    print("پارامترهای embedding:", model.embeddings.trainable_parameter_count)
    for index, block in enumerate(model.blocks):
        print(f"پارامترهای بلوک {index + 1}:", block.trainable_parameter_count)
    print("پارامترهای نرمال‌سازی نهایی:", sum(p.numel() for p in model.final_norm.parameters()))
    print("پارامترهای سر خروجی:", sum(p.numel() for p in model.lm_head.parameters()))
    print("مجموع بدون اشتراک وزن:", model.trainable_parameter_count)
    tied = MiniCoderGPT(replace(config, tie_weights=True))
    print("مجموع با اشتراک وزن جدول توکن و سر خروجی:", tied.trainable_parameter_count)
    print("آیا واقعاً یک پارامتر مشترک‌اند؟", tied.lm_head.weight is tied.embeddings.token_embedding.embedding.weight)
    print("داشتن این معماری به‌معنای داشتن دانش زبانی نیست؛ تمام وزن‌ها تازه و تصادفی‌اند.")


if __name__ == "__main__":
    main()
