"""درس ۱۰: افزودن اطلاعات جایگاه با بردارهای مطلق قابل یادگیری."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import PositionEmbedding
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode("AAAA"), tokenizer.encode("BBBB")],
                             dtype=torch.long, device=DEVICE)
    embedding = PositionEmbedding()
    output = embedding(input_ids)

    print("شناسه‌های موقعیت برای هر نمونه:\n", embedding.position_ids(input_ids))
    print("شکل جدول موقعیت:", tuple(embedding.embedding.weight.shape))
    print("شکل خروجی:", tuple(output.shape))
    print("پنج مؤلفهٔ موقعیت صفر:", output[0, 0, :5].detach())
    print("پنج مؤلفهٔ موقعیت یک:", output[0, 1, :5].detach())
    print("بردار موقعیت صفر در دو نمونه یکسان است؟", torch.equal(output[0, 0], output[1, 0]))
    print("تعداد پارامترهای قابل یادگیری:", embedding.trainable_parameter_count)
    print("جدول موقعیت به جایگاه توکن وابسته است و محتوای شناسه را نمی‌خواند.")
    print("شماره‌گذاری در هر دنباله از صفر شروع می‌شود و طول مجاز بر حسب توکن است.")


if __name__ == "__main__":
    main()
