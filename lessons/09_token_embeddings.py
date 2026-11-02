"""درس ۹: هر شناسهٔ توکن چگونه به یک بردار قابل یادگیری تبدیل می‌شود؟"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import TokenEmbedding
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    # هر دو متن چهار حرف و هشت بایت UTF-8 دارند؛ نیازی به پدگذاری نیست.
    texts = ["سلام", "درود"]
    input_ids = torch.tensor([tokenizer.encode(text) for text in texts], dtype=torch.long, device=DEVICE)
    embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size)
    output = embedding(input_ids)

    print("متن‌ها:", texts)
    print("شناسه‌های ورودی:\n", input_ids)
    print("شکل ورودی:", tuple(input_ids.shape))
    print("شکل جدول توکن:", tuple(embedding.embedding.weight.shape))
    print("شکل خروجی:", tuple(output.shape))
    print("پنج مؤلفهٔ اول بردار اولین توکن:", output[0, 0, :5].detach())
    print("تعداد پارامترهای قابل یادگیری:", embedding.trainable_parameter_count)
    print("شناسه فقط شمارهٔ سطر جدول است؛ بزرگی عدد شناسه معنای زبانی ندارد.")
    print("این بردارها فعلاً تصادفی‌اند؛ برای یادگیری کاربرد مفید به خطا و گرادیان نیاز دارند.")


if __name__ == "__main__":
    main()
