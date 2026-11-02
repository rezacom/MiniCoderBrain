"""درس ۱۱: ترکیب بردارها و مشاهدهٔ یک گام یادگیری با خطای ساختگی."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, LEARNING_RATE, SEED
from minicoder.model import CombinedEmbedding
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode("AB")], dtype=torch.long, device=DEVICE)
    embedding = CombinedEmbedding(vocab_size=tokenizer.vocab_size)
    token_id = input_ids[0, 0].item()
    token_weights = embedding.token_embedding.embedding.weight
    position_weights = embedding.position_embedding.embedding.weight

    print("پنج مؤلفهٔ اولیهٔ بردار توکن A:", token_weights[token_id, :5].detach().clone())
    print("پنج مؤلفهٔ اولیهٔ بردار موقعیت صفر:", position_weights[0, :5].detach().clone())

    # خطای ساختگی: جمع بردارها را به صفر نزدیک می‌کنیم تا مسیر گرادیان را ببینیم.
    output = embedding(input_ids)
    loss = output.square().mean()
    print("شکل خروجی ترکیبی:", tuple(output.shape))
    print("پنج مؤلفهٔ خروجی اولین توکن:", output[0, 0, :5].detach())
    print("خطای اولیه:", loss.item())
    loss.backward()
    print("گرادیان بردار توکن A:", token_weights.grad[token_id, :5])
    print("گرادیان بردار موقعیت صفر:", position_weights.grad[0, :5])

    with torch.no_grad():
        for parameter in embedding.parameters():
            parameter -= LEARNING_RATE * parameter.grad
    embedding.zero_grad()

    print("بردار توکن A پس از تغییر:", token_weights[token_id, :5].detach())
    print("بردار موقعیت صفر پس از تغییر:", position_weights[0, :5].detach())
    with torch.no_grad():
        updated_loss = embedding(input_ids).square().mean().item()
    print("خطا پس از یک گام:", updated_loss)
    print("پارامترهای جدول توکن:", embedding.token_embedding.trainable_parameter_count)
    print("پارامترهای جدول موقعیت:", embedding.position_embedding.trainable_parameter_count)
    print("مجموع پارامترهای قابل یادگیری:", embedding.trainable_parameter_count)
    print("گرادیان مسیر تغییر وزن‌ها برای کاهش خطا را مشخص می‌کند.")
    print("این هدف ساختگی فقط سازوکار یادگیری را نشان می‌دهد و معنای واژه‌ها را آموزش نمی‌دهد.")


if __name__ == "__main__":
    main()
