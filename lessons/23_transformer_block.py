"""درس ۲۳: توجه و شبکهٔ پیش‌خور در یک بلوک Pre-LayerNorm."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import CombinedEmbedding, TransformerBlock
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    inputs = torch.tensor([tokenizer.encode("سلام")], dtype=torch.long, device=DEVICE)
    embeddings, block = CombinedEmbedding(), TransformerBlock()
    vectors = embeddings(inputs)
    output = block(vectors)
    print("مسیر: نرمال‌سازی، توجه، جمع باقیمانده، نرمال‌سازی، پیش‌خور، جمع باقیمانده")
    print("شکل ورودی و خروجی:", tuple(vectors.shape), tuple(output.shape))
    print("پارامترهای یک بلوک:", block.trainable_parameter_count)
    loss = output.square().mean()
    loss.backward()
    print("خطای ساختگی:", loss.item())
    print("گرادیان Query:", block.attention.query.weight.grad.norm().item())
    print("گرادیان لایهٔ اول پیش‌خور:", block.feedforward.linear1.weight.grad.norm().item())
    print("گرادیان جدول توکن:", embeddings.token_embedding.embedding.weight.grad.norm().item())
    print("توجه اطلاعات گذشته را جابه‌جا می‌کند؛ پیش‌خور ویژگی‌های هر جایگاه را تبدیل می‌کند.")


if __name__ == "__main__":
    main()
