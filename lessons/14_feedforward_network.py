"""درس ۱۴: عبور بردارهای توکن از شبکهٔ پیش‌خور و شمارش پارامترها."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import CombinedEmbedding, FeedForward
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode("سلام")], dtype=torch.long, device=DEVICE)
    embeddings = CombinedEmbedding()(input_ids)
    network = FeedForward()
    output = network(embeddings)
    dimension, hidden = network.embedding_dim, network.hidden_dim
    formula_count = dimension * hidden + hidden + hidden * dimension + dimension
    pytorch_count = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)

    print("معماری: خطی ← GELU ← خطی؛ بُعدها:", dimension, "←", hidden, "←", dimension)
    print("شکل ورودی:", tuple(embeddings.shape))
    print("شکل خروجی:", tuple(output.shape))
    print("پنج مؤلفهٔ خروجی اولین توکن:", output[0, 0, :5].detach())
    print(f"شمارش دستی: {dimension}×{hidden} + {hidden} + {hidden}×{dimension} + {dimension} = {formula_count}")
    print("شمارش پارامترها با PyTorch:", pytorch_count)
    print("شمارش از ویژگی ماژول:", network.trainable_parameter_count)
    print("وزن‌ها بین همهٔ نمونه‌ها و جایگاه‌ها مشترک‌اند؛ هر بردار جداگانه تبدیل می‌شود.")


if __name__ == "__main__":
    main()
