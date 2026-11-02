"""درس ۱۹: جابه‌جایی محورهای تنسور برای ساخت چهار سر."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, EMBEDDING_DIM, SEED
from minicoder.model import MultiHeadCausalAttention


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    attention = MultiHeadCausalAttention()
    inputs = torch.arange(2 * 3 * EMBEDDING_DIM, dtype=torch.float32, device=DEVICE).reshape(2, 3, EMBEDDING_DIM)
    heads = attention.split_heads(inputs)
    merged = attention.merge_heads(heads)
    print("شکل اصلی:", tuple(inputs.shape))
    print("پس از reshape:", (2, 3, attention.num_heads, attention.head_dim))
    print("پس از transpose:", tuple(heads.shape))
    print("هشت عدد اول سر صفر:", heads[0, 0, 0, :8])
    print("هشت عدد اول سر یک:", heads[0, 1, 0, :8])
    print("شکل پس از اتصال:", tuple(merged.shape), "| برابری دقیق:", torch.equal(merged, inputs))
    print("در شبکهٔ واقعی، این تفکیک پس از تصویرهای آموختنی Q، K و V انجام می‌شود.")


if __name__ == "__main__":
    main()
