"""درس ۲۵: از متن تا logits و احتمال‌های توکن بعدی، بدون آموزش."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import MiniCoderGPT
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    inputs = torch.tensor([tokenizer.encode("سلام"), tokenizer.encode("درود")], dtype=torch.long, device=DEVICE)
    model = MiniCoderGPT().eval()
    with torch.inference_mode():
        logits, loss = model(inputs)
        probability = logits[:, -1].softmax(-1)
    print("شکل شناسه‌ها:", tuple(inputs.shape))
    print("شکل logits:", tuple(logits.shape))
    print("آیا همهٔ logits متناهی‌اند؟", torch.isfinite(logits).all().item())
    print("پنج امتیاز اول در آخرین جایگاه:", logits[0, -1, :5])
    print("جمع احتمال‌های توکن بعدی:", probability.sum(-1))
    print("شناسه‌های دارای بیشترین احتمال:", probability.argmax(-1))
    print("بدون هدف، مقدار خطا:", loss)
    print("این پیش‌بینی‌ها حاصل وزن‌های تصادفی‌اند؛ یک توکن بایتی نیز ممکن است به‌تنهایی UTF-8 معتبری نباشد.")


if __name__ == "__main__":
    main()
