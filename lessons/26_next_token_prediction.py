"""درس ۲۶: هدفِ یک توکن جلوتر، نادیده گرفتن PAD و پس‌انتشار."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import DEVICE, SEED
from minicoder.model import MiniCoderGPT
from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    # دنباله‌ای دست‌ساز برای آموزش قرارداد هدف؛ مجموعه‌داده‌ای خوانده نمی‌شود.
    sequence = torch.tensor([[BOS_ID, 65, 66, EOS_ID, PAD_ID]], device=DEVICE)
    input_ids = sequence[:, :-1]
    target_ids = sequence[:, 1:]
    model = MiniCoderGPT()
    logits, loss = model(input_ids, target_ids)
    valid = target_ids != model.config.ignore_index
    manual_loss = -logits.log_softmax(-1)[valid].gather(1, target_ids[valid].unsqueeze(1)).mean()
    print("دنبالهٔ اصلی:", sequence)
    print("ورودی (بدون آخرین توکن):", input_ids)
    print("هدف (بدون اولین توکن):", target_ids)
    print("جایگاه‌های دارای هدف معتبر:", valid)
    print("شکل logits:", tuple(logits.shape))
    print("خطای آنتروپی متقاطع:", loss.item())
    print("خطای محاسبه‌شده از log-softmax:", manual_loss.item())
    loss.backward()
    print("گرادیان جدول توکن:", model.embeddings.token_embedding.embedding.weight.grad.norm().item())
    print("گرادیان Query بلوک اول:", model.blocks[0].attention.query.weight.grad.norm().item())
    print("گرادیان سر خروجی:", model.lm_head.weight.grad.norm().item())
    print("هدف‌ها فقط بیرون مدل یک بار جابه‌جا می‌شوند؛ درون مدل شیفت دیگری وجود ندارد.")
    print("این مثال فقط گذر رو به جلو و عقب است و هیچ به‌روزرسانی وزنی انجام نمی‌دهد.")


if __name__ == "__main__":
    main()
