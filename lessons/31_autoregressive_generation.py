"""درس ۳۱: توکن بعدی به ورودی مرحلهٔ بعد تبدیل می‌شود."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.config import SEED
from minicoder.inference import GenerationConfig, IncrementalUTF8Decoder, TextGenerator
from minicoder.model import MiniCoderGPT, ModelConfig


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    model = MiniCoderGPT(ModelConfig(embedding_dim=32, num_heads=4, num_layers=1, hidden_dim=64, max_context_length=16))
    engine = TextGenerator(model)
    config = GenerationConfig(max_new_tokens=8, utf8_errors="backslashreplace")
    print("مدل تازه و تصادفی است؛ این مثال فقط سازوکار تولید را نشان می‌دهد.")
    result = None
    for event in engine.stream("سلام", config):
        if event.token_id is not None:
            print("شناسهٔ تازه:", event.token_id, "| متن کامل‌شده:", repr(event.text))
        if event.result is not None:
            result = event.result
    print("متن نهایی (نمایش صریح):", repr(result.text))
    print("دلیل پایان:", result.finish_reason, "| UTF-8 معتبر؟", result.utf8_valid)
    print("توکن‌های تازه:", result.generated_token_ids)
    print("نمونهٔ رمزگشایی امن یک حرف فارسی:")
    decoder = IncrementalUTF8Decoder()
    for byte in "س".encode("utf-8"):
        print("بایت:", byte, "| خروجی:", repr(decoder.feed(byte)))
    print("باقیمانده در پایان:", repr(decoder.finish()))
    print("هر بایت جداگانه به متن تبدیل نمی‌شود؛ بایت اول منتظر ادامه می‌ماند.")


if __name__ == "__main__":
    main()
