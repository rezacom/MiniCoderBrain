"""درس ۲۸: دسته‌های ثابت‌شکل و shuffle تکرارپذیر."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader, write_synthetic_corpus


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with TemporaryDirectory() as directory:
        documents = TextDataset(write_synthetic_corpus(Path(directory) / "synthetic"))
        windows = CausalLanguageModelDataset(documents, 128, stride=128)
        first_loader = create_dataloader(windows, batch_size=2, seed=42)
        second_loader = create_dataloader(windows, batch_size=2, seed=42)
        inputs, targets = next(iter(first_loader))
        repeated_inputs, repeated_targets = next(iter(second_loader))
        print("پیکره فقط دادهٔ ساختگی آموزشی است.")
        print("شکل ورودی و هدف دسته:", tuple(inputs.shape), tuple(targets.shape))
        print("نوع داده:", inputs.dtype, "| دستگاه:", inputs.device)
        print("برابری دسته‌ها با بذر یکسان:", torch.equal(inputs, repeated_inputs) and torch.equal(targets, repeated_targets))
        print("تعداد هدف معتبر:", (targets != windows.ignore_index).sum().item())
        print("برای پنجرهٔ لغزان stride=1 و برای قطعه‌های ورودی جدا stride=128 انتخاب می‌شود.")
        print("توکن مرزی مشترک برای هدف انتهای قطعه و ورودی قطعهٔ بعدی، داخل همان بخش داده است.")


if __name__ == "__main__":
    main()
