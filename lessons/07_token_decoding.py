"""درس ۷: ساخت دوبارهٔ متن از بایت‌های UTF-8."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tokenizer = ByteTokenizer()
    for text in ["A", "س", "😀", "a\n    b", ""]:
        token_ids = tokenizer.encode(text)
        restored = tokenizer.decode(token_ids)
        print("متن:", repr(text))
        print("نقطه‌های کد یونیکد:", [f"U+{ord(char):04X}" for char in text])
        print("بایت‌ها = شناسه‌ها:", token_ids)
        print("متن بازسازی‌شده:", repr(restored), "| برابر؟", restored == text, "\n")

    print("یک نویسهٔ فارسی یا ایموجی ممکن است چند بایت و بنابراین چند توکن داشته باشد.")
    print("decode بایت‌های کامل را با UTF-8 به نویسه تبدیل می‌کند.")


if __name__ == "__main__":
    main()
