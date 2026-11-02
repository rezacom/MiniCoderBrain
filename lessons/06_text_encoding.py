"""درس ۶: متن فارسی، انگلیسی و کد پایتون چگونه به شناسه تبدیل می‌شوند؟"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.tokenizer import ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tokenizer = ByteTokenizer()
    examples = {
        "فارسی": "سلام دنیا",
        "انگلیسی": "Hello, World!",
        "کد پایتون": "def greet(name):\n    return f'سلام {name}'\n",
        "متن ترکیبی": "Python و پایتون 🐍",
    }
    print("هر بایت UTF-8 یک توکن با شناسهٔ ۰ تا ۲۵۵ است.")
    print("فاصله، بزرگی حروف و نویسه‌های یونیکد بدون تغییر می‌مانند.\n")
    for label, text in examples.items():
        token_ids = tokenizer.encode(text)
        print(f"{label}: {text!r}")
        print("شناسه‌ها:", token_ids)
        print("تعداد نویسه‌ها:", len(text), "| تعداد توکن‌ها:", len(token_ids))
        print("بازگشت دقیق؟", tokenizer.decode(token_ids) == text, "\n")


if __name__ == "__main__":
    main()
