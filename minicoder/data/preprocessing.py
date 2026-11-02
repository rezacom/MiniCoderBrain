"""خواندن فایل صریح، بررسی اولیهٔ محتوای حساس و ساخت نمونهٔ آموزشی محلی."""

from pathlib import Path
import re

from minicoder.tokenizer import ByteTokenizer
from minicoder.tokenizer.vocabulary import SPECIAL_TOKENS


def tokenizer_configuration(tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    return {"type": "ByteTokenizer", "encoding": "utf-8", "vocab_size": tokenizer.vocab_size,
            "special_tokens": dict(SPECIAL_TOKENS)}


def reject_obvious_secrets(text: str) -> None:
    # محتوا یا مقدار حساس هرگز در متن خطا چاپ نمی‌شود. این بررسی جای بازبینی انسان نیست.
    patterns = (
        r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret(?:[_-]?key)?|client[_-]?secret)\s*['\"]?\s*[:=]\s*['\"]?[^\s'\"]+",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"\bAKIA[A-Z0-9]{16}\b",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError("محتوای احتمالاً محرمانه شناسایی شد؛ فایل را پیش از استفاده بازبینی کنید.")


def read_local_utf8(path, *, max_bytes=2 * 1024 * 1024) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError("فقط فایل محلیِ مشخص‌شده پذیرفته می‌شود؛ پوشه یا فایل ناموجود مجاز نیست.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("سقف اندازهٔ فایل باید عدد صحیح مثبت باشد.")
    with path.open("rb") as file:
        content = file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("اندازهٔ فایل از سقف حافظهٔ تعیین‌شده بیشتر است.")
    text = content.decode("utf-8", errors="strict")
    if not text:
        raise ValueError("فایل UTF-8 خالی است.")
    reject_obvious_secrets(text)
    return text


def write_synthetic_corpus(directory):
    """چهار سند ساختگی می‌سازد؛ هیچ فایل موجودی را بازنویسی نمی‌کند."""
    directory = Path(directory)
    samples = {
        "persian.txt": "دادهٔ ساختگی آموزشی؛ نمونه‌ای برای ساخت مدل عمومی نیست.\n" +
                       "سلام دنیا. این یک جملهٔ سادهٔ فارسی است. کتاب روی میز است.\n" * 24,
        "english.txt": "Synthetic educational demonstration, not a general language training corpus.\n" +
                       "Hello world. The small cat sits near the window. Learning uses examples.\n" * 24,
        "python_example.py": "# Synthetic educational demonstration only.\n" +
                             "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n" * 24,
        "mixed.txt": "دادهٔ ساختگی آموزشی / Synthetic demonstration only.\n" +
                     "Python و پایتون: x = 2 + 3\nنتیجه پنج است. The result is five.\n" * 24,
    }
    paths = [directory / name for name in samples]
    if any(path.exists() for path in paths):
        raise FileExistsError("فایل نمونه از قبل موجود است؛ پوشهٔ جدیدی انتخاب کنید.")
    directory.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, samples.values()):
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    return paths
