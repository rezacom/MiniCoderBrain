"""تبدیل بی‌اتلاف متن به بایت‌های UTF-8 و شناسه‌های ویژه."""

import json
from pathlib import Path

from .vocabulary import (
    BOS_ID,
    BYTE_VOCAB_SIZE,
    EOS_ID,
    ID_TO_SPECIAL,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)


class ByteTokenizer:
    """توکن‌ساز بدون آموزش؛ شناسهٔ هر بایت با مقدار آن برابر است."""

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """متن را دقیقاً به بایت‌های UTF-8 تبدیل می‌کند؛ علائم ویژه اختیاری‌اند."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        token_ids = list(text.encode("utf-8", errors="strict"))
        if add_bos:
            token_ids.insert(0, BOS_ID)
        if add_eos:
            token_ids.append(EOS_ID)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """بایت‌ها را به متن برمی‌گرداند؛ علائم ویژه را نمایش می‌دهد یا حذف می‌کند."""
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers")

        parts: list[str] = []
        byte_buffer = bytearray()
        for token_id in token_ids:
            if type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
                raise ValueError(f"invalid token ID: {token_id!r}")
            if token_id < BYTE_VOCAB_SIZE:
                byte_buffer.append(token_id)
                continue

            # یک علامت ویژه مرز بخش بایتی است. UTF-8 ناقص خطا می‌دهد.
            parts.append(byte_buffer.decode("utf-8", errors="strict"))
            byte_buffer.clear()
            if not skip_special_tokens:
                parts.append(ID_TO_SPECIAL[token_id])

        parts.append(byte_buffer.decode("utf-8", errors="strict"))
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        """قرارداد شناسه‌ها را به JSON ذخیره می‌کند؛ واژگان آموختنی وجود ندارد."""
        metadata = {
            "format": "minicoder-byte-tokenizer",
            "format_version": 1,
            "encoding": "utf-8",
            "byte_vocab_size": BYTE_VOCAB_SIZE,
            "special_tokens": SPECIAL_TOKENS,
        }
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        """فقط فایل سازگار با همین قرارداد ثابت را می‌پذیرد."""
        with Path(path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        expected = {
            "format": "minicoder-byte-tokenizer",
            "format_version": 1,
            "encoding": "utf-8",
            "byte_vocab_size": BYTE_VOCAB_SIZE,
            "special_tokens": SPECIAL_TOKENS,
        }
        if metadata != expected:
            raise ValueError("incompatible byte-tokenizer metadata")
        return cls()
