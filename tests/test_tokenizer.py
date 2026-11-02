"""آزمون‌های مستقل از PyTorch برای قرارداد توکن‌ساز بایتی."""

import json

import pytest

from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteTokenizer


@pytest.mark.parametrize("text", [
    "سلام، دنیا!",
    "Hello, World!",
    "def greet(name):\n    return f'سلام {name}'\n",
    "😀 🐍 🚀",
    "",
    "یک\nدو\r\nسه\tچهار",
    "  \t\n    ",
    "Python و پایتون 🐍 123",
    "é e\u0301",  # صورت ترکیبی و صورت جداگانه نباید یکسان‌سازی شوند.
    "A\x00B",
])
def test_exact_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode(text) == list(text.encode("utf-8"))


def test_stable_ids_and_vocab_size():
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == 260
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (256, 257, 258, 259)
    assert tokenizer.encode("A") == [65]
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_special_tokens_are_explicit():
    tokenizer = ByteTokenizer()
    text = "سلام"
    plain = tokenizer.encode(text)
    marked = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert marked == [BOS_ID, *plain, EOS_ID]
    assert tokenizer.decode(marked) == f"<BOS>{text}<EOS>"
    assert tokenizer.decode(marked, skip_special_tokens=True) == text
    assert tokenizer.decode([PAD_ID, UNK_ID]) == "<PAD><UNK>"
    assert tokenizer.encode("<BOS>") == list(b"<BOS>")


def test_serialization_round_trip(tmp_path):
    tokenizer = ByteTokenizer()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteTokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("سلام Python 😀") == tokenizer.encode("سلام Python 😀")
    assert loaded.decode(loaded.encode("کد:\n    x = 1")) == "کد:\n    x = 1"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["special_tokens"] == {"PAD": 256, "BOS": 257, "EOS": 258, "UNK": 259}


def test_incompatible_serialization_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    ByteTokenizer().save(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["special_tokens"]["BOS"] = 999
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        ByteTokenizer.load(path)


@pytest.mark.parametrize("token_ids", [[-1], [260], [65, 999], [True], [1.0]])
def test_invalid_token_ids(token_ids):
    with pytest.raises(ValueError, match="invalid token ID"):
        ByteTokenizer().decode(token_ids)


def test_malformed_utf8_is_rejected():
    with pytest.raises(UnicodeDecodeError):
        ByteTokenizer().decode([0xD8])


def test_invalid_input_types():
    tokenizer = ByteTokenizer()
    with pytest.raises(TypeError):
        tokenizer.encode(b"bytes")
    with pytest.raises(TypeError):
        tokenizer.decode((65,))
