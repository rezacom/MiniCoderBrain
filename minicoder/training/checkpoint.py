"""checkpoint کامل با انتشار اتمی و ممنوعیت بازنویسی فایل موجود."""

import os
from pathlib import Path
import tempfile

import torch


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint موجود بازنویسی نمی‌شود؛ نام تازه‌ای انتخاب کنید.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        # پیوند سخت، انتشار کامل فایل را بدون جایگزینی مقصدِ موجود انجام می‌دهد.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "model_config", "model_state", "optimizer_state", "training_config",
                "tokenizer_config", "step", "tokens_processed", "duration_seconds", "history", "rng", "data"}
    if not isinstance(payload, dict) or not required.issubset(payload) or payload["format_version"] != 1:
        raise ValueError("قالب checkpoint کاملِ آموزش معتبر نیست.")
    return payload
