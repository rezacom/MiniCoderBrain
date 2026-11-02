"""تفکیک سندها پیش از ساخت پنجره؛ نسخه‌های یکسان در یک سمت می‌مانند."""

import math
import random

from minicoder.config import SEED
from .dataset import TextDataset


def split_documents(dataset: TextDataset, validation_fraction=0.25, *, seed=SEED):
    if not isinstance(dataset, TextDataset):
        raise TypeError("ورودی باید TextDataset باشد.")
    if not isinstance(validation_fraction, (int, float)) or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("سهم اعتبارسنجی باید بین صفر و یک باشد.")
    groups = list(dict.fromkeys(document.content_hash for document in dataset))
    if len(groups) < 2:
        raise ValueError("برای جداسازی امن دست‌کم دو سند با محتوای متفاوت لازم است.")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_hashes = set(groups[:count])
    train = [index for index, document in enumerate(dataset) if document.content_hash not in validation_hashes]
    validation = [index for index, document in enumerate(dataset) if document.content_hash in validation_hashes]
    return dataset.subset(train), dataset.subset(validation)
