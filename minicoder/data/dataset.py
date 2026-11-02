"""اسناد محلی و پنجره‌های تنبلِ T+1 توکنی برای آموزش علّی."""

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from minicoder.config import MAX_CONTEXT_LENGTH, SEED
from minicoder.tokenizer import PAD_ID, ByteTokenizer
from .preprocessing import read_local_utf8, tokenizer_configuration


@dataclass(frozen=True)
class TextDocument:
    source: str
    text: str
    token_ids: tuple[int, ...]
    content_hash: str


class TextDataset(Dataset):
    def __init__(self, paths, *, tokenizer=None, add_bos=False, add_eos=False,
                 max_file_bytes=2 * 1024 * 1024, max_total_bytes=8 * 1024 * 1024):
        if isinstance(paths, (str, Path)):
            raise TypeError("فهرست صریح فایل‌ها لازم است؛ یک رشته یا پوشه کافی نیست.")
        self.tokenizer = tokenizer or ByteTokenizer()
        self.add_bos, self.add_eos = add_bos, add_eos
        if type(max_total_bytes) is not int or max_total_bytes <= 0:
            raise ValueError("سقف مجموع بایت‌ها باید مثبت باشد.")
        documents, seen, total = [], set(), 0
        for path in paths:
            resolved = Path(path).resolve()
            if resolved in seen:
                raise ValueError("یک فایل بیش از یک بار در فهرست آمده است.")
            text = read_local_utf8(resolved, max_bytes=max_file_bytes)
            total += len(text.encode("utf-8"))
            if total > max_total_bytes:
                raise ValueError("مجموع فایل‌ها از سقف حافظهٔ تنظیم‌شده بیشتر است.")
            seen.add(resolved)
            tokens = tuple(self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))
            documents.append(TextDocument(str(resolved), text, tokens,
                                          hashlib.sha256(text.encode("utf-8")).hexdigest()))
        if not documents:
            raise ValueError("مجموعهٔ فایل‌های ورودی خالی است.")
        self.documents = tuple(documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def subset(self, indices):
        subset = object.__new__(TextDataset)
        subset.tokenizer = self.tokenizer
        subset.add_bos, subset.add_eos = self.add_bos, self.add_eos
        subset.documents = tuple(self.documents[index] for index in indices)
        if not subset.documents:
            raise ValueError("زیرمجموعهٔ سندها خالی است.")
        return subset


class CausalLanguageModelDataset(Dataset):
    def __init__(self, documents: TextDataset, sequence_length=MAX_CONTEXT_LENGTH, *, stride=None,
                 short_document_policy="pad", pad_token_id=PAD_ID, ignore_index=PAD_ID):
        if not isinstance(documents, TextDataset):
            raise TypeError("ورودی باید TextDataset باشد.")
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("طول دنباله باید عدد صحیح مثبت باشد.")
        stride = sequence_length if stride is None else stride
        if type(stride) is not int or stride <= 0:
            raise ValueError("stride باید عدد صحیح مثبت باشد.")
        if short_document_policy not in ("pad", "drop"):
            raise ValueError("سیاست سند کوتاه باید pad یا drop باشد.")
        if type(pad_token_id) is not int or not 0 <= pad_token_id < documents.tokenizer.vocab_size:
            raise ValueError("شناسهٔ PAD باید در واژگان باشد.")
        if type(ignore_index) is not int:
            raise TypeError("ignore_index باید عدد صحیح باشد.")
        self.documents = documents
        self.sequence_length, self.stride = sequence_length, stride
        self.short_document_policy = short_document_policy
        self.pad_token_id, self.ignore_index = pad_token_id, ignore_index
        self._ends, total = [], 0
        for document in documents:
            available = len(document.token_ids) - (2 if short_document_policy == "pad" else sequence_length + 1)
            total += max(0, available // stride + 1)
            self._ends.append(total)
        if total == 0:
            raise ValueError("هیچ پنجرهٔ دارای هدفی ساخته نشد؛ طول یا سیاست سند کوتاه را بررسی کنید.")

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        document_index = bisect_right(self._ends, index)
        previous = self._ends[document_index - 1] if document_index else 0
        start = (index - previous) * self.stride
        tokens = self.documents[document_index].token_ids[start:start + self.sequence_length + 1]
        inputs, targets = list(tokens[:-1]), list(tokens[1:])
        padding = self.sequence_length - len(inputs)
        inputs.extend([self.pad_token_id] * padding)
        targets.extend([self.ignore_index] * padding)
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    @property
    def configuration(self):
        return {"sequence_length": self.sequence_length, "stride": self.stride,
                "short_document_policy": self.short_document_policy,
                "pad_token_id": self.pad_token_id, "ignore_index": self.ignore_index,
                "add_bos": self.documents.add_bos, "add_eos": self.documents.add_eos,
                "tokenizer": tokenizer_configuration(self.documents.tokenizer),
                "documents": [document.content_hash for document in self.documents]}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode("utf-8")).hexdigest()


class DeterministicBatchSampler(Sampler):
    """نمونه‌گیر بدون پیش‌خوانی؛ epoch و مکان دسته برای ادامهٔ دقیق ذخیره می‌شوند."""

    def __init__(self, size, batch_size, *, shuffle=True, seed=SEED, drop_last=False):
        if type(batch_size) is not int or batch_size <= 0 or size <= 0:
            raise ValueError("اندازهٔ داده و دسته باید مثبت باشند.")
        if type(seed) is not int or seed < 0:
            raise ValueError("بذر باید عدد صحیح نامنفی باشد.")
        self.size, self.batch_size = size, batch_size
        self.shuffle, self.seed, self.drop_last = shuffle, seed, drop_last
        self.epoch, self.offset = 0, 0
        if len(self) == 0:
            raise ValueError("drop_last تمام دسته‌ها را حذف می‌کند.")

    def __len__(self):
        return self.size // self.batch_size if self.drop_last else math.ceil(self.size / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist() if self.shuffle else list(range(self.size))
        for batch_index in range(self.offset, len(self)):
            self.offset = batch_index + 1
            start = batch_index * self.batch_size
            yield indices[start:start + self.batch_size]
        self.epoch += 1
        self.offset = 0

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "shuffle": self.shuffle,
                "seed": self.seed, "drop_last": self.drop_last, "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state):
        for key in ("size", "batch_size", "shuffle", "seed", "drop_last"):
            if state.get(key) != getattr(self, key):
                raise ValueError("تنظیمات نمونه‌گیر با checkpoint یکسان نیست.")
        if type(state.get("epoch")) is not int or state["epoch"] < 0 or not 0 <= state.get("offset", -1) <= len(self):
            raise ValueError("وضعیت نمونه‌گیر نامعتبر است.")
        self.epoch, self.offset = state["epoch"], state["offset"]


def create_dataloader(dataset, batch_size=2, *, shuffle=True, seed=SEED, drop_last=False):
    sampler = DeterministicBatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False,
                      generator=torch.Generator().manual_seed(seed))
