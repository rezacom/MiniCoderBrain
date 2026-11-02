"""داده‌های محلی، تفکیک سند و دسته‌های هدفِ توکن بعدی."""

from .dataset import CausalLanguageModelDataset, TextDataset, create_dataloader
from .preprocessing import write_synthetic_corpus
from .splitting import split_documents

__all__ = ["TextDataset", "CausalLanguageModelDataset", "create_dataloader",
           "split_documents", "write_synthetic_corpus"]
