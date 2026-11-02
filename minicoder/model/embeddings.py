"""بردارهای قابل یادگیری برای هویت توکن و موقعیت آن در دنباله."""

import torch
from torch import nn

from minicoder.config import DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LENGTH
from minicoder.tokenizer import ByteTokenizer


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_input_ids(input_ids: torch.Tensor) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must use torch.int32 or torch.int64")
    if input_ids.device.type != DEVICE:
        raise ValueError("input_ids must be on CPU")


class TokenEmbedding(nn.Module):
    """شناسهٔ هر توکن را به یک سطر قابل یادگیری از جدول وزن نگاشت می‌کند."""

    def __init__(self, vocab_size: int | None = None, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        if vocab_size is None:
            vocab_size = ByteTokenizer().vocab_size
        _validate_positive_int("vocab_size", vocab_size)
        _validate_positive_int("embedding_dim", embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=DEVICE)

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _validate_input_ids(input_ids)
        if input_ids.numel() and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size
        ):
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")
        return self.embedding(input_ids)


class PositionEmbedding(nn.Module):
    """برای هر جایگاه از صفر تا سقف زمینه، یک بردار مستقل و قابل یادگیری دارد."""

    def __init__(
        self,
        max_context_length: int = MAX_CONTEXT_LENGTH,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        _validate_positive_int("max_context_length", max_context_length)
        _validate_positive_int("embedding_dim", embedding_dim)
        self.embedding = nn.Embedding(max_context_length, embedding_dim, device=DEVICE)

    @property
    def max_context_length(self) -> int:
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """شکل ورودی را می‌خواند و شناسه‌های ۰ تا طول منهای یک را برای هر نمونه می‌سازد."""
        _validate_input_ids(input_ids)
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_context_length:
            raise ValueError(f"sequence length exceeds max_context_length={self.max_context_length}")
        positions = torch.arange(sequence_length, dtype=torch.long, device=DEVICE)
        return positions.unsqueeze(0).expand(batch_size, -1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.position_ids(input_ids))


class CombinedEmbedding(nn.Module):
    """جمع عنصر به عنصر بردار هویت توکن و بردار موقعیت آن."""

    def __init__(
        self,
        vocab_size: int | None = None,
        embedding_dim: int = EMBEDDING_DIM,
        max_context_length: int = MAX_CONTEXT_LENGTH,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = PositionEmbedding(
            max_context_length=max_context_length, embedding_dim=embedding_dim
        )

    @property
    def vocab_size(self) -> int:
        return self.token_embedding.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self.token_embedding.embedding_dim

    @property
    def max_context_length(self) -> int:
        return self.position_embedding.max_context_length

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids) + self.position_embedding(input_ids)
