"""GPT کوچک از اجزای قبلی؛ هدف‌ها باید بیرون مدل یک توکن جابه‌جا شوند."""

import torch
from torch import nn
from torch.nn import functional as F

from minicoder.config import DEVICE
from .embeddings import CombinedEmbedding, _validate_input_ids
from .model_config import ModelConfig
from .transformer_block import TransformerBlock


class MiniCoderGPT(nn.Module):
    """همیشه (logits, loss) برمی‌گرداند؛ بدون target_ids مقدار loss برابر None است."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        if config is not None and not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config if config is not None else ModelConfig()
        cfg = self.config
        self.embeddings = CombinedEmbedding(cfg.vocab_size, cfg.embedding_dim, cfg.max_context_length)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.embedding_dim, cfg.num_heads, cfg.hidden_dim,
                cfg.max_context_length, cfg.layer_norm_eps,
            ) for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_eps, device=DEVICE)
        # قرارداد وزن nn.Linear برابر واژگان × بُعد است و اشتراک مستقیم را ممکن می‌کند.
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=True, device=DEVICE)
        if cfg.tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

    @property
    def trainable_parameter_count(self):
        # parameters به‌طور پیش‌فرض پارامتر مشترک را فقط یک بار برمی‌گرداند.
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_targets(self, target_ids, input_ids):
        if not isinstance(target_ids, torch.Tensor):
            raise TypeError("target_ids must be a torch.Tensor")
        if target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must have exactly the same shape as input_ids; shift outside the model")
        if target_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("target_ids must use torch.int32 or torch.int64")
        if target_ids.device.type != DEVICE:
            raise ValueError("target_ids must be on CPU")
        valid = target_ids != self.config.ignore_index
        if ((target_ids[valid] < 0) | (target_ids[valid] >= self.config.vocab_size)).any():
            raise ValueError("target IDs must be in vocabulary range or equal ignore_index")

    def forward(self, input_ids, target_ids=None):
        """X=sequence[:,:-1] و Y=sequence[:,1:]؛ درون مدل هیچ شیفتی انجام نمی‌شود."""
        _validate_input_ids(input_ids)
        batch, length = input_ids.shape
        if batch == 0 or length == 0:
            raise ValueError("batch size and sequence length must be positive")
        if length > self.config.max_context_length:
            raise ValueError(f"sequence length exceeds max_context_length={self.config.max_context_length}")
        if target_ids is not None:
            self._validate_targets(target_ids, input_ids)
        hidden = self.embeddings(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if target_ids is not None:
            if (target_ids != self.config.ignore_index).any():
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size), target_ids.long().reshape(-1),
                    ignore_index=self.config.ignore_index,
                )
            else:
                # میانگین روی مجموعهٔ خالی NaN می‌دهد؛ قرارداد ما خطای صفرِ قابل مشتق است.
                loss = logits.sum() * 0.0
        return logits, loss
