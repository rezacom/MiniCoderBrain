"""آموزش کوتاه CPU، اعتبارسنجی وزن‌دار و ادامهٔ دقیق از مکان نمونه‌گیر."""

from dataclasses import asdict, dataclass
import math
import random
import time

import torch

from minicoder.config import SEED
from minicoder.data.dataset import CausalLanguageModelDataset, DeterministicBatchSampler
from minicoder.data.preprocessing import tokenizer_configuration
from .checkpoint import load_checkpoint as read_checkpoint, save_checkpoint as write_checkpoint
from .metrics import perplexity
from .optimizer import create_optimizer


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 5
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_interval: int = 5
    seed: int = SEED
    cpu_threads: int = 2

    def __post_init__(self):
        for name in ("max_steps", "batch_size", "eval_interval", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}: عدد صحیح مثبت لازم است.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("بذر نامنفی لازم است.")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name}: عدد مثبت متناهی لازم است.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("واپاشی وزن نامعتبر است.")


class Trainer:
    def __init__(self, model, train_loader, validation_loader=None, *, config=None):
        self.config = config or TrainingConfig()
        self.model = model
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("موتور فعلی به مدل float32 روی CPU نیاز دارد.")
        torch.set_num_threads(self.config.cpu_threads)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        for loader in (train_loader, validation_loader):
            if loader is None:
                continue
            if not isinstance(loader.dataset, CausalLanguageModelDataset) or not isinstance(loader.batch_sampler, DeterministicBatchSampler):
                raise TypeError("برای ادامهٔ دقیق، DataLoader را با create_dataloader بسازید.")
            if loader.num_workers != 0 or loader.batch_sampler.batch_size != self.config.batch_size:
                raise ValueError("تعداد worker باید صفر و اندازهٔ دسته مطابق تنظیمات باشد.")
            if loader.dataset.sequence_length > model.config.max_context_length:
                raise ValueError("طول نمونه از زمینهٔ مدل بیشتر است.")
            if loader.dataset.ignore_index != model.config.ignore_index:
                raise ValueError("ignore_index داده و مدل باید یکسان باشند.")
            if loader.dataset.documents.tokenizer.vocab_size != model.config.vocab_size:
                raise ValueError("واژگان داده و مدل باید یکسان باشند.")
        if validation_loader is not None:
            train_hashes = {doc.content_hash for doc in train_loader.dataset.documents}
            validation_hashes = {doc.content_hash for doc in validation_loader.dataset.documents}
            if train_hashes & validation_hashes:
                raise ValueError("سند مشترک بین آموزش و اعتبارسنجی مجاز نیست.")
        self.optimizer = create_optimizer(model, learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.step, self.tokens_processed, self.duration_seconds = 0, 0, 0.0
        self.history = []
        self._train_iterator = None
        self._resume_iterator = False

    def _next_batch(self):
        if self._train_iterator is None:
            state = self.train_loader.generator.get_state() if self._resume_iterator else None
            self._train_iterator = iter(self.train_loader)
            if state is not None:
                self.train_loader.generator.set_state(state)
            self._resume_iterator = False
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def train_step(self):
        started = time.perf_counter()
        self.model.train()
        inputs, targets = self._next_batch()
        valid_tokens = (targets != self.model.config.ignore_index).sum().item()
        if valid_tokens == 0:
            raise ValueError("دسته هیچ هدف معتبرِ قابل آموزش ندارد.")
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(inputs, targets)
        if not torch.isfinite(loss):
            raise ValueError("خطای آموزش نامتناهی است؛ گام بهینه‌ساز اجرا نشد.")
        loss.backward()
        norm_before = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True,
        ).item()
        norm_after = torch.stack([parameter.grad.detach().norm() for parameter in self.model.parameters()
                                  if parameter.grad is not None]).norm().item()
        self.optimizer.step()
        self.step += 1
        self.tokens_processed += valid_tokens
        self.duration_seconds += time.perf_counter() - started
        record = {"step": self.step, "train_loss": loss.item(), "train_perplexity": perplexity(loss.item()),
                  "validation_loss": None, "validation_perplexity": None,
                  "tokens_processed": self.tokens_processed, "duration_seconds": self.duration_seconds,
                  "grad_norm_before": norm_before, "grad_norm_after": norm_after}
        self.history.append(record)
        return record

    def evaluate(self, loader=None):
        loader = loader if loader is not None else self.validation_loader
        if loader is None:
            raise ValueError("DataLoader اعتبارسنجی تعیین نشده است.")
        previous_mode = self.model.training
        sampler_state, generator_state = loader.batch_sampler.state_dict(), loader.generator.get_state()
        weighted_loss, tokens = 0.0, 0
        try:
            loader.batch_sampler.epoch, loader.batch_sampler.offset = 0, 0
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in loader:
                    count = (targets != self.model.config.ignore_index).sum().item()
                    if count:
                        _, loss = self.model(inputs, targets)
                        if not torch.isfinite(loss):
                            raise ValueError("خطای اعتبارسنجی متناهی نیست.")
                        weighted_loss += loss.item() * count
                        tokens += count
        finally:
            loader.batch_sampler.load_state_dict(sampler_state)
            loader.generator.set_state(generator_state)
            self.model.train(previous_mode)
        if tokens == 0:
            raise ValueError("اعتبارسنجی هیچ هدف معتبری ندارد.")
        loss = weighted_loss / tokens
        return {"loss": loss, "perplexity": perplexity(loss), "tokens": tokens}

    def fit(self, max_steps=None, *, log_callback=None):
        target_step = self.config.max_steps if max_steps is None else max_steps
        if type(target_step) is not int or target_step <= 0:
            raise ValueError("گام نهایی باید عدد صحیح مثبت باشد.")
        records = []
        while self.step < target_step:
            record = self.train_step()
            if self.validation_loader is not None and (self.step % self.config.eval_interval == 0 or self.step == target_step):
                started = time.perf_counter()
                validation = self.evaluate()
                self.duration_seconds += time.perf_counter() - started
                record.update(validation_loss=validation["loss"], validation_perplexity=validation["perplexity"],
                              duration_seconds=self.duration_seconds)
            if log_callback is not None:
                log_callback(record)
            records.append(record)
        return records

    def _loader_state(self, loader):
        if loader is None:
            return None
        return {"fingerprint": loader.dataset.fingerprint, "dataset_config": loader.dataset.configuration,
                "sampler": loader.batch_sampler.state_dict(), "generator": loader.generator.get_state()}

    def save_checkpoint(self, path):
        payload = {"format_version": 1, "model_config": asdict(self.model.config),
                   "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
                   "training_config": asdict(self.config),
                   "tokenizer_config": tokenizer_configuration(self.train_loader.dataset.documents.tokenizer),
                   "step": self.step, "tokens_processed": self.tokens_processed,
                   "duration_seconds": self.duration_seconds, "history": self.history,
                   "rng": {"torch": torch.get_rng_state(), "python": random.getstate()},
                   "data": {"train": self._loader_state(self.train_loader),
                            "validation": self._loader_state(self.validation_loader),
                            "iterator_active": self._train_iterator is not None}}
        write_checkpoint(path, payload)

    def load_checkpoint(self, path):
        saved = read_checkpoint(path)
        if saved["model_config"] != asdict(self.model.config):
            raise ValueError("معماری مدل با checkpoint یکسان نیست.")
        if saved["tokenizer_config"] != tokenizer_configuration(self.train_loader.dataset.documents.tokenizer):
            raise ValueError("قرارداد توکن‌ساز با checkpoint یکسان نیست.")
        current_training = asdict(self.config)
        if any(value != current_training[key] for key, value in saved["training_config"].items() if key != "max_steps"):
            raise ValueError("تنظیمات آموزش برای ادامه باید یکسان باشند؛ فقط max_steps قابل تغییر است.")
        for name, loader in (("train", self.train_loader), ("validation", self.validation_loader)):
            state = saved["data"][name]
            if (loader is None) != (state is None) or (loader is not None and state["fingerprint"] != loader.dataset.fingerprint):
                raise ValueError("داده یا تنظیمات پنجره‌ها با checkpoint یکسان نیست.")
            if loader is not None:
                current = loader.batch_sampler.state_dict()
                if any(current[key] != state["sampler"][key] for key in ("size", "batch_size", "shuffle", "seed", "drop_last")):
                    raise ValueError("نمونه‌گیر با checkpoint یکسان نیست.")
        self.model.load_state_dict(saved["model_state"])
        self.optimizer.load_state_dict(saved["optimizer_state"])
        self.step, self.tokens_processed = saved["step"], saved["tokens_processed"]
        self.duration_seconds, self.history = saved["duration_seconds"], saved["history"]
        for name, loader in (("train", self.train_loader), ("validation", self.validation_loader)):
            if loader is not None:
                loader.batch_sampler.load_state_dict(saved["data"][name]["sampler"])
                loader.generator.set_state(saved["data"][name]["generator"])
        self._train_iterator = None
        self._resume_iterator = saved["data"]["iterator_active"]
        torch.set_rng_state(saved["rng"]["torch"])
        random.setstate(saved["rng"]["python"])
        return self
