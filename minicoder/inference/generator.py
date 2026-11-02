"""تولید خودبازگشتی CPU با جریان رویداد و رمزگشایی افزایشی UTF-8."""

import codecs
from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import torch
from torch import nn

from minicoder.config import SEED
from minicoder.data.preprocessing import tokenizer_configuration
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteTokenizer
from minicoder.training.checkpoint import load_checkpoint
from .sampling import sample_token, validate_sampling_parameters


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = SEED
    utf8_errors: str = "strict"

    def __post_init__(self):
        validate_sampling_parameters(self.strategy, self.temperature, self.top_k, self.top_p)
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens باید عدد صحیح نامنفی باشد.")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ValueError("بذر باید عدد صحیح بین صفر و ۲ به توان ۶۳ منهای یک باشد.")
        if self.utf8_errors not in ("strict", "backslashreplace"):
            raise ValueError("سیاست UTF-8 باید strict یا backslashreplace باشد.")


class IncrementalUTF8Decoder:
    """بایت ناقص تا رسیدن ادامه نگه داشته می‌شود؛ خطا هیچ‌گاه بی‌صدا حذف نمی‌شود."""

    def __init__(self, errors="strict"):
        if errors not in ("strict", "backslashreplace"):
            raise ValueError("سیاست UTF-8 نامعتبر است.")
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)
        self._validator = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.had_errors = False
        self.incomplete_bytes = b""
        self.finished = False

    @property
    def pending_bytes(self):
        return self._decoder.getstate()[0]

    def feed(self, token_id):
        if self.finished:
            raise ValueError("رمزگشا پایان یافته است.")
        if type(token_id) is not int or not 0 <= token_id <= 255:
            raise ValueError("رمزگشا فقط شناسهٔ بایت ۰ تا ۲۵۵ می‌پذیرد.")
        data = bytes([token_id])
        if not self.had_errors:
            try:
                self._validator.decode(data, final=False)
            except UnicodeDecodeError:
                self.had_errors = True
        return self._decoder.decode(data, final=False)

    def finish(self):
        if self.finished:
            raise ValueError("رمزگشا قبلاً پایان یافته است.")
        self.incomplete_bytes = self.pending_bytes
        if self.incomplete_bytes:
            self.had_errors = True
        result = self._decoder.decode(b"", final=True)
        self.finished = True
        return result


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    generated_text: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    finish_reason: str
    utf8_valid: bool
    incomplete_bytes: bytes

    @property
    def token_ids(self):
        return self.prompt_token_ids + self.generated_token_ids


@dataclass(frozen=True)
class GenerationEvent:
    token_id: int | None = None
    text: str = ""
    result: GenerationResult | None = None


class GenerationDecodingError(UnicodeError):
    def __init__(self, message, generated_token_ids, partial_text):
        super().__init__(message)
        self.generated_token_ids = tuple(generated_token_ids)
        self.partial_text = partial_text


class TextGenerator:
    def __init__(self, model, tokenizer=None):
        self.tokenizer = tokenizer or ByteTokenizer()
        if not isinstance(model, nn.Module) or not isinstance(getattr(model, "config", None), ModelConfig):
            raise TypeError("مدل باید رابط MiniCoderGPT و ModelConfig معتبر داشته باشد.")
        if model.config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError("اندازهٔ واژگان مدل و توکن‌ساز یکسان نیست.")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("تولید فقط با مدل روی CPU پشتیبانی می‌شود.")
        self.model = model
        self.checkpoint_metadata = None

    def stream(self, prompt, config=None):
        config = GenerationConfig() if config is None else config
        if not isinstance(config, GenerationConfig):
            raise TypeError("تنظیمات باید GenerationConfig باشد.")
        if not isinstance(prompt, str):
            raise TypeError("پرامپت باید رشته باشد.")
        if config.top_k is not None and config.top_k > self.tokenizer.vocab_size:
            raise ValueError("top_k از اندازهٔ واژگان بیشتر است.")
        # BOS صریحِ موتور تولید، آغاز خالی و قرارداد آموزشِ سندها را پوشش می‌دهد.
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        history = list(prompt_ids)
        generated, chunks = [], []
        decoder = IncrementalUTF8Decoder(config.utf8_errors)
        rng = torch.Generator(device="cpu").manual_seed(config.seed)
        reason = "max_new_tokens"
        previous_mode = self.model.training
        self.model.eval()
        try:
            for _ in range(config.max_new_tokens):
                context = history[-self.model.config.max_context_length:]
                # زمینهٔ inference هرگز از روی yield به کد مصرف‌کننده نشت نمی‌کند.
                with torch.inference_mode():
                    inputs = torch.tensor([context], dtype=torch.long, device="cpu")
                    logits, _ = self.model(inputs)
                    if logits.shape != (1, len(context), self.tokenizer.vocab_size):
                        raise ValueError("شکل logits خروجی مدل با قرارداد تولید سازگار نیست.")
                    token_id = sample_token(
                        logits[0, -1], strategy=config.strategy, temperature=config.temperature,
                        top_k=config.top_k, top_p=config.top_p, generator=rng,
                        forbidden_ids=(PAD_ID, BOS_ID, UNK_ID),
                    )
                history.append(token_id)
                generated.append(token_id)
                if token_id == EOS_ID:
                    reason = "eos"
                    break
                try:
                    text = decoder.feed(token_id)
                except UnicodeDecodeError as error:
                    raise GenerationDecodingError(
                        "مدل بایت UTF-8 نامعتبر تولید کرد؛ برای نمایش صریح بایت‌ها backslashreplace را انتخاب کنید.",
                        generated, prompt + "".join(chunks),
                    ) from error
                chunks.append(text)
                yield GenerationEvent(token_id=token_id, text=text)
            try:
                tail = decoder.finish()
            except UnicodeDecodeError as error:
                raise GenerationDecodingError(
                    "دنبالهٔ UTF-8 در پایان تولید ناقص است؛ بایت‌ها را بررسی کنید یا backslashreplace را صریحاً انتخاب کنید.",
                    generated, prompt + "".join(chunks),
                ) from error
            chunks.append(tail)
            continuation = "".join(chunks)
            result = GenerationResult(
                prompt=prompt, text=prompt + continuation, generated_text=continuation,
                prompt_token_ids=tuple(prompt_ids), generated_token_ids=tuple(generated),
                finish_reason=reason, utf8_valid=not decoder.had_errors,
                incomplete_bytes=decoder.incomplete_bytes,
            )
        finally:
            self.model.train(previous_mode)
        yield GenerationEvent(token_id=EOS_ID if reason == "eos" else None, text=tail, result=result)

    def generate(self, prompt, config=None):
        result = None
        for event in self.stream(prompt, config):
            if event.result is not None:
                result = event.result
        return result


def load_generator_checkpoint(path, *, trusted=False):
    """فقط checkpoint کاملِ محلی فاز دهم؛ بهینه‌ساز و RNG برای inference بازیابی نمی‌شوند."""
    if trusted is not True:
        raise ValueError("فقط checkpoint محلیِ مورد اعتماد را با trusted=True معرفی کنید.")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", str(path)) or str(path).startswith("\\\\"):
        raise ValueError("فقط مسیر فایل محلی پذیرفته می‌شود؛ دانلود مدل انجام نمی‌شود.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("فایل checkpoint محلی پیدا نشد.")
    if str(path.resolve()).startswith("\\\\"):
        raise ValueError("checkpoint باید روی فایل‌سیستم محلی باشد.")
    try:
        saved = load_checkpoint(path)
    except (EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("فایل checkpoint قابل خواندن و سازگار نیست؛ بارگذاری ناامن امتحان نمی‌شود.") from error
    if saved["tokenizer_config"] != tokenizer_configuration():
        raise ValueError("قرارداد توکن‌ساز checkpoint با ByteTokenizer پروژه یکسان نیست.")
    try:
        config = ModelConfig(**saved["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("تنظیمات معماری checkpoint معتبر نیست.") from error
    if config.vocab_size != ByteTokenizer().vocab_size:
        raise ValueError("واژگان checkpoint باید با توکن‌ساز ۲۶۰ عضوی سازگار باشد.")
    state = saved["model_state"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) and
                                            value.dtype == torch.float32 and torch.isfinite(value).all()
                                            for value in state.values()):
        raise ValueError("وزن‌های checkpoint باید تنسورهای متناهی float32 باشند.")
    if config.tie_weights:
        embedding = state.get("embeddings.token_embedding.embedding.weight")
        head = state.get("lm_head.weight")
        if embedding is None or head is None or not torch.equal(embedding, head):
            raise ValueError("وزن‌های مشترک سر خروجی و embedding در checkpoint ناسازگارند.")
    with torch.random.fork_rng(devices=[]):
        model = MiniCoderGPT(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("شکل یا نام وزن‌ها با معماری checkpoint سازگار نیست.") from error
    model.eval()
    engine = TextGenerator(model)
    # ارزیابی از همان payload اعتبارسنجی‌شده استفاده می‌کند؛ وزن‌ها دوباره خوانده نمی‌شوند.
    engine.checkpoint_metadata = {
        key: saved[key] for key in ("step", "training_config", "history", "tokens_processed", "duration_seconds")
    }
    engine.checkpoint_metadata["data"] = {
        name: None if saved["data"][name] is None else {
            "fingerprint": saved["data"][name]["fingerprint"],
            "dataset_config": saved["data"][name]["dataset_config"],
        } for name in ("train", "validation")
    }
    engine.checkpoint_metadata["source"] = path.name
    return engine
