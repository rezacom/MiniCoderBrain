"""ارزیابی تکرارپذیر با جداسازی تاریخچهٔ آموزش از اندازه‌گیری فعلی."""

from contextlib import contextmanager
from dataclasses import asdict
import math
import hashlib
import json
import platform
import random

import torch

from minicoder import __version__
from minicoder.config import SEED
from minicoder.data import CausalLanguageModelDataset
from minicoder.data.dataset import DeterministicBatchSampler
from minicoder.inference import GenerationConfig, TextGenerator, load_generator_checkpoint
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import TrainingConfig
from minicoder.training.metrics import perplexity
from .benchmarks import generation_benchmarks, memorization_probe, process_memory_usage


def training_summary(metadata=None, *, declared_untrained=False):
    empty = {"status": "untrained" if declared_untrained else "unknown", "source": None,
             "steps": 0 if declared_untrained else None, "configuration": None,
             "last_training_batch_loss": None, "last_training_step": None,
             "last_training_perplexity": None, "last_validation_perplexity": None,
             "last_validation_loss": None, "last_validation_step": None,
             "final_validation_loss": None, "tokens_processed": None, "active_duration_seconds": None}
    if metadata is None:
        return empty
    step, history = metadata.get("step"), metadata.get("history")
    if type(step) is not int or step < 0 or not isinstance(history, list):
        raise ValueError("تاریخچه و گام آموزش در متادیتا معتبر نیست.")
    try:
        configuration = asdict(TrainingConfig(**metadata["training_config"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("تنظیمات آموزش ثبت‌شده معتبر نیست.") from error
    previous = 0
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("هر سطر تاریخچه باید یک نگاشت معتبر باشد.")
        current = record.get("step")
        if type(current) is not int or not previous < current <= step:
            raise ValueError("ترتیب گام‌های ثبت‌شده معتبر نیست.")
        for name in ("train_loss", "validation_loss"):
            value = record.get(name)
            if (name == "train_loss" and value is None) or (value is not None and
                    (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0)):
                raise ValueError("خطای ثبت‌شده باید عدد نامنفی متناهی باشد.")
        previous = current
    if (step > 0 and (not history or history[-1]["step"] != step)) or (step == 0 and history):
        raise ValueError("تاریخچه با گام checkpoint سازگار نیست؛ نتیجهٔ آموزش حدس زده نمی‌شود.")
    if declared_untrained and step > 0:
        raise ValueError("مدلی با گام آموزش ثبت‌شده نباید آموزش‌ندیده معرفی شود.")
    latest_validation = next((record for record in reversed(history) if record.get("validation_loss") is not None), None)
    return {**empty, "status": "recorded_training" if step else "untrained", "source": metadata.get("source"),
            "steps": step, "configuration": configuration,
            "last_training_batch_loss": history[-1]["train_loss"] if history else None,
            "last_training_perplexity": perplexity(history[-1]["train_loss"]) if history else None,
            "last_validation_perplexity": perplexity(latest_validation["validation_loss"]) if latest_validation else None,
            "last_training_step": history[-1]["step"] if history else None,
            "last_validation_loss": latest_validation["validation_loss"] if latest_validation else None,
            "last_validation_step": latest_validation["step"] if latest_validation else None,
            "final_validation_loss": latest_validation["validation_loss"] if latest_validation and latest_validation["step"] == step else None,
            "tokens_processed": metadata.get("tokens_processed"),
            "active_duration_seconds": metadata.get("duration_seconds")}


def model_information(model, metadata=None, *, declared_untrained=False):
    components = {}
    for name, parameter in model.named_parameters():
        pieces = name.split(".")
        component = ".".join(pieces[:2]) if pieces[0] == "blocks" else pieces[0]
        components[component] = components.get(component, 0) + parameter.numel()
    parameters = list(model.parameters())
    return {"configuration": asdict(model.config), "parameters": sum(parameter.numel() for parameter in parameters),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
            "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
            "component_parameters": components, "transformer_layers": len(model.blocks),
            "attention_heads": model.config.num_heads, "head_dim": model.config.embedding_dim // model.config.num_heads,
            "vocab_size": model.config.vocab_size, "context_length": model.config.max_context_length,
            "device": "cpu", "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "training_record": training_summary(metadata, declared_untrained=declared_untrained)}


@contextmanager
def evaluation_environment(seed, cpu_threads):
    python_state, threads = random.getstate(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.set_num_threads(cpu_threads)
            torch.use_deterministic_algorithms(True)
            yield
    finally:
        random.setstate(python_state)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=previous_warn_only)


class Evaluator:
    def __init__(self, model, *, metadata=None, declared_untrained=False, seed=SEED, cpu_threads=2):
        if not isinstance(model, MiniCoderGPT):
            raise TypeError("ارزیاب از همان MiniCoderGPT پروژه استفاده می‌کند.")
        if type(seed) is not int or not 0 <= seed < 2 ** 63 or type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("بذر نامنفی معتبر و تعداد رشتهٔ مثبت لازم است.")
        self.engine = TextGenerator(model)
        self.model, self.metadata = model, metadata
        self.seed, self.cpu_threads, self.declared_untrained = seed, cpu_threads, declared_untrained
        training_summary(metadata, declared_untrained=declared_untrained)

    @classmethod
    def untrained(cls, config=None, *, seed=SEED, cpu_threads=2):
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("بذر نامعتبر است.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = MiniCoderGPT(config if config is not None else ModelConfig())
        return cls(model, declared_untrained=True, seed=seed, cpu_threads=cpu_threads)

    @classmethod
    def from_checkpoint(cls, path, *, trusted=False, seed=SEED, cpu_threads=2):
        engine = load_generator_checkpoint(path, trusted=trusted)
        return cls(engine.model, metadata=engine.checkpoint_metadata, seed=seed, cpu_threads=cpu_threads)

    def information(self):
        return model_information(self.model, self.metadata, declared_untrained=self.declared_untrained)

    def _training_hashes(self):
        if self.metadata is None or self.metadata.get("step", 0) == 0 or self.metadata.get("data", {}).get("train") is None:
            return None
        return set(self.metadata["data"]["train"]["dataset_config"]["documents"])

    def evaluate_loader(self, loader):
        if not isinstance(loader.dataset, CausalLanguageModelDataset) or not isinstance(loader.batch_sampler, DeterministicBatchSampler):
            raise TypeError("داده و DataLoader را با pipeline پروژه بسازید.")
        data = loader.dataset
        if data.sequence_length > self.model.config.max_context_length or data.ignore_index != self.model.config.ignore_index:
            raise ValueError("طول نمونه یا ignore_index با مدل یکسان نیست.")
        if data.documents.tokenizer.vocab_size != self.model.config.vocab_size:
            raise ValueError("واژگان داده و مدل یکسان نیست.")
        previous_mode = self.model.training
        sampler_state, generator_state = loader.batch_sampler.state_dict(), loader.generator.get_state()
        weighted_loss, tokens, correct, examples, exact, batches = 0.0, 0, 0, 0, 0, 0
        try:
            loader.batch_sampler.epoch, loader.batch_sampler.offset = 0, 0
            self.model.eval()
            with evaluation_environment(self.seed, self.cpu_threads), torch.inference_mode():
                for inputs, targets in loader:
                    valid = targets != self.model.config.ignore_index
                    count = valid.sum().item()
                    if not count:
                        continue
                    logits, loss = self.model(inputs, targets)
                    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                        raise ValueError("خروجی یا خطای ارزیابی متناهی نیست.")
                    matches = logits.argmax(-1) == targets
                    weighted_loss += loss.item() * count
                    tokens += count
                    correct += (matches & valid).sum().item()
                    has_target = valid.any(-1)
                    examples += has_target.sum().item()
                    exact += ((matches | ~valid).all(-1) & has_target).sum().item()
                    batches += 1
        finally:
            loader.batch_sampler.load_state_dict(sampler_state)
            loader.generator.set_state(generator_state)
            self.model.train(previous_mode)
        if not tokens:
            raise ValueError("دادهٔ ارزیابی هیچ هدف معتبری ندارد.")
        loss = weighted_loss / tokens
        return {"loss": loss, "perplexity": perplexity(loss), "token_accuracy": correct / tokens,
                "exact_sequence_accuracy": exact / examples, "tokens": tokens, "examples": examples,
                "batches": batches, "fingerprint": data.fingerprint,
                "loader_configuration": {key: sampler_state[key] for key in ("size", "batch_size", "shuffle", "seed", "drop_last")},
                "scope": "پیش‌بینی توکن بعدی با teacher forcing؛ میانگین وزن‌دار بر هدف‌های غیر PAD"}

    def run(self, *, train_loader=None, validation_loader=None, generation_config=None,
            memorization_text="ABABABAB", data_label="دادهٔ معرفی‌شده؛ منشأ را بررسی کنید", test_results=None):
        if not isinstance(memorization_text, str) or not memorization_text:
            raise ValueError("متن آزمون یادآوری باید رشتهٔ غیرخالی باشد.")
        training_hashes = self._training_hashes()
        train_hashes = set() if train_loader is None else {doc.content_hash for doc in train_loader.dataset.documents}
        val_hashes = set() if validation_loader is None else {doc.content_hash for doc in validation_loader.dataset.documents}
        if train_hashes & val_hashes:
            raise ValueError("سند مشترک در دادهٔ مرجع آموزش و اعتبارسنجی وجود دارد.")
        if training_hashes is not None and training_hashes & val_hashes:
            raise ValueError("اسناد اعتبارسنجی داخل اسناد آموزش checkpoint هستند؛ دادهٔ مستقل معرفی کنید.")
        config = generation_config if generation_config is not None else GenerationConfig(
            max_new_tokens=16, seed=self.seed, utf8_errors="backslashreplace")
        reference_matches = None
        if training_hashes is not None:
            saved_config = self.metadata["data"]["train"]["dataset_config"]
            reference_matches = bool(train_hashes) and train_hashes <= training_hashes
            if reference_matches:
                reference_matches = all(train_loader.dataset.configuration[key] == saved_config[key] for key in (
                    "sequence_length", "stride", "short_document_policy", "pad_token_id", "ignore_index", "add_bos", "add_eos"))
        before_memory = process_memory_usage()
        with evaluation_environment(self.seed, self.cpu_threads):
            train_metrics = self.evaluate_loader(train_loader) if train_loader is not None else None
            val_metrics = self.evaluate_loader(validation_loader) if validation_loader is not None else None
            train_data = self.metadata.get("data", {}).get("train") if self.metadata is not None else None
            token_flags = train_data["dataset_config"] if train_data else {}
            known_probe = memorization_text.encode("utf-8")
            probe_is_training = training_hashes is not None and hashlib.sha256(known_probe).hexdigest() in training_hashes
            probe = memorization_probe(self.engine, memorization_text, training_hashes=training_hashes or (),
                                       add_bos=token_flags.get("add_bos", False) if probe_is_training else False,
                                       add_eos=token_flags.get("add_eos", False) if probe_is_training else False)
            generations = generation_benchmarks(self.engine, config)
        return {"schema_version": 1, "report_type": "evaluation", "project_version": __version__,
                "model": self.information(),
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "platform": platform.system(), "device": "cpu", "cpu_threads": self.cpu_threads},
                "data_evaluation": {"label": data_label, "training_reference": train_metrics, "validation": val_metrics,
                                    "reference_matches_checkpoint_training": reference_matches,
                                    "validation_disjoint_from_checkpoint_training": None if training_hashes is None or not val_hashes else True},
                "benchmarks": {"memorization": probe, "heldout_next_token": val_metrics, "generation": generations},
                "memory": {"before": before_memory, "after": process_memory_usage()},
                "protocol": {"seed": self.seed, "cpu_threads": self.cpu_threads,
                             "generation": asdict(config), "memorization_sha256": probe["text_sha256"],
                             "memorization_bos": probe["add_bos"], "memorization_eos": probe["add_eos"],
                             "generation_context_length": self.model.config.max_context_length,
                             "generation_prompts": [(example["task"], example["prompt"]) for example in generations["examples"]],
                             "training_reference_fingerprint": train_metrics["fingerprint"] if train_metrics else None,
                             "validation_fingerprint": val_metrics["fingerprint"] if val_metrics else None,
                             "training_loader": train_metrics["loader_configuration"] if train_metrics else None,
                             "validation_loader": val_metrics["loader_configuration"] if val_metrics else None},
                "tests": test_results,
                "limitations": ["نمونه‌های تولید شاهد فهم زبان یا برنامه‌نویسی نیستند؛ هیچ کدی اجرا نمی‌شود.",
                                "تست یادآوری روی دادهٔ ناشناختهٔ آموزش فقط نمایش سازوکار است.",
                                "زمان و حافظه اندازه‌گیری محیط جاری‌اند و بین اجراها یکسان نمی‌مانند.",
                                "اسناد متفاوت با بخش‌های مشابه خودکار حذف نمی‌شوند؛ جداسازی بر هش دقیق سند تکیه دارد."]}


def compare_reports(reports):
    if len(reports) < 2:
        raise ValueError("برای مقایسه دست‌کم دو گزارش لازم است.")
    protocol = json.dumps(reports[0]["protocol"], sort_keys=True, allow_nan=False)
    if any(json.dumps(report["protocol"], sort_keys=True, allow_nan=False) != protocol for report in reports[1:]):
        raise ValueError("مقایسه به داده، پنجره، seed، رشته و تنظیمات تولید یکسان نیاز دارد.")
    rows = []
    for index, report in enumerate(reports):
        metrics = report["data_evaluation"]["validation"]
        if metrics is None:
            return {"comparable": False, "ranking": [], "reason": "اعتبارسنجی مشترک در دسترس نیست."}
        rows.append({"index": index, "source": report["model"]["training_record"]["source"],
                     "parameters": report["model"]["parameters"], "validation_loss": metrics["loss"],
                     "perplexity": metrics["perplexity"]})
    return {"comparable": True, "ranking": sorted(rows, key=lambda row: row["validation_loss"]),
            "reason": "رتبه فقط بر خطای همین اسناد است؛ نتیجهٔ توانایی عمومی یا درک برنامه‌نویسی نیست."}
