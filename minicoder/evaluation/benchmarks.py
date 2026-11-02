"""آزمون‌های محدود و نمایشی؛ متن پایتون صرفاً متن می‌ماند و اجرا نمی‌شود."""

from dataclasses import asdict
import hashlib
import sys
import time

import torch

from minicoder.inference import GenerationConfig
from minicoder.training.metrics import perplexity


GENERATION_PROMPTS = (
    ("english", "ادامهٔ کوتاه انگلیسی", "Hello, the small cat"),
    ("persian", "ادامهٔ کوتاه فارسی", "سلام، امروز"),
    ("python", "ادامهٔ متن پایتون؛ بدون اجرای کد", "def add(a, b):\n    "),
)


def process_memory_usage():
    """RSS و اوج کل فرایند؛ این اعداد حافظهٔ مختص مدل نیستند."""
    unavailable = {"available": False, "rss_bytes": None, "peak_rss_bytes": None,
                   "source": None, "scope": "کل فرایند؛ اوج از آغاز عمر فرایند"}
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    )
                ]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return unavailable
            return {**unavailable, "available": True, "rss_bytes": int(counters.WorkingSetSize),
                    "peak_rss_bytes": int(counters.PeakWorkingSetSize), "source": "Windows working set"}
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
        return {**unavailable, "available": True, "peak_rss_bytes": peak_bytes, "source": "resource.ru_maxrss"}
    except (ImportError, OSError, AttributeError):
        return unavailable


def memorization_probe(engine, text="ABABABAB", *, training_hashes=(), add_bos=False, add_eos=False):
    if not isinstance(text, str) or not text:
        raise ValueError("برای آزمون یادآوری، متن غیرخالی لازم است.")
    ids = engine.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    if len(ids) < 2:
        raise ValueError("آزمون یادآوری دست‌کم دو توکن می‌خواهد.")
    known = hashlib.sha256(text.encode("utf-8")).hexdigest() in set(training_hashes)
    truncated = len(ids) > engine.model.config.max_context_length + 1
    ids = ids[:engine.model.config.max_context_length + 1]
    inputs, targets = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
    previous = engine.model.training
    engine.model.eval()
    try:
        with torch.inference_mode():
            logits, loss = engine.model(inputs, targets)
            valid = targets != engine.model.config.ignore_index
            count = valid.sum().item()
            if not count or not torch.isfinite(loss):
                raise ValueError("هدف معتبر و خطای متناهی برای آزمون یادآوری لازم است.")
            correct = ((logits.argmax(-1) == targets) & valid).sum().item()
    finally:
        engine.model.train(previous)
    return {"task": "memorization", "known_training_document": known,
            "demonstration_only": not known, "scope": "پیش‌بینی با ورودی صحیح قبلی؛ teacher forcing",
            "token_count": count, "truncated": truncated, "loss": loss.item(),
            "perplexity": perplexity(loss.item()), "token_accuracy": correct / count,
            "exact_token_match": correct == count,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "add_bos": add_bos, "add_eos": add_eos,
            "interpretation": "تطبیق روی این توالی کوچک، شاهد فهم زبان یا برنامه‌نویسی نیست."}


def generation_benchmarks(engine, config=None):
    config = config if config is not None else GenerationConfig(max_new_tokens=16, utf8_errors="backslashreplace")
    # گرم‌کردن یک forward بیرون زمان‌سنج؛ ساخت مدل و بارگذاری فایل نیز زمان‌گیری نمی‌شوند.
    previous = engine.model.training
    engine.model.eval()
    try:
        warmup = engine.tokenizer.encode(GENERATION_PROMPTS[0][2], add_bos=True)[-engine.model.config.max_context_length:]
        with torch.inference_mode():
            engine.model(torch.tensor([warmup], dtype=torch.long))
    finally:
        engine.model.train(previous)
    examples = []
    for task, title, prompt in GENERATION_PROMPTS:
        started = time.perf_counter()
        result = engine.generate(prompt, config)
        duration = time.perf_counter() - started
        count = len(result.generated_token_ids)
        examples.append({"task": task, "title": title, "prompt": prompt,
                         "generated_text": result.generated_text, "full_text": result.text,
                         "generated_token_ids": list(result.generated_token_ids),
                         "finish_reason": result.finish_reason, "utf8_valid": result.utf8_valid,
                         "incomplete_bytes_hex": result.incomplete_bytes.hex(),
                         "new_tokens": count, "duration_seconds": duration,
                         "tokens_per_second": count / duration if duration > 0 else None,
                         "demonstration_only": True, "code_executed": False})
    elapsed = sum(example["duration_seconds"] for example in examples)
    tokens = sum(example["new_tokens"] for example in examples)
    return {"configuration": asdict(config), "examples": examples,
            "speed": {"new_tokens": tokens, "duration_seconds": elapsed,
                      "tokens_per_second": tokens / elapsed if elapsed > 0 else None,
                      "warmup_forward_passes": 1,
                      "scope": "تولید انتهابه‌انتها، شامل نمونه‌گیری و رمزگشایی؛ توکن بایتی و EOS در شمارش‌اند"},
            "interpretation": "این نمونه‌ها نمایش خروجی‌اند، نه ارزیابی معتبر توانایی عمومی؛ کد تولیدی اجرا نمی‌شود."}
