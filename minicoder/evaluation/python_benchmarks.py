"""ارزیابی محدود پایتون؛ اندازه‌گیری نحو با ast.parse و بدون اجرای کد."""

import ast
from dataclasses import asdict
import time

import torch

from minicoder import __version__
from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.data.python_corpus import explicit_corpus_paths
from minicoder.evaluation.benchmarks import process_memory_usage
from minicoder.evaluation.evaluator import evaluation_environment
from minicoder.experiments.python_training import DEFAULT_CONFIG, load_python_experiment_config
from minicoder.inference import GenerationConfig


PYTHON_PROMPTS = (
    "def add(a, b):", "def square(x):", "for i in range(", "if x > 0:", "def reverse_string(text):",
)


def repetition_statistics(token_ids):
    byte_ids = [token_id for token_id in token_ids if 0 <= token_id < 256]
    size = len(byte_ids)
    trigrams = [tuple(byte_ids[index:index + 3]) for index in range(max(0, size - 2))]
    longest, run, previous = 0, 0, None
    for token_id in byte_ids:
        run = run + 1 if token_id == previous else 1
        longest, previous = max(longest, run), token_id
    return {"byte_tokens": size, "unique_byte_tokens": len(set(byte_ids)),
            "repeated_token_fraction": (size - len(set(byte_ids))) / size if size else 0.0,
            "repeated_trigram_fraction": (len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0.0,
            "longest_identical_run": longest,
            "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."}


def python_syntax_statistics(text, *, utf8_valid=True):
    if not utf8_valid:
        return {"checked": False, "valid": None, "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."}
    try:
        ast.parse(text, mode="exec")
        return {"checked": True, "valid": True, "reason": "فقط نحو؛ صحت عملکرد یا ایمنی اثبات نشده است."}
    except (SyntaxError, ValueError):
        return {"checked": True, "valid": False,
                "reason": "متن برنامهٔ کاملِ قابل parse نیست؛ توقف کوتاه ممکن است وسط عبارت یا بلوک باشد."}


def assess_python_generations(engine, training_texts, *, max_new_tokens=64, seed=42,
                              temperature=0.8, top_k=40, top_p=0.9):
    training_texts = tuple(training_texts)
    examples = []
    for strategy in ("greedy", "sample"):
        for index, prompt in enumerate(PYTHON_PROMPTS):
            config = GenerationConfig(
                max_new_tokens=max_new_tokens, strategy=strategy, seed=seed + index,
                temperature=temperature if strategy == "sample" else 1.0,
                top_k=top_k if strategy == "sample" else None, top_p=top_p if strategy == "sample" else None,
                utf8_errors="backslashreplace",
            )
            started = time.perf_counter()
            result = engine.generate(prompt, config)
            elapsed = time.perf_counter() - started
            examples.append({
                "task": "python", "title": f"پرسش {index + 1}، روش {strategy}", "prompt": prompt,
                "strategy": strategy, "sampling_configuration": asdict(config),
                "generated_text": result.generated_text, "full_text": result.text,
                "generated_token_ids": list(result.generated_token_ids), "finish_reason": result.finish_reason,
                "utf8_valid": result.utf8_valid, "incomplete_bytes_hex": result.incomplete_bytes.hex(),
                "new_tokens": len(result.generated_token_ids), "duration_seconds": elapsed,
                "tokens_per_second": len(result.generated_token_ids) / elapsed if elapsed > 0 else None,
                "repetition": repetition_statistics(result.generated_token_ids),
                "python_syntax": python_syntax_statistics(result.text, utf8_valid=result.utf8_valid),
                "prompt_exposure": {"configured_training_examples_containing_fragment": sum(prompt in source for source in training_texts),
                                    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."},
                "demonstration_only": True, "code_executed": False,
            })
    aggregates = {}
    for strategy in ("greedy", "sample"):
        rows = [row for row in examples if row["strategy"] == strategy]
        syntax = [row["python_syntax"] for row in rows if row["python_syntax"]["checked"]]
        aggregates[strategy] = {
            "examples": len(rows), "utf8_valid_examples": sum(row["utf8_valid"] for row in rows),
            "syntax_checked": len(syntax), "syntax_valid": sum(row["valid"] for row in syntax),
            "syntax_valid_fraction": sum(row["valid"] for row in syntax) / len(syntax) if syntax else None,
            "mean_repeated_token_fraction": sum(row["repetition"]["repeated_token_fraction"] for row in rows) / len(rows),
            "mean_repeated_trigram_fraction": sum(row["repetition"]["repeated_trigram_fraction"] for row in rows) / len(rows),
        }
    elapsed = sum(row["duration_seconds"] for row in examples)
    count = sum(row["new_tokens"] for row in examples)
    return {"configuration": {"max_new_tokens": max_new_tokens, "seed": seed, "sampling_temperature": temperature,
                              "sampling_top_k": top_k, "sampling_top_p": top_p},
            "examples": examples, "strategy_summary": aggregates,
            "speed": {"new_tokens": count, "duration_seconds": elapsed,
                      "tokens_per_second": count / elapsed if elapsed > 0 else None,
                      "scope": "تولید انتهابه‌انتها؛ زمان معیار کیفیت نیست."},
            "interpretation": "مقایسهٔ نمونه‌ها و آمار تکرار و نحو؛ کیفیت معنایی یا درک برنامه‌نویسی اندازه‌گیری نشده است."}


def verify_python_checkpoint(evaluator, paths, *, profile):
    if any(getattr(evaluator.model.config, key) != value for key, value in profile["expected_model"].items()):
        raise ValueError("این آزمایش باید معماری ثابت ۸۷۶٬۵۴۸ پارامتری را حفظ کند.")
    if evaluator.metadata is None or evaluator.metadata.get("step", 0) <= 0:
        raise ValueError("برای ارزیابی آزمایش، checkpoint دارای گام آموزش واقعی لازم است.")
    train_hashes = {document.content_hash for document in TextDataset(paths["train"])}
    val_hashes = {document.content_hash for document in TextDataset(paths["validation"])}
    for split, hashes in (("train", train_hashes), ("validation", val_hashes)):
        metadata = evaluator.metadata["data"].get(split)
        if metadata is None or set(metadata["dataset_config"]["documents"]) != hashes:
            raise ValueError("checkpoint با تقسیم ثابت پیکرهٔ پایتون آموزش ندیده است.")
        config = metadata["dataset_config"]
        expected = {"sequence_length": 128, "stride": 128, "short_document_policy": "pad",
                    "add_bos": True, "add_eos": True, "ignore_index": evaluator.model.config.ignore_index}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("قرارداد پنجره و کدگذاری checkpoint با آزمایش پایتون یکسان نیست.")


def evaluate_python_experiment(evaluator, root, *, config_path=DEFAULT_CONFIG, include_test=False,
                               max_new_tokens=None, test_results=None):
    profile = load_python_experiment_config(config_path)
    paths, quality = explicit_corpus_paths(root)
    verify_python_checkpoint(evaluator, paths, profile=profile)
    before = process_memory_usage()
    metrics, train_texts = {}, []
    with evaluation_environment(evaluator.seed, evaluator.cpu_threads):
        for split in ("train", "validation", "test"):
            if split == "test" and not include_test:
                metrics[split] = None
                continue
            documents = TextDataset(paths[split], add_bos=True, add_eos=True)
            if split == "train":
                train_texts = [document.text for document in documents]
            dataset = CausalLanguageModelDataset(documents, 128, stride=128,
                                                  ignore_index=evaluator.model.config.ignore_index)
            metrics[split] = evaluator.evaluate_loader(create_dataloader(dataset, batch_size=2, shuffle=False, seed=evaluator.seed))
        parameters = profile["evaluation"]
        generation = assess_python_generations(
            evaluator.engine, train_texts,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else parameters["max_new_tokens"],
            seed=parameters["seed"], temperature=parameters["temperature"],
            top_k=parameters["top_k"], top_p=parameters["top_p"],
        )
    return {
        "schema_version": 1, "report_type": "evaluation", "project_version": __version__,
        "model": evaluator.information(), "python_corpus": quality,
        "data_evaluation": {"label": "پیکرهٔ پایتون ۱٫۱؛ خانواده‌های ساختاری جدا و هدف‌های یک توکن جلوتر.",
                            "training_reference": metrics["train"], "validation": metrics["validation"],
                            "test": metrics["test"], "test_evaluated": include_test,
                            "reference_matches_checkpoint_training": True,
                            "validation_disjoint_from_checkpoint_training": True},
        "benchmarks": {"memorization": {"scope": "خطا و دقت روی train در کنار خانواده‌های دیده‌نشدهٔ validation/test",
                                         "interpretation": "شکاف آموزش و دادهٔ جدا ممکن است نشانهٔ بیش‌برازش باشد؛ مثال تولید شاهد فهم نیست."},
                       "heldout_next_token": metrics["validation"], "generation": generation},
        "memory": {"before": before, "after": process_memory_usage()},
        "environment": {"device": "cpu", "cpu_threads": evaluator.cpu_threads, "seed": evaluator.seed},
        "protocol": {"corpus_sha256": quality["corpus_sha256"], "test_evaluated": include_test,
                     "generation": generation["configuration"], "prompts": list(PYTHON_PROMPTS)},
        "tests": test_results,
        "limitations": ["نحو معتبر به‌معنای صحت برنامه یا فهم برنامه‌نویسی نیست؛ کدی اجرا نمی‌شود.",
                        "اعداد تکرار برحسب بایت‌اند و با طول تولید تغییر می‌کنند.",
                        "validation برای انتخاب تنظیمات است؛ test را فقط پس از انتخاب نهایی ارزیابی کنید.",
                        "گروه‌بندی AST تضمین نبود برنامه‌های هم‌معنای دارای ساختار متفاوت نیست."],
    }
