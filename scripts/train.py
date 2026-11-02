"""فرمان آموزش محلی؛ اجرای پیش‌فرض پنج گام روی دادهٔ ساختگی است."""

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader, split_documents, write_synthetic_corpus
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig, seed_everything
from minicoder.training.checkpoint import load_checkpoint
from minicoder.training.metrics import estimate_training_memory_mib


def parser():
    command = argparse.ArgumentParser(description="آموزش کوتاه CPU با فایل‌های محلی صریح یا پیکرهٔ ساختگی.")
    command.add_argument("--files", nargs="+", type=Path, help="فهرست صریح فایل‌های UTF-8؛ پوشه و پیمایش خودکار نداریم.")
    command.add_argument("--steps", type=int, help="گام نهایی مطلق؛ پیش‌فرض پنج گام تازه است.")
    command.add_argument("--sequence-length", type=int, help="طول ورودی و هدف؛ پیش‌فرض ۳۲.")
    command.add_argument("--stride", type=int, help="فاصلهٔ شروع پنجره‌ها؛ پیش‌فرض برابر طول.")
    command.add_argument("--batch-size", type=int, help="اندازهٔ دسته؛ پیش‌فرض ۲.")
    command.add_argument("--learning-rate", type=float, help="نرخ AdamW؛ پیش‌فرض ۰٫۰۰۰۳.")
    command.add_argument("--weight-decay", type=float, help="واپاشی وزن ماتریس‌ها؛ پیش‌فرض ۰٫۰۱.")
    command.add_argument("--grad-clip", type=float, help="سقف نرم گرادیان؛ پیش‌فرض ۱.")
    command.add_argument("--eval-interval", type=int, help="فاصلهٔ گام‌های اعتبارسنجی؛ پیش‌فرض ۵.")
    command.add_argument("--validation-fraction", type=float, default=0.25, help="سهم گروه‌های سند برای اعتبارسنجی.")
    command.add_argument("--seed", type=int, help="بذر تصادفی؛ پیش‌فرض ۴۲.")
    command.add_argument("--threads", type=int, help="تعداد رشته‌های CPU؛ پیش‌فرض ۲.")
    command.add_argument("--short-documents", choices=["pad", "drop"], help="رفتار سند یا انتهای کوتاه؛ پیش‌فرض pad.")
    command.add_argument("--checkpoint", type=Path, help="مسیر جدید برای ذخیرهٔ checkpoint کامل؛ فایل موجود رد می‌شود.")
    command.add_argument("--resume", type=Path, help="ادامه از checkpoint کامل؛ فایل‌های شخصی را دوباره صریحاً مشخص کنید.")
    for flag, help_text in (("embedding-dim", "بُعد بردار"), ("num-heads", "تعداد سرها"),
                            ("num-layers", "تعداد بلوک‌ها"), ("hidden-dim", "بُعد پیش‌خور"),
                            ("max-context-length", "سقف زمینهٔ مدل")):
        command.add_argument("--" + flag, type=int, help=help_text)
    return command


def _restored_document_split(documents, saved):
    available = defaultdict(deque)
    for index, document in enumerate(documents):
        available[document.content_hash].append(index)
    subsets = []
    for name in ("train", "validation"):
        state = saved["data"][name]
        if state is None:
            raise ValueError("این فرمان به checkpoint دارای بخش اعتبارسنجی نیاز دارد.")
        indices = []
        for content_hash in state["dataset_config"]["documents"]:
            if not available[content_hash]:
                raise ValueError("فایل‌های صریحِ این اجرا با اسناد checkpoint یکسان نیستند.")
            indices.append(available[content_hash].popleft())
        subsets.append(documents.subset(indices))
    if any(available.values()):
        raise ValueError("فایل اضافه‌ای نسبت به checkpoint معرفی شده است.")
    return subsets


def run(args, synthetic_directory, *, document_splitter=None):
    if args.checkpoint is not None and args.checkpoint.exists():
        raise FileExistsError("checkpoint خروجی از قبل موجود است؛ پیش از آموزش نام دیگری انتخاب کنید.")
    saved = load_checkpoint(args.resume) if args.resume is not None else None
    training_values = dict(saved["training_config"]) if saved else asdict(TrainingConfig())
    for argument, field in (("batch_size", "batch_size"), ("learning_rate", "learning_rate"),
                            ("weight_decay", "weight_decay"), ("grad_clip", "max_grad_norm"),
                            ("eval_interval", "eval_interval"), ("seed", "seed"), ("threads", "cpu_threads")):
        value = getattr(args, argument)
        if value is not None:
            training_values[field] = value
    training_values["max_steps"] = args.steps if args.steps is not None else (saved["step"] + 5 if saved else 5)
    training_config = TrainingConfig(**training_values)
    if saved and training_config.max_steps <= saved["step"]:
        raise ValueError("--steps باید از گام checkpoint بیشتر باشد و تعداد کل گام‌ها را تعیین می‌کند.")
    seed_everything(training_config.seed, cpu_threads=training_config.cpu_threads)
    model_values = dict(saved["model_config"]) if saved else asdict(ModelConfig())
    for field in ("embedding_dim", "num_heads", "num_layers", "hidden_dim", "max_context_length"):
        value = getattr(args, field)
        if value is not None:
            model_values[field] = value
    model_config = ModelConfig(**model_values)
    previous_data = saved["data"]["train"]["dataset_config"] if saved else {}
    sequence_length = args.sequence_length if args.sequence_length is not None else previous_data.get("sequence_length", 32)
    stride = args.stride if args.stride is not None else previous_data.get("stride", sequence_length)
    policy = args.short_documents or previous_data.get("short_document_policy", "pad")
    if args.files is None:
        paths = write_synthetic_corpus(synthetic_directory)
        print("ورودی: پیکرهٔ ساختگی آموزشی محلی؛ برای مدل عمومی کافی نیست.")
    else:
        paths = args.files
        print("فقط فایل‌های مشخص‌شده خوانده می‌شوند؛ شبکه و جست‌وجوی پوشه استفاده نمی‌شود.")
    documents = TextDataset(paths, add_bos=previous_data.get("add_bos", True), add_eos=previous_data.get("add_eos", True))
    if saved:
        train_docs, validation_docs = _restored_document_split(documents, saved)
    elif document_splitter is not None:
        train_docs, validation_docs = document_splitter(documents)
    else:
        train_docs, validation_docs = split_documents(documents, args.validation_fraction, seed=training_config.seed)
    options = dict(sequence_length=sequence_length, stride=stride, short_document_policy=policy,
                   ignore_index=model_config.ignore_index)
    train_data, validation_data = CausalLanguageModelDataset(train_docs, **options), CausalLanguageModelDataset(validation_docs, **options)
    train_loader = create_dataloader(train_data, training_config.batch_size, seed=training_config.seed)
    validation_loader = create_dataloader(validation_data, training_config.batch_size, shuffle=False, seed=training_config.seed)
    trainer = Trainer(MiniCoderGPT(model_config), train_loader, validation_loader, config=training_config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print("ادامه از گام:", trainer.step)
    estimate = estimate_training_memory_mib(trainer.model, training_config.batch_size, sequence_length)
    print(f"آموزش CPU تا گام {training_config.max_steps} | اسناد آموزش: {len(train_docs)} | اعتبارسنجی: {len(validation_docs)}")
    print(f"برآورد تقریبی حافظهٔ تنسورها: {estimate:.1f} MiB؛ حافظهٔ کل فرایند بیشتر است.")

    def log(record):
        validation = "—" if record["validation_loss"] is None else f'{record["validation_loss"]:.4f}'
        print(f'گام {record["step"]} | خطای آموزش {record["train_loss"]:.4f} | اعتبارسنجی {validation} | توکن {record["tokens_processed"]}')

    trainer.fit(log_callback=log)
    last = trainer.history[-1]
    print(f'زمان فعال آموزش و اعتبارسنجی: {trainer.duration_seconds:.3f} ثانیه | perplexity آموزش: {last["train_perplexity"]}')
    if last["validation_perplexity"] is not None:
        print("perplexity اعتبارسنجی:", last["validation_perplexity"])
    if args.checkpoint is not None:
        trainer.save_checkpoint(args.checkpoint)
        print("checkpoint کامل ذخیره شد:", args.checkpoint)
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        with TemporaryDirectory(prefix="minicoder-synthetic-") as directory:
            return run(args, Path(directory) / "corpus")
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
