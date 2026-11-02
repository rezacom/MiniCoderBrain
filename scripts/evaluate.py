"""ارزیابی محدود آفلاین؛ پیش‌فرض مدل آموزش‌ندیده و دادهٔ ساختگی است."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder import __version__
from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader, split_documents, write_synthetic_corpus
from minicoder.data.preprocessing import read_local_utf8
from minicoder.evaluation import Evaluator, compare_reports
from minicoder.evaluation.report import read_test_results, render_markdown, write_report
from minicoder.inference import GenerationConfig


def parser():
    command = argparse.ArgumentParser(description="ارزیابی و گزارش مدل؛ بدون checkpoint، فقط مدل تازه و دادهٔ ساختگی سنجیده می‌شود.")
    command.add_argument("--checkpoint", "--checkpoints", dest="checkpoints", nargs="+", type=Path,
                         help="یک یا چند checkpoint کامل محلی برای ارزیابی یا مقایسه.")
    command.add_argument("--trust-checkpoint", action="store_true", help="اعلام اعتماد به فایل‌های مشخص‌شده.")
    command.add_argument("--training-files", nargs="+", type=Path, help="فایل‌های مرجع آموزش؛ فقط همین مسیرها خوانده می‌شوند.")
    command.add_argument("--validation-files", nargs="+", type=Path, help="فایل‌های مستقل اعتبارسنجی؛ یک سند هم پذیرفته است.")
    command.add_argument("--memorization-file", type=Path, help="متن کوچکِ صریح برای آزمون teacher forcing یادآوری.")
    command.add_argument("--sequence-length", type=int, help="طول مشترک ارزیابی؛ پیش‌فرض حداقل ۳۲ و سقف مدل‌ها.")
    command.add_argument("--stride", type=int, help="فاصلهٔ شروع پنجره؛ پیش‌فرض طول نمونه.")
    command.add_argument("--batch-size", type=int, default=2, help="اندازهٔ دستهٔ ارزیابی؛ پیش‌فرض ۲.")
    command.add_argument("--no-special-tokens", action="store_true", help="BOS و EOS به فایل‌ها افزوده نشوند.")
    command.add_argument("--max-new-tokens", type=int, default=16, help="بودجهٔ هر نمونهٔ نمایشی تولید؛ پیش‌فرض ۱۶.")
    command.add_argument("--strategy", choices=["greedy", "sample"], default="greedy", help="روش انتخاب توکن نمونه‌ها.")
    command.add_argument("--temperature", type=float, default=1.0, help="دمای مثبت در حالت sample.")
    command.add_argument("--top-k", type=int, help="Top-K در حالت sample.")
    command.add_argument("--top-p", type=float, help="Top-P در حالت sample.")
    command.add_argument("--seed", type=int, default=42, help="بذر ارزیابی؛ پیش‌فرض ۴۲.")
    command.add_argument("--threads", type=int, default=2, help="رشته‌های CPU هنگام ارزیابی؛ پیش‌فرض ۲.")
    command.add_argument("--test-results", type=Path, help="JUnit واقعی pytest؛ شمار آزمون از این فایل خوانده می‌شود.")
    command.add_argument("--output", type=Path, help="مسیر جدید گزارش JSON.")
    command.add_argument("--markdown", type=Path, help="مسیر جدید گزارش فارسی Markdown.")
    command.add_argument("--json", action="store_true", help="نمایش گزارش JSON روی stdout.")
    return command


def run(args, directory):
    if args.output is not None and args.output.exists() or args.markdown is not None and args.markdown.exists():
        raise FileExistsError("مسیر گزارش موجود است؛ نام تازه‌ای انتخاب کنید.")
    if args.training_files and not args.validation_files:
        raise ValueError("با فایل‌های آموزش، فایل اعتبارسنجی مستقل را هم صریحاً معرفی کنید.")
    evaluators = ([Evaluator.from_checkpoint(path, trusted=args.trust_checkpoint, seed=args.seed, cpu_threads=args.threads)
                   for path in args.checkpoints] if args.checkpoints else
                  [Evaluator.untrained(seed=args.seed, cpu_threads=args.threads)])
    length = args.sequence_length if args.sequence_length is not None else min(32, *(item.model.config.max_context_length for item in evaluators))
    special = not args.no_special_tokens
    if args.validation_files:
        train_docs = TextDataset(args.training_files, add_bos=special, add_eos=special) if args.training_files else None
        validation_docs = TextDataset(args.validation_files, add_bos=special, add_eos=special)
        label = "فایل‌های محلی صریح؛ صحت مجوز و استقلال محتوایی با معرفی‌کننده است."
    else:
        documents = TextDataset(write_synthetic_corpus(directory), add_bos=special, add_eos=special)
        train_docs, validation_docs = split_documents(documents, seed=args.seed)
        label = "پیکرهٔ ساختگی نمایشی؛ تقسیم اسناد به مرجع و اعتبارسنجی، به‌معنای آموزش قبلی مدل نیست."
    test_results = read_test_results(args.test_results) if args.test_results else None
    text = read_local_utf8(args.memorization_file) if args.memorization_file else "ABABABAB"
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, strategy=args.strategy,
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                  seed=args.seed, utf8_errors="backslashreplace")
    reports = []
    for evaluator in evaluators:
        options = dict(sequence_length=length, stride=args.stride, ignore_index=evaluator.model.config.ignore_index)
        train_loader = create_dataloader(CausalLanguageModelDataset(train_docs, **options), args.batch_size,
                                         shuffle=False, seed=args.seed) if train_docs else None
        val_loader = create_dataloader(CausalLanguageModelDataset(validation_docs, **options), args.batch_size,
                                       shuffle=False, seed=args.seed)
        reports.append(evaluator.run(train_loader=train_loader, validation_loader=val_loader,
                                     generation_config=generation, memorization_text=text, data_label=label,
                                     test_results=test_results))
    report = reports[0] if len(reports) == 1 else {
        "schema_version": 1, "project_version": __version__, "report_type": "comparison",
        "reports": reports, "comparison": compare_reports(reports),
    }
    if args.output is not None or args.markdown is not None:
        write_report(report, json_path=args.output, markdown_path=args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.json else render_markdown(report))
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        with TemporaryDirectory(prefix="minicoder-evaluation-") as directory:
            return run(args, Path(directory) / "synthetic")
    except (ValueError, TypeError, OSError, RuntimeError, UnicodeError, ET.ParseError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
