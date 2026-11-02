"""ارزیابی checkpoint آزمایش پایتون؛ test فقط با درخواست صریح."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data.python_corpus import explicit_corpus_paths
from minicoder.evaluation import Evaluator
from minicoder.evaluation.python_benchmarks import PYTHON_PROMPTS, evaluate_python_experiment
from minicoder.evaluation.report import read_test_results, render_markdown, write_report
from minicoder.experiments.python_training import DEFAULT_CONFIG, PROJECT_ROOT, load_python_experiment_config
from minicoder.inference import GenerationConfig


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="مقایسهٔ greedy و نمونه‌گیری و ارزیابی دادهٔ جدا، بدون اجرای کد.")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint جدیدِ آزمایش پایتون.")
    parser.add_argument("--trust-checkpoint", action="store_true", help="اعلام اعتماد به checkpoint محلی.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "python_training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-new-tokens", type=int, help="بودجهٔ یکسان هر پرامپت؛ پیش‌فرض ۶۴.")
    parser.add_argument("--include-test", action="store_true", help="ارزیابی نهایی test؛ برای انتخاب تنظیمات استفاده نکنید.")
    parser.add_argument("--test-results", type=Path, help="JUnit واقعی تست‌های نرم‌افزار.")
    parser.add_argument("--output", type=Path, help="فایل جدید گزارش JSON.")
    parser.add_argument("--markdown", type=Path, help="فایل جدید گزارش فارسی.")
    parser.add_argument("--dry-run", action="store_true", help="فقط بررسی پیکره و نمایش برنامه؛ بدون بارگذاری مدل یا تولید.")
    args = parser.parse_args(argv)
    try:
        profile = load_python_experiment_config(args.config)
        evaluation = profile["evaluation"]
        GenerationConfig(max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else evaluation["max_new_tokens"])
        _, quality = explicit_corpus_paths(args.data_root)
        if args.dry_run:
            print(json.dumps({"counts": quality["counts"], "prompts": PYTHON_PROMPTS,
                              "strategies": ["greedy", "sample"], "test_will_be_evaluated": args.include_test,
                              "checkpoint_loaded": False, "generation_performed": False}, ensure_ascii=False, indent=2))
            return 0
        if args.checkpoint is None:
            raise ValueError("پس از آموزش، checkpoint آزمایش را صریحاً بدهید؛ مدل آموزش‌دیده حدس زده نمی‌شود.")
        if any(path is not None and path.exists() for path in (args.output, args.markdown)):
            raise FileExistsError("گزارش موجود بازنویسی نمی‌شود؛ نام تازه بدهید.")
        evaluator = Evaluator.from_checkpoint(args.checkpoint, trusted=args.trust_checkpoint,
                                              seed=evaluation["seed"], cpu_threads=profile["training"]["cpu_threads"])
        report = evaluate_python_experiment(
            evaluator, args.data_root, config_path=args.config, include_test=args.include_test,
            max_new_tokens=args.max_new_tokens,
            test_results=read_test_results(args.test_results) if args.test_results else None,
        )
        if args.output is not None or args.markdown is not None:
            write_report(report, json_path=args.output, markdown_path=args.markdown)
        print(render_markdown(report))
        print("خلاصهٔ روش‌ها:", json.dumps(report["benchmarks"]["generation"]["strategy_summary"], ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
