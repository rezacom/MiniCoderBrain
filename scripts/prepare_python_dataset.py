"""ساخت یا اعتبارسنجی پیکرهٔ محلی پایتون، بدون اجرای نمونه‌ها."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data.python_corpus import CorpusValidationError, DEFAULT_SEED, generate_python_corpus, validate_python_corpus
from minicoder.experiments.python_training import PROJECT_ROOT


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="پیکرهٔ پایتون ساختگی و آموزشی؛ بدون دانلود، اجرا یا آموزش مدل.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "python_training", help="پوشهٔ صریح پیکره.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="بذر قطعی تولید؛ پیش‌فرض ۱۱۰۱.")
    parser.add_argument("--validate-only", action="store_true", help="فقط بررسی فایل‌های موجود بدون بازنویسی.")
    args = parser.parse_args(argv)
    try:
        report = validate_python_corpus(args.output) if args.validate_only else generate_python_corpus(args.output, seed=args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except CorpusValidationError as error:
        print(json.dumps(error.report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
