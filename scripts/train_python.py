"""فراخوانی سازگار train.py با فهرست صریح و تقسیم ثابت پایتون."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.experiments.python_training import DEFAULT_CONFIG, prepare_python_training
from scripts import train as existing_train


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="آزمایش پایتون با معماری ثابت؛ پیش‌فرض پنج گام از وزن تصادفی است.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="پروفایل محلی آزمایش.")
    parser.add_argument("--data-root", type=Path, help="پوشهٔ پیکرهٔ اعتبارسنجی‌شده.")
    parser.add_argument("--steps", type=int, help="تعداد گام؛ پیش‌فرض پنج.")
    parser.add_argument("--checkpoint", type=Path, help="مسیر checkpoint تازه؛ فایل موجود رد می‌شود.")
    parser.add_argument("--dry-run", action="store_true", help="نمایش برنامه بدون ساخت مدل یا شروع آموزش.")
    parser.add_argument("--show-files", action="store_true", help="نمایش تمام آرگومان‌های صریح --files.")
    args = parser.parse_args(argv)
    try:
        plan = prepare_python_training(args.config, data_root=args.data_root, steps=args.steps, checkpoint=args.checkpoint)
        print(json.dumps(plan.summary(include_files=args.show_files), ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        training_args = existing_train.parser().parse_args(list(plan.cli_arguments))
        with TemporaryDirectory(prefix="minicoder-python-run-") as directory:
            return existing_train.run(training_args, Path(directory), document_splitter=plan.split_documents)
    except (OSError, ValueError, TypeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
