"""مشخصات معماری پیش‌فرض یا checkpoint محلی مورد اعتماد، بدون آموزش."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder import __version__
from minicoder.evaluation import Evaluator
from minicoder.evaluation.benchmarks import process_memory_usage
from minicoder.evaluation.report import render_model_information, write_report


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="گزارش مشخصات مدل CPU؛ بدون checkpoint، مدل تازه آموزش‌ندیده است.")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint کامل محلی.")
    parser.add_argument("--trust-checkpoint", action="store_true", help="اعلام اعتماد به فایل مشخص‌شده.")
    parser.add_argument("--seed", type=int, default=42, help="بذر ساخت مدل تازه.")
    parser.add_argument("--json", action="store_true", help="خروجی JSON روی stdout.")
    parser.add_argument("--output", type=Path, help="فایل جدید JSON؛ بازنویسی نمی‌شود.")
    args = parser.parse_args(argv)
    try:
        evaluator = (Evaluator.from_checkpoint(args.checkpoint, trusted=args.trust_checkpoint, seed=args.seed)
                     if args.checkpoint is not None else Evaluator.untrained(seed=args.seed))
        report = {"schema_version": 1, "project_version": __version__, "report_type": "model_info",
                  "model": evaluator.information(), "memory": process_memory_usage()}
        if args.output is not None:
            write_report(report, json_path=args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.json
              else render_model_information(report["model"]))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
