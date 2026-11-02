"""فرمان تولید محلی با checkpoint مورد اعتماد یا مدل تصادفیِ صریح."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.inference import GenerationConfig, GenerationDecodingError, TextGenerator, load_generator_checkpoint
from minicoder.model import MiniCoderGPT


def parser():
    command = argparse.ArgumentParser(description="تولید خودبازگشتی CPU با خروجی امن UTF-8.")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="مسیر checkpoint کاملِ محلی فاز دهم.")
    source.add_argument("--untrained", action="store_true", help="ساخت صریح مدل تصادفی فقط برای مشاهدهٔ سازوکار.")
    command.add_argument("--trust-checkpoint", action="store_true", help="اعلام مورد اعتماد بودن فایل معرفی‌شده.")
    command.add_argument("--prompt", default="", help="متن آغازین؛ پرامپت خالی با BOS شروع می‌شود.")
    command.add_argument("--max-new-tokens", type=int, default=32, help="سقف توکن تازه، شامل EOS؛ پیش‌فرض ۳۲.")
    command.add_argument("--strategy", choices=["greedy", "sample"], default="greedy", help="روش انتخاب توکن.")
    command.add_argument("--temperature", type=float, default=1.0, help="دمای مثبت برای sample.")
    command.add_argument("--top-k", type=int, help="نگه داشتن K گزینهٔ نخست، فقط در sample.")
    command.add_argument("--top-p", type=float, help="جرم احتمال هسته، فقط در sample.")
    command.add_argument("--seed", type=int, default=42, help="بذر مولد اختصاصی نمونه‌گیری.")
    command.add_argument("--utf8-errors", choices=["strict", "backslashreplace"], default="strict",
                         help="خطا روی بایت نامعتبر یا نمایش آشکار آن با بک‌اسلش.")
    command.add_argument("--stream", action="store_true", help="نمایش تدریجیِ فقط نویسه‌های کامل.")
    command.add_argument("--threads", type=int, default=2, help="تعداد رشتهٔ CPU؛ پیش‌فرض ۲.")
    return command


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens, strategy=args.strategy, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, seed=args.seed, utf8_errors=args.utf8_errors,
        )
        if args.threads <= 0:
            raise ValueError("تعداد رشته‌های CPU باید مثبت باشد.")
        torch.set_num_threads(args.threads)
        if args.untrained:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                model = MiniCoderGPT()
            engine = TextGenerator(model)
            print("مدل تصادفی و آموزش‌ندیده است؛ خروجی آن دانش زبانی یا کیفیت کاربردی را نشان نمی‌دهد.", file=sys.stderr)
        else:
            engine = load_generator_checkpoint(args.checkpoint, trusted=args.trust_checkpoint)
        if args.stream:
            print(args.prompt, end="", flush=True)
            result = None
            for event in engine.stream(args.prompt, config):
                print(event.text, end="", flush=True)
                if event.result is not None:
                    result = event.result
            print()
        else:
            result = engine.generate(args.prompt, config)
            print(result.text)
        reason = "EOS" if result.finish_reason == "eos" else "سقف توکن تازه"
        print(f"پایان: {reason} | توکن تازه: {len(result.generated_token_ids)}", file=sys.stderr)
        if not result.utf8_valid:
            print("خروجی خام UTF-8 معتبر نبود؛ بایت‌های نامعتبر آشکارا با \\xNN نمایش داده شدند.", file=sys.stderr)
        if result.incomplete_bytes:
            print("بایت‌های ناقص در پایان:", result.incomplete_bytes.hex(), file=sys.stderr)
        return 0
    except GenerationDecodingError as error:
        print(f"خطای UTF-8: {error}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError, RuntimeError, UnicodeError) as error:
        print(f"خطا: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
