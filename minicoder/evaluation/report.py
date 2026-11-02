"""گزارش خوانای فارسی و JSON استاندارد؛ اعداد آزمون فقط از JUnit خوانده می‌شوند."""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def read_test_results(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("فایل محلی JUnit موجود و با اندازهٔ مجاز لازم است.")
    root = ET.parse(path).getroot()
    suites = [suite for suite in root.iter("testsuite") if not list(suite.iterfind("testsuite"))]
    if not suites:
        raise ValueError("JUnit فاقد testsuite است.")
    counts = {}
    for name in ("tests", "failures", "errors", "skipped"):
        try:
            values = [int(suite.attrib.get(name, "0")) for suite in suites]
        except ValueError as error:
            raise ValueError("شمار آزمون‌ها در JUnit معتبر نیست.") from error
        if any(value < 0 for value in values):
            raise ValueError("شمار آزمون‌ها نباید منفی باشد.")
        counts[name] = sum(values)
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if passed < 0:
        raise ValueError("جمع وضعیت‌های آزمون با تعداد کل سازگار نیست.")
    return {**counts, "passed": passed, "source": path.name}


def _json_block(value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    maximum = max((len(run) for run in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, maximum + 1)
    return f"{fence}json\n{content}\n{fence}"


def _number(value):
    return "موجود نیست" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))


def render_model_information(info):
    training = info["training_record"]
    status = {"untrained": "آموزش‌ندیده", "recorded_training": "دارای گام آموزش ثبت‌شده در checkpoint",
              "unknown": "نامعلوم؛ سابقهٔ آموزش ارائه نشده"}[training["status"]]
    lines = [f"وضعیت مدل: {status}", "", "منبع سوابق آموزش:", "", _json_block(training["source"]),
             "", "| مشخصه | مقدار |", "| --- | ---: |"]
    for label, value in (("پارامترهای یکتا", info["parameters"]), ("پارامترهای قابل یادگیری", info["trainable_parameters"]),
                         ("بلوک‌های ترنسفورمر", info["transformer_layers"]), ("سرهای هر بلوک", info["attention_heads"]),
                         ("اندازهٔ واژگان", info["vocab_size"]), ("سقف زمینه", info["context_length"]),
                         ("گام آموزش ثبت‌شده", training["steps"]),
                         ("آخرین خطای دستهٔ آموزش پیش از به‌روزرسانی", training["last_training_batch_loss"]),
                         ("آخرین خطای اعتبارسنجی ثبت‌شده", training["last_validation_loss"]),
                         ("گام آن اعتبارسنجی", training["last_validation_step"]),
                         ("خطای اعتبارسنجی ثبت‌شده در گام نهایی", training["final_validation_loss"])):
        lines.append(f"| {label} | {_number(value)} |")
    lines += ["", "تنظیمات معماری:", "", _json_block(info["configuration"]), "",
              "تنظیمات آموزش ثبت‌شده یا برنامه‌ریزی‌شده؛ null یعنی ارائه نشده:", "", _json_block(training["configuration"]), "",
              "خطای دستهٔ آموزش، میانگین کل پیکره یا خطای وزن‌ها پس از آخرین به‌روزرسانی نیست."]
    return "\n".join(lines)


def render_markdown(report):
    if report.get("report_type") == "comparison":
        lines = ["# مقایسهٔ checkpointهای محلی", "", report["comparison"]["reason"], "",
                 _json_block(report["comparison"])]
        for index, item in enumerate(report["reports"]):
            lines += ["", f"## مدل {index + 1}", "", render_markdown(item)]
        return "\n".join(lines) + "\n"
    if report.get("report_type") == "model_info":
        return "# مشخصات مدل\n\n" + render_model_information(report["model"]) + "\n"
    lines = ["# گزارش ارزیابی MiniCoder Brain", "", f"نسخهٔ پروژه: {report['project_version']}", "",
             render_model_information(report["model"]), "", "## ارزیابی تازهٔ داده", "",
             report["data_evaluation"]["label"], "",
             "این اندازه‌گیری‌ها تاریخچهٔ آموزش نیستند. برای مدل آموزش‌ندیده، فقط رفتار وزن‌های تصادفی را می‌سنجند.", "",
             _json_block(report["data_evaluation"]), "", "## آزمون محدود یادآوری", "",
             _json_block(report["benchmarks"]["memorization"]), "", "## نمونه‌های نمایشی تولید", "",
             "هیچ کد تولیدی اجرا نشده و هیچ ادعای فهم برنامه‌نویسی از ظاهر خروجی نتیجه‌گیری نمی‌شود."]
    for example in report["benchmarks"]["generation"]["examples"]:
        fields = ["prompt", "generated_text", "utf8_valid", "finish_reason", "new_tokens"]
        fields += [key for key in ("strategy", "sampling_configuration", "repetition", "python_syntax", "prompt_exposure") if key in example]
        lines += ["", f"### {example['title']}", "",
                  _json_block({key: example[key] for key in fields})]
    if "strategy_summary" in report["benchmarks"]["generation"]:
        lines += ["", "### مقایسهٔ روش‌های تولید", "", _json_block(report["benchmarks"]["generation"]["strategy_summary"])]
    lines += ["", "## سرعت و حافظه", "", _json_block({"inference": report["benchmarks"]["generation"]["speed"],
                                                      "memory": report["memory"], "environment": report["environment"]}),
              "", "## آزمون‌های خودکار", ""]
    results = report.get("tests")
    lines += [f"تعداد پاس‌شده: {results['passed']} از {results['tests']}؛ شکست: {results['failures']}؛ خطا: {results['errors']}؛ ردشده: {results['skipped']}."
              if results is not None else "فایل نتیجهٔ آزمون معرفی نشده است؛ تعداد آزمون حدس زده نمی‌شود.",
              "", "## محدودیت‌ها", ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def write_report(report, *, json_path=None, markdown_path=None):
    destinations = [Path(path) for path in (json_path, markdown_path) if path is not None]
    if not destinations:
        raise ValueError("دست‌کم یک مسیر خروجی لازم است.")
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("مسیرهای گزارش باید متفاوت باشند.")
    if any(path.exists() for path in destinations):
        raise FileExistsError("گزارش موجود بازنویسی نمی‌شود؛ مسیر تازه‌ای انتخاب کنید.")
    # همهٔ محتوا پیش از ایجاد فایل آماده و اعتبارسنجی می‌شود.
    contents = []
    if json_path is not None:
        contents.append((Path(json_path), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))
    if markdown_path is not None:
        contents.append((Path(markdown_path), render_markdown(report)))
    for path, content in contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
