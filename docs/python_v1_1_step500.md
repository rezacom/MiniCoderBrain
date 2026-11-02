# آزمایش کنترل‌شدهٔ ۵۰۰گامی پایتون — نسخهٔ ۱٫۱٫۰

این سند آماده‌سازی آزمایش است؛ اجرای ۵۰۰گامی آغاز نشده و نتیجهٔ کیفیت آن موجود نیست. checkpoint پنج‌گامی `checkpoints/python_v1_1_step5.pt` موجود است و به‌عنوان ورودی این آزمایش بارگذاری نمی‌شود. آزمایش جدید از وزن‌های تصادفی با بذر ۴۲ شروع خواهد شد. پیکره، ByteTokenizer و معماری ۸۷۶٬۵۴۸ پارامتری ثابت می‌مانند.

## تنظیمات و مسیر اجرا

فایل مستقل [تنظیمات ۵۰۰گامی](../configs/python_training_v1_1_step500.json) تمام گزینه‌های درخواستی را تعیین می‌کند:

| گزینه | مقدار |
| --- | --- |
| گام بهینه‌ساز | دقیقاً ۵۰۰، شروع از گام صفر |
| اندازهٔ دسته | ۲ |
| طول توالی و stride | ۱۲۸ و ۱۲۸ |
| دادهٔ کوتاه | padding راست؛ هدف PAD در خطا لحاظ نمی‌شود |
| نرخ یادگیری | `0.0003` |
| weight decay | `0.01` برای ماتریس‌ها؛ رفتار موجودِ بایاس و LayerNorm بدون واپاشی |
| AdamW | همان پیاده‌سازی موجود؛ `betas=(0.9, 0.95)` |
| سقف نرم گرادیان | `1.0`، پیش از هر به‌روزرسانی |
| فاصلهٔ اعتبارسنجی | هر ۵۰ گام، از ۵۰ تا ۵۰۰، جمعاً ده بار |
| بذر | ۴۲ برای Python، PyTorch و نمونه‌گیر |
| رشتهٔ CPU | ۴ |
| checkpoint خروجی | `checkpoints/python_v1_1_step500.pt` |

همهٔ گزینه‌ها پشتیبانی می‌شوند. اسکریپت واسط `train_python.py` آن‌ها را از JSON می‌خواند و به پرچم‌های موجود parser در `train.py` تبدیل می‌کند؛ برای مثال `cpu_threads` به `--threads` و `max_grad_norm` به `--grad-clip`. پرچم جدیدی به CLI آموزش اضافه نشده است. تنظیمات پیش‌فرض قدیمی همچنان پنج گام و دو رشته دارند؛ برای این آزمایش باید `--config` زیر را بدهید.

فایل‌های بررسی‌شده: `scripts/train_python.py`، `scripts/train.py`، `minicoder/experiments/python_training.py`، `minicoder/training/trainer.py`، `optimizer.py`، `metrics.py` و `checkpoint.py`؛ همچنین سازندهٔ داده و نمونه‌گیر در `minicoder/data/dataset.py` و مسیر ارزیابی در `scripts/evaluate_python.py` و `minicoder/evaluation/python_benchmarks.py`.

همان `Trainer` موجود استفاده می‌شود. seed پیش از ساخت مدل تنظیم می‌شود؛ اجرای قطعی PyTorch فعال است و Trainer مدل float32 روی CPU را الزام می‌کند. حلقهٔ `fit` تا رسیدن از صفر به `max_steps=500` ادامه دارد؛ افزایش گام پس از یک فراخوانی `optimizer.step()` است و توقف زودهنگام یا انتخاب بهترین checkpoint وجود ندارد. خطای غیرمتناهی باعث شکست اجرا می‌شود و به‌عنوان تکمیل موفق معرفی نمی‌شود.

## جداسازی داده و حفاظت از خروجی

۳۲۰ فایل train و ۴۰ فایل validation با مسیرهای صریح `--files` به pipeline موجود داده می‌شوند. تابع تفکیک ثابت نقش‌ها و هش محتوا را بررسی می‌کند؛ فایل‌ها دوباره تصادفی تفکیک نمی‌شوند. چهل فایل test وارد بهینه‌سازی، اعتبارسنجی، انتخاب checkpoint یا تنظیم ابرپارامتر نمی‌شوند. ابزار آماده‌سازی فقط یک بررسی ایستای صحت manifest و جدایی پیکره انجام می‌دهد؛ این بررسی ارزیابی مدل روی test نیست.

خروجی ثابت این پروفایل checkpoint نهایی گام ۵۰۰ است، بدون انتخاب از روی test یا validation. مسیر پنج‌گامی مستقل و محفوظ است. فایل خروجی موجود پیش از آموزش رد می‌شود. ذخیرهٔ نهایی با فایل موقت هم‌پوشه و پیوند سخت اتمی انجام می‌شود؛ اگر مقصد هنگام اجرا ایجاد شده باشد، جایگزین نمی‌شود. پوشهٔ مقصد باید روی فایل‌سیستم محلی با پشتیبانی از hard link و با فضای کافی باشد.

## فرمان‌ها در PowerShell

ابتدا به ریشهٔ پروژه بروید:

```powershell
Set-Location -LiteralPath 'M:\Apps\minimodel\MiniCoderBrain'
```

پیش‌نمایش امن؛ بدون ساخت مدل یا آغاز آموزش:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_python.py --config configs/python_training_v1_1_step500.json --dry-run
```

آغاز دستی آموزش، فقط زمانی که قصد اجرای آزمایش را دارید:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_python.py --config configs/python_training_v1_1_step500.json
```

این فرمان resume ندارد و checkpoint پنج‌گامی را بارگذاری نمی‌کند. نام checkpoint از JSON خوانده می‌شود. در صورت وجود فایل خروجی، فایل قبلی را حذف یا بازنویسی نکنید؛ برای آزمایشی دیگر مسیر تازه را صریحاً با `--checkpoint` تعیین کنید.

پس از تکمیل موفق آموزش، ارزیابی با همان پروفایل و چهار رشته:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_python.py --config configs/python_training_v1_1_step500.json --checkpoint checkpoints/python_v1_1_step500.pt --trust-checkpoint --output reports/python_v1_1_step500_evaluation.json --markdown reports/python_v1_1_step500_evaluation.md
```

این فرمان `--include-test` ندارد؛ فقط خطا، دقت توکنی و perplexity روی train و validation و ده نمونهٔ تولید از پنج پرامپت ثابت با greedy و نمونه‌گیری بذرگذاری‌شده را گزارش می‌کند. تکرار بایت و قابلیت parse پایتون نیز ثبت می‌شوند؛ کد تولیدی اجرا نمی‌شود. نام‌های گزارش باید تازه باشند. تنظیمات تولید همان نسخهٔ پنج‌گامی است: بودجهٔ ۶۴ توکن، بذر پایهٔ ۴۲، دمای ۰٫۸، Top-K برابر ۴۰ و Top-P برابر ۰٫۹. مقایسه فقط با پروتکل یکسان معتبر است؛ زمان اجرا با دو رشته و چهار رشته مستقیماً هم‌ارز نیست.

## تاریخچهٔ ماشینی موجود

قابلیت ذخیرهٔ تاریخچه از قبل وجود دارد؛ برای افزودن آن هیچ تغییری در موتور، checkpoint یا CLI لازم نیست. `Trainer.history` در فیلد `history` داخل checkpoint کامل ذخیره و هنگام resume بازیابی می‌شود. checkpoint پنج‌گامی موجود واقعاً پنج رکورد دارد و اعتبارسنجی آن در گام پنجم ثبت شده است.

در اجرای موفق این آزمایش، ۵۰۰ رکورد انتظار می‌رود. فیلدهای هر رکورد شامل `step`، `train_loss`، `train_perplexity`، `validation_loss`، `validation_perplexity`، `tokens_processed`، `duration_seconds`، `grad_norm_before` و `grad_norm_after` هستند. مقدار validation در گام‌های خارج از فاصلهٔ ۵۰ برابر `None` است؛ به‌معنای صفر نیست. خطای train مربوط به دستهٔ همان گام پیش از به‌روزرسانی است؛ خطای validation با وزن‌های پس از به‌روزرسانی و به‌صورت وزن‌دار برحسب تعداد هدف معتبر محاسبه می‌شود. خطای کامل train در فرمان ارزیابی جدا اندازه‌گیری می‌شود.

در صورت نیاز، پس از آموزش تاریخچه را بدون تغییر checkpoint به JSON تازه صادر کنید:

```powershell
$historyExport = @'
import json
from pathlib import Path
from minicoder.training.checkpoint import load_checkpoint
saved = load_checkpoint('checkpoints/python_v1_1_step500.pt')
with Path('reports/python_v1_1_step500_history.json').open('x', encoding='utf-8') as output:
    json.dump(saved['history'], output, ensure_ascii=False, indent=2, allow_nan=False)
'@
.\.venv\Scripts\python.exe -X utf8 -c $historyExport
```

## محدودیت‌ها

تاریخچه در حافظه نگه داشته و همراه checkpoint در پایان موفق ذخیره می‌شود؛ فایل JSONL زنده یا checkpoint دوره‌ای وجود ندارد. توقف یا خطای فرایند پیش از ذخیرهٔ نهایی می‌تواند تاریخچهٔ این اجرا را از دست بدهد. ادامه از snapshot موجود در pipeline اصلی پشتیبانی می‌شود، اما فرمان این آزمایش عمداً شروع تازه است.

بذر و الگوریتم قطعی برای بازتولید در همین محیط کمک می‌کنند؛ برابری بیتی بین نسخه‌های متفاوت PyTorch، سیستم‌عامل یا تعداد رشته تضمین نمی‌شود. تعداد گام بیشتر تضمین کاهش خطای validation یا بهبود تولید نیست. پیکره همچنان کوچک است؛ حفظ نمونه، نحو معتبر یا تولید مشابه پایتون شاهد فهم برنامه‌نویسی نیست. نتایج کیفیت و زمان اجرای ۵۰۰ گام فقط پس از اجرای واقعی و ارزیابی قابل گزارش‌اند.

تست‌های جدید پیکربندی، انتقال تمام گزینه‌ها به CLI موجود، بذر و CPU، جدایی نقش‌ها و منع بازنویسی را بررسی می‌کنند. تست اتصال در مرز `fit` متوقف می‌شود؛ تست ماندگاری تاریخچه فقط سه گام روی دادهٔ موقت و مدل آزمایشی کوچک دارد. پیکرهٔ اصلی بازتولید یا ویرایش نمی‌شود.

نتیجهٔ اجرای مرتبط با پایتون `.venv`: **۲۶ تست پاس، صفر شکست، صفر خطا و صفر مورد ردشده**، در ۳۲٫۹۴ ثانیه. این اجرا شامل چهار تست جدید و تست‌های قبلی اتصال پیکره و موتور آموزش است؛ کل مجموعه در این نوبت دوباره اجرا نشده است.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_python_step500.py tests/test_python_experiment.py tests/test_training.py -v --junitxml=reports/python_v1_1_step500_relevant_tests.xml
```

[گزارش ماشینی آماده‌سازی](../reports/python_v1_1_step500_preparation.json) تنظیمات، نتیجهٔ تست و تطبیق هش فایل‌های محافظت‌شده را ثبت می‌کند. گزارش آماده‌سازی شامل نتیجهٔ آموزش یا تولید ۵۰۰گامی نیست.
