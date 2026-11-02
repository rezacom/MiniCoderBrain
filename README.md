# MiniCoderBrain — فازهای ۰۱ تا ۰۳

نسخهٔ `0.3.0` با مثال‌های کوچک و قابل اجرا، مبانی PyTorch، توکن‌سازی بایتی UTF-8 و بردارهای قابل یادگیری توکن و موقعیت را آموزش می‌دهد. همهٔ محاسبات روی CPU انجام می‌شوند. هنوز توجه یا ترنسفورمر پیاده‌سازی نشده است. هیچ مدل آماده، دادهٔ دانلودی یا سرویس ابری استفاده نمی‌شود.

## ساختار

```text
MiniCoderBrain/
├── minicoder/
│   ├── __init__.py
│   ├── config.py                 # تنظیمات ثابت و بذر تصادفی
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── byte_tokenizer.py     # تبدیل متن به شناسه و بازسازی آن
│   │   └── vocabulary.py         # قرارداد ثابت شناسه‌ها
│   └── model/
│       ├── __init__.py
│       └── embeddings.py         # جدول توکن، موقعیت و جمع آن‌ها
├── lessons/
│   ├── 01_tensors.py             # تنسور، شکل، نوع و تغییر شکل
│   ├── 02_matrix_operations.py   # ضرب ماتریسی و پخش خودکار
│   ├── 03_parameters.py          # وزن‌های قابل یادگیری
│   ├── 04_autograd.py            # مشتق‌گیری خودکار
│   ├── 05_gradient_descent.py    # رگرسیون خطی با به‌روزرسانی دستی
│   ├── 06_text_encoding.py       # تبدیل متن به شناسه
│   ├── 07_token_decoding.py      # بازسازی دقیق متن
│   ├── 08_special_tokens.py      # علائم ویژه و ذخیره‌سازی
│   ├── 09_token_embeddings.py    # تبدیل شناسه‌ها به بردار
│   ├── 10_position_embeddings.py # بردارهای قابل یادگیری جایگاه
│   └── 11_combined_embeddings.py # ترکیب بردارها و نمایش گرادیان
├── tests/
│   ├── test_fundamentals.py      # آزمون‌های مبانی و همگرایی
│   ├── test_tokenizer.py         # آزمون‌های توکن‌ساز
│   └── test_embeddings.py        # آزمون‌های شکل، گرادیان و اعتبار ورودی
├── docs/
│   ├── phase_01.md               # توضیح مفهومی و ریاضی فارسی
│   ├── phase_02.md               # توضیح UTF-8 و توکن‌سازی
│   └── phase_03.md               # توضیح بردارها و یادگیری آن‌ها
├── requirements.txt
├── run_tests.ps1                # اجرای آزمون‌ها با پایتون محیط مجازی
├── run_lesson.ps1               # اجرای یک درس با پایتون محیط مجازی
└── .gitignore
```

## نصب روی Windows و Python 3.11

اگر محیط مجازی هنوز ساخته نشده است، در پوشهٔ `MiniCoderBrain` آن را بسازید. نصب وابستگی‌ها دستی است:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

درس‌های فاز ۰۲ فقط به کتابخانهٔ استاندارد پایتون نیاز دارند؛ PyTorch برای فازهای ۰۱ و ۰۳ و pytest برای تست‌هاست. نصب بسته‌ها ممکن است به اینترنت نیاز داشته باشد، ولی خود پروژه کاملاً آفلاین اجرا می‌شود و دادهٔ آموزشی دانلود نمی‌کند. فعال‌سازی محیط مجازی برای دستورهای زیر لازم نیست؛ مفسر آن صریحاً انتخاب می‌شود.

## اجرای آزمون‌ها و درس‌ها

هر دستور را از پوشهٔ `MiniCoderBrain` اجرا کنید:

```powershell
.\run_tests.ps1
.\run_lesson.ps1 01_tensors.py
.\run_lesson.ps1 02_matrix_operations.py
.\run_lesson.ps1 03_parameters.py
.\run_lesson.ps1 04_autograd.py
.\run_lesson.ps1 05_gradient_descent.py
.\run_lesson.ps1 06_text_encoding.py
.\run_lesson.ps1 07_token_decoding.py
.\run_lesson.ps1 08_special_tokens.py
.\run_lesson.ps1 09_token_embeddings.py
.\run_lesson.ps1 10_position_embeddings.py
.\run_lesson.ps1 11_combined_embeddings.py
```

`run_tests.ps1` وجود محیط مجازی و قابل بارگذاری بودن PyTorch و pytest را بررسی می‌کند، سپس دقیقاً آزمون‌ها را با پایتون همان محیط اجرا می‌کند. کد خروج pytest بدون تغییر بازگردانده می‌شود. هیچ بسته‌ای خودکار نصب نمی‌شود. `run_lesson.ps1` فقط نام فایل پایتون موجود در پوشهٔ `lessons` را می‌پذیرد؛ مسیرهای بیرون پوشه و پیوندها رد می‌شوند. اسکریپت‌ها مسیرها را نسبت به محل خود پیدا می‌کنند و از پوشه‌های دیگر هم قابل فراخوانی‌اند.

برای اجرای مستقیم بدون اسکریپت نیز همیشه مفسر محیط مجازی را مشخص کنید:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -X utf8 lessons/01_tensors.py
```

اگر PyTorch هنوز نصب نیست، درس‌های ۰۶ تا ۰۸ و آزمون‌های مستقل فاز ۰۲ را با `.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -v` اجرا کنید. اجرای کل آزمون‌ها به PyTorch هم نیاز دارد.

درس پنجم ۱۰۱ نمونهٔ ساختگی از رابطهٔ `y = 2x + 1` می‌سازد. وزن و بایاس از ابتدا مقداردهی می‌شوند و با محاسبهٔ گرادیان و گام‌های دستی به مقدارهای نزدیک ۲ و ۱ می‌رسند. [مستندات فاز اول](docs/phase_01.md) محاسبات را توضیح می‌دهد.

توکن‌ساز فاز ۰۲ هر بایت UTF-8 را به همان شناسهٔ ۰ تا ۲۵۵ تبدیل می‌کند. شناسه‌های ویژه ثابت‌اند: `PAD=256`، `BOS=257`، `EOS=258` و `UNK=259`؛ در نتیجه اندازهٔ واژگان ۲۶۰ است. هیچ علامت ویژه‌ای خودکار به متن افزوده نمی‌شود. [مستندات فاز دوم](docs/phase_02.md) تفاوت نویسه، بایت و توکن و محدودیت‌های این روش را توضیح می‌دهد.

فاز ۰۳ در `minicoder/model/embeddings.py` سه ماژول دارد: `TokenEmbedding`، `PositionEmbedding` و `CombinedEmbedding`. اندازهٔ واژگان از توکن‌ساز خوانده می‌شود؛ بُعد بردار و سقف طول زمینه در `minicoder/config.py` هر دو ۱۲۸ هستند. ورودی با شکل `(B, T)` به خروجی `(B, T, 128)` تبدیل می‌شود. جدول توکن ۳۳٬۲۸۰ و جدول موقعیت ۱۶٬۳۸۴ پارامتر دارند؛ مجموع ۴۹٬۶۶۴ پارامتر است. درس یازدهم دریافت گرادیان و تغییر وزن هر دو جدول را با یک خطای ساختگی نشان می‌دهد. برای توضیح ریاضی و مثال‌ها [مستندات فاز سوم](docs/phase_03.md) را بخوانید.
