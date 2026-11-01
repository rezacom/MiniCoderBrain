# MiniCoderBrain — فاز ۰۱: مبانی PyTorch

این پروژه با مثال‌های کوچک و قابل اجرا، پایه‌های ساخت یک مدل زبانی را آموزش می‌دهد. همهٔ محاسبات روی CPU انجام می‌شوند. در این فاز هنوز توکن‌ساز، توجه یا ترنسفورمر وجود ندارد. هیچ مدل آماده، دادهٔ دانلودی یا سرویس ابری استفاده نمی‌شود.

## ساختار

```text
MiniCoderBrain/
├── minicoder/
│   ├── __init__.py
│   └── config.py                 # تنظیمات ثابت و بذر تصادفی
├── lessons/
│   ├── 01_tensors.py             # تنسور، شکل، نوع و تغییر شکل
│   ├── 02_matrix_operations.py   # ضرب ماتریسی و پخش خودکار
│   ├── 03_parameters.py          # وزن‌های قابل یادگیری
│   ├── 04_autograd.py            # مشتق‌گیری خودکار
│   └── 05_gradient_descent.py    # رگرسیون خطی با به‌روزرسانی دستی
├── tests/
│   └── test_fundamentals.py      # آزمون‌های مبانی و همگرایی
├── docs/
│   └── phase_01.md               # توضیح مفهومی و ریاضی فارسی
├── requirements.txt
└── .gitignore
```

## نصب روی Windows و Python 3.11

در پوشهٔ `MiniCoderBrain`، محیط مجازی بسازید و وابستگی‌ها را نصب کنید:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

نصب PyTorch ممکن است برای دریافت بستهٔ نصب به اینترنت نیاز داشته باشد؛ دادهٔ آموزشی دانلود نمی‌شود. اگر اجرای اسکریپت PowerShell محدود بود، دستورات پایین را با `.\.venv\Scripts\python.exe` به‌جای `python` اجرا کنید.

## اجرای درس‌ها

هر دستور را از پوشهٔ `MiniCoderBrain` اجرا کنید:

```powershell
python lessons/01_tensors.py
python lessons/02_matrix_operations.py
python lessons/03_parameters.py
python lessons/04_autograd.py
python lessons/05_gradient_descent.py
python -m pytest tests/
```

درس پنجم ۱۰۱ نمونهٔ ساختگی از رابطهٔ `y = 2x + 1` می‌سازد. وزن و بایاس از ابتدا مقداردهی می‌شوند و با محاسبهٔ گرادیان و گام‌های دستی به مقدارهای نزدیک ۲ و ۱ می‌رسند. برای توضیح دقیق‌تر، [مستندات فاز اول](docs/phase_01.md) را بخوانید.
