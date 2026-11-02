# گزارش ارزیابی MiniCoder Brain

نسخهٔ پروژه: 1.0.0

وضعیت مدل: آموزش‌ندیده

منبع سوابق آموزش:

```json
null
```

| مشخصه | مقدار |
| --- | ---: |
| پارامترهای یکتا | 876548 |
| پارامترهای قابل یادگیری | 876548 |
| بلوک‌های ترنسفورمر | 4 |
| سرهای هر بلوک | 4 |
| اندازهٔ واژگان | 260 |
| سقف زمینه | 128 |
| گام آموزش ثبت‌شده | 0 |
| آخرین خطای دستهٔ آموزش پیش از به‌روزرسانی | موجود نیست |
| آخرین خطای اعتبارسنجی ثبت‌شده | موجود نیست |
| گام آن اعتبارسنجی | موجود نیست |
| خطای اعتبارسنجی ثبت‌شده در گام نهایی | موجود نیست |

تنظیمات معماری:

```json
{
  "vocab_size": 260,
  "embedding_dim": 128,
  "num_heads": 4,
  "num_layers": 4,
  "hidden_dim": 512,
  "max_context_length": 128,
  "layer_norm_eps": 1e-05,
  "ignore_index": 256,
  "tie_weights": false
}
```

تنظیمات آموزش ثبت‌شده یا برنامه‌ریزی‌شده؛ null یعنی ارائه نشده:

```json
null
```

خطای دستهٔ آموزش، میانگین کل پیکره یا خطای وزن‌ها پس از آخرین به‌روزرسانی نیست.

## ارزیابی تازهٔ داده

پیکرهٔ ساختگی نمایشی؛ تقسیم اسناد به مرجع و اعتبارسنجی، به‌معنای آموزش قبلی مدل نیست.

این اندازه‌گیری‌ها تاریخچهٔ آموزش نیستند. برای مدل آموزش‌ندیده، فقط رفتار وزن‌های تصادفی را می‌سنجند.

```json
{
  "label": "پیکرهٔ ساختگی نمایشی؛ تقسیم اسناد به مرجع و اعتبارسنجی، به‌معنای آموزش قبلی مدل نیست.",
  "training_reference": {
    "loss": 5.752567855174129,
    "perplexity": 314.9984931533803,
    "token_accuracy": 0.0009380863039399625,
    "exact_sequence_accuracy": 0.0,
    "tokens": 6396,
    "examples": 202,
    "batches": 101,
    "fingerprint": "955db1214292f846dacb8c19f990c3d992aa05c2c6dcc488c9520fda26afa9a7",
    "loader_configuration": {
      "size": 202,
      "batch_size": 2,
      "shuffle": false,
      "seed": 42,
      "drop_last": false
    },
    "scope": "پیش‌بینی توکن بعدی با teacher forcing؛ میانگین وزن‌دار بر هدف‌های غیر PAD"
  },
  "validation": {
    "loss": 5.698296217171543,
    "perplexity": 298.35862936254375,
    "token_accuracy": 0.0024096385542168677,
    "exact_sequence_accuracy": 0.0,
    "tokens": 1245,
    "examples": 39,
    "batches": 20,
    "fingerprint": "34ddc78d665185a055e533a14c66b6f6473cfb419b91cb0e44b3966bdbae3f99",
    "loader_configuration": {
      "size": 39,
      "batch_size": 2,
      "shuffle": false,
      "seed": 42,
      "drop_last": false
    },
    "scope": "پیش‌بینی توکن بعدی با teacher forcing؛ میانگین وزن‌دار بر هدف‌های غیر PAD"
  },
  "reference_matches_checkpoint_training": null,
  "validation_disjoint_from_checkpoint_training": null
}
```

## آزمون محدود یادآوری

```json
{
  "task": "memorization",
  "known_training_document": false,
  "demonstration_only": true,
  "scope": "پیش‌بینی با ورودی صحیح قبلی؛ teacher forcing",
  "token_count": 7,
  "truncated": false,
  "loss": 5.698413848876953,
  "perplexity": 298.39372786124716,
  "token_accuracy": 0.0,
  "exact_token_match": false,
  "text_sha256": "06e39dc6170e54239d73836b0574a2482720539f91ed8ca308b9e3a1a51225d2",
  "add_bos": false,
  "add_eos": false,
  "interpretation": "تطبیق روی این توالی کوچک، شاهد فهم زبان یا برنامه‌نویسی نیست."
}
```

## نمونه‌های نمایشی تولید

هیچ کد تولیدی اجرا نشده و هیچ ادعای فهم برنامه‌نویسی از ظاهر خروجی نتیجه‌گیری نمی‌شود.

### ادامهٔ کوتاه انگلیسی

```json
{
  "prompt": "Hello, the small cat",
  "generated_text": "P\\xa5\\xb9\\x85\\x9e]\\xbb?",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 8
}
```

### ادامهٔ کوتاه فارسی

```json
{
  "prompt": "سلام، امروز",
  "generated_text": "\\xa6\\xb9\\x85\\x9e]\\xbb?\\xdc",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 8
}
```

### ادامهٔ متن پایتون؛ بدون اجرای کد

```json
{
  "prompt": "def add(a, b):\n    ",
  "generated_text": "9\\x8d\\xa5\\xb9\\x85\\x9e]\\xbb",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 8
}
```

## سرعت و حافظه

```json
{
  "inference": {
    "new_tokens": 24,
    "duration_seconds": 0.28186180000193417,
    "tokens_per_second": 85.14811159169248,
    "warmup_forward_passes": 1,
    "scope": "تولید انتهابه‌انتها، شامل نمونه‌گیری و رمزگشایی؛ توکن بایتی و EOS در شمارش‌اند"
  },
  "memory": {
    "before": {
      "available": true,
      "rss_bytes": 210120704,
      "peak_rss_bytes": 211845120,
      "source": "Windows working set",
      "scope": "کل فرایند؛ اوج از آغاز عمر فرایند"
    },
    "after": {
      "available": true,
      "rss_bytes": 300281856,
      "peak_rss_bytes": 300302336,
      "source": "Windows working set",
      "scope": "کل فرایند؛ اوج از آغاز عمر فرایند"
    }
  },
  "environment": {
    "python": "3.11.0",
    "torch": "2.15.0+cpu",
    "platform": "Windows",
    "device": "cpu",
    "cpu_threads": 2
  }
}
```

## آزمون‌های خودکار

تعداد پاس‌شده: 251 از 251؛ شکست: 0؛ خطا: 0؛ ردشده: 0.

## محدودیت‌ها

- نمونه‌های تولید شاهد فهم زبان یا برنامه‌نویسی نیستند؛ هیچ کدی اجرا نمی‌شود.
- تست یادآوری روی دادهٔ ناشناختهٔ آموزش فقط نمایش سازوکار است.
- زمان و حافظه اندازه‌گیری محیط جاری‌اند و بین اجراها یکسان نمی‌مانند.
- اسناد متفاوت با بخش‌های مشابه خودکار حذف نمی‌شوند؛ جداسازی بر هش دقیق سند تکیه دارد.
