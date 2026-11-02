# گزارش ارزیابی MiniCoder Brain

نسخهٔ پروژه: 1.1.0

وضعیت مدل: دارای گام آموزش ثبت‌شده در checkpoint

منبع سوابق آموزش:

```json
"python_v1_1_step5.pt"
```

| مشخصه | مقدار |
| --- | ---: |
| پارامترهای یکتا | 876548 |
| پارامترهای قابل یادگیری | 876548 |
| بلوک‌های ترنسفورمر | 4 |
| سرهای هر بلوک | 4 |
| اندازهٔ واژگان | 260 |
| سقف زمینه | 128 |
| گام آموزش ثبت‌شده | 5 |
| آخرین خطای دستهٔ آموزش پیش از به‌روزرسانی | 5.224786 |
| آخرین خطای اعتبارسنجی ثبت‌شده | 5.100880 |
| گام آن اعتبارسنجی | 5 |
| خطای اعتبارسنجی ثبت‌شده در گام نهایی | 5.100880 |

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
{
  "max_steps": 5,
  "batch_size": 2,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "eval_interval": 5,
  "seed": 42,
  "cpu_threads": 2
}
```

خطای دستهٔ آموزش، میانگین کل پیکره یا خطای وزن‌ها پس از آخرین به‌روزرسانی نیست.

## ارزیابی تازهٔ داده

پیکرهٔ پایتون ۱٫۱؛ خانواده‌های ساختاری جدا و هدف‌های یک توکن جلوتر.

این اندازه‌گیری‌ها تاریخچهٔ آموزش نیستند. برای مدل آموزش‌ندیده، فقط رفتار وزن‌های تصادفی را می‌سنجند.

```json
{
  "label": "پیکرهٔ پایتون ۱٫۱؛ خانواده‌های ساختاری جدا و هدف‌های یک توکن جلوتر.",
  "training_reference": {
    "loss": 5.032820472909159,
    "perplexity": 153.36496498335083,
    "token_accuracy": 0.2081106570260924,
    "exact_sequence_accuracy": 0.0,
    "tokens": 15905,
    "examples": 320,
    "batches": 160,
    "fingerprint": "9ddc6a34355993b059e583a3b813870d4b35e3bbaa569acea842dae42c4280f9",
    "loader_configuration": {
      "size": 320,
      "batch_size": 2,
      "shuffle": false,
      "seed": 42,
      "drop_last": false
    },
    "scope": "پیش‌بینی توکن بعدی با teacher forcing؛ میانگین وزن‌دار بر هدف‌های غیر PAD"
  },
  "validation": {
    "loss": 5.100879519504258,
    "perplexity": 164.16623122504765,
    "token_accuracy": 0.1903566047212456,
    "exact_sequence_accuracy": 0.0,
    "tokens": 1991,
    "examples": 40,
    "batches": 20,
    "fingerprint": "f37fc1e18c68b28371eee09af3deb3bfb5ee6732752f7ba7e4860ef0c5077f20",
    "loader_configuration": {
      "size": 40,
      "batch_size": 2,
      "shuffle": false,
      "seed": 42,
      "drop_last": false
    },
    "scope": "پیش‌بینی توکن بعدی با teacher forcing؛ میانگین وزن‌دار بر هدف‌های غیر PAD"
  },
  "test": null,
  "test_evaluated": false,
  "reference_matches_checkpoint_training": true,
  "validation_disjoint_from_checkpoint_training": true
}
```

## آزمون محدود یادآوری

```json
{
  "scope": "خطا و دقت روی train در کنار خانواده‌های دیده‌نشدهٔ validation/test",
  "interpretation": "شکاف آموزش و دادهٔ جدا ممکن است نشانهٔ بیش‌برازش باشد؛ مثال تولید شاهد فهم نیست."
}
```

## نمونه‌های نمایشی تولید

هیچ کد تولیدی اجرا نشده و هیچ ادعای فهم برنامه‌نویسی از ظاهر خروجی نتیجه‌گیری نمی‌شود.

### پرسش 1، روش greedy

```json
{
  "prompt": "def add(a, b):",
  "generated_text": "  =      \u0005\\xfdo                                                    ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "greedy",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "greedy",
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "seed": 42,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 5,
    "repeated_token_fraction": 0.921875,
    "repeated_trigram_fraction": 0.8548387096774194,
    "longest_identical_run": 52,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 1,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 2، روش greedy

```json
{
  "prompt": "def square(x):",
  "generated_text": "  =      \u0005\\xfdo                                                    ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "greedy",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "greedy",
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "seed": 43,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 5,
    "repeated_token_fraction": 0.921875,
    "repeated_trigram_fraction": 0.8548387096774194,
    "longest_identical_run": 52,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 0,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 3، روش greedy

```json
{
  "prompt": "for i in range(",
  "generated_text": "\\xab6      \u0005\\xfdo                                                     ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "greedy",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "greedy",
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "seed": 44,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 6,
    "repeated_token_fraction": 0.90625,
    "repeated_trigram_fraction": 0.8709677419354839,
    "longest_identical_run": 53,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 4,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 4، روش greedy

```json
{
  "prompt": "if x > 0:",
  "generated_text": "       k      \u0005\\xfdo                                               ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "greedy",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "greedy",
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "seed": 45,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 5,
    "repeated_token_fraction": 0.921875,
    "repeated_trigram_fraction": 0.8548387096774194,
    "longest_identical_run": 47,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 2,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 5، روش greedy

```json
{
  "prompt": "def reverse_string(text):",
  "generated_text": "\\xab        =  )     \\xf2\\xb5                                            ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "greedy",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "greedy",
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "seed": 46,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 6,
    "repeated_token_fraction": 0.90625,
    "repeated_trigram_fraction": 0.8064516129032258,
    "longest_identical_run": 44,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 0,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 1، روش sample

```json
{
  "prompt": "def add(a, b):",
  "generated_text": "\\xbb\\xdc\\xefe\\xfd\u000f\r\\xa9\u0007[\\x8e\\xd6\\xf8\\x8e\\xf1S\\xf1",
  "utf8_valid": false,
  "finish_reason": "eos",
  "new_tokens": 20,
  "strategy": "sample",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "sample",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 42,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 19,
    "unique_byte_tokens": 17,
    "repeated_token_fraction": 0.10526315789473684,
    "repeated_trigram_fraction": 0.0,
    "longest_identical_run": 1,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 1,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 2، روش sample

```json
{
  "prompt": "def square(x):",
  "generated_text": " \\xe3_\\x86\\xe5&e\u0014\\xa4\u000b\\xdf̀h\\x85$`\\xb8*\\xf5\\xe3-s\u000b\\x9b\u0004\\xb9\\xe7\\xf1\u0014p\\xfd\u001c\\xed\u0006\\x9c\np\\xe6&CĊrB\u0011\\xcf\\xdf\u001e\\xbd2\\xe0\\xbeL#R\\xbc;G\\xc4_\\xd1e ",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "sample",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "sample",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 43,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 54,
    "repeated_token_fraction": 0.15625,
    "repeated_trigram_fraction": 0.0,
    "longest_identical_run": 1,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 0,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 3، روش sample

```json
{
  "prompt": "for i in range(",
  "generated_text": "s\\xdf$f  K\\xa5\\xd3S+)\\xaf\\xd6\\xd9B\\xe6\\x8c\rE0\\xa97y?\\xbd\\xe1F4\\xd10\\xfe\\xc0\\x8c/\\x8bTS\u0010\\xf4\\xb4\n\\x80[\\xed\u0005>\\xe9\\x98 \\x8ci\t\\xfd\\xb3A\u0007\\xd5\\xe9\\xc0C{{a",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "sample",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "sample",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 44,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 55,
    "repeated_token_fraction": 0.140625,
    "repeated_trigram_fraction": 0.0,
    "longest_identical_run": 2,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 4,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 4، روش sample

```json
{
  "prompt": "if x > 0:",
  "generated_text": "#\\xbe^\\x9de\\x8ce\\xbd Y?ۥ\\x91\r\\xa0d\u0013$\\xc5<K\\xfa\\xbd$\\xe3\\xdf\u000f.Qe\u00077\\xef\\x86z4s\n㭂\\x87Ep\\xb3\\xb4\\xd9`\\xe6Om\n\\xb8\\xfd\\x9b\\xa1\\xbdg\\x8e>\\xb9\\xe6+",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "sample",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "sample",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 45,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 56,
    "repeated_token_fraction": 0.125,
    "repeated_trigram_fraction": 0.0,
    "longest_identical_run": 1,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 2,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### پرسش 5، روش sample

```json
{
  "prompt": "def reverse_string(text):",
  "generated_text": "4\\xbb\n\u0013uL\u00189\\xd3\r\\x86\\xf0\u0002\\x9b\\xaa\\xa5٫\\x86v^\\x9c7\\xe2\\xfc\\xdfw\\xd5\\xe3\\xec\\x8f\\xde\\xcb_x\\xb8\\x9e\\xc7\u001c\\xf5\\xbf\f4\\x8e\\xaa\\xd6j\\xd9\\xd6ej.*\\x90e ѕ\\x9d*\\xec\u0005\\x90U",
  "utf8_valid": false,
  "finish_reason": "max_new_tokens",
  "new_tokens": 64,
  "strategy": "sample",
  "sampling_configuration": {
    "max_new_tokens": 64,
    "strategy": "sample",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 46,
    "utf8_errors": "backslashreplace"
  },
  "repetition": {
    "byte_tokens": 64,
    "unique_byte_tokens": 54,
    "repeated_token_fraction": 0.15625,
    "repeated_trigram_fraction": 0.0,
    "longest_identical_run": 1,
    "interpretation": "تکرار بایت، نه واژه؛ این اعداد به طول خروجی وابسته‌اند و امتیاز فهم نیستند."
  },
  "python_syntax": {
    "checked": false,
    "valid": null,
    "reason": "بایت خام UTF-8 معتبر نیست؛ متن نمایشی جای برنامهٔ اصلی گرفته نمی‌شود."
  },
  "prompt_exposure": {
    "configured_training_examples_containing_fragment": 0,
    "interpretation": "وجود در پیکره به‌معنای دیده‌شدن در گام‌های انجام‌شده نیست؛ نبود عبارت هم اثبات تعمیم نیست."
  }
}
```

### مقایسهٔ روش‌های تولید

```json
{
  "greedy": {
    "examples": 5,
    "utf8_valid_examples": 0,
    "syntax_checked": 0,
    "syntax_valid": 0,
    "syntax_valid_fraction": null,
    "mean_repeated_token_fraction": 0.915625,
    "mean_repeated_trigram_fraction": 0.8483870967741935
  },
  "sample": {
    "examples": 5,
    "utf8_valid_examples": 0,
    "syntax_checked": 0,
    "syntax_valid": 0,
    "syntax_valid_fraction": null,
    "mean_repeated_token_fraction": 0.13667763157894736,
    "mean_repeated_trigram_fraction": 0.0
  }
}
```

## سرعت و حافظه

```json
{
  "inference": {
    "new_tokens": 596,
    "duration_seconds": 18.10549100000935,
    "tokens_per_second": 32.9181903986858,
    "scope": "تولید انتهابه‌انتها؛ زمان معیار کیفیت نیست."
  },
  "memory": {
    "before": {
      "available": true,
      "rss_bytes": 224460800,
      "peak_rss_bytes": 226549760,
      "source": "Windows working set",
      "scope": "کل فرایند؛ اوج از آغاز عمر فرایند"
    },
    "after": {
      "available": true,
      "rss_bytes": 304599040,
      "peak_rss_bytes": 312651776,
      "source": "Windows working set",
      "scope": "کل فرایند؛ اوج از آغاز عمر فرایند"
    }
  },
  "environment": {
    "device": "cpu",
    "cpu_threads": 2,
    "seed": 42
  }
}
```

## آزمون‌های خودکار

فایل نتیجهٔ آزمون معرفی نشده است؛ تعداد آزمون حدس زده نمی‌شود.

## محدودیت‌ها

- نحو معتبر به‌معنای صحت برنامه یا فهم برنامه‌نویسی نیست؛ کدی اجرا نمی‌شود.
- اعداد تکرار برحسب بایت‌اند و با طول تولید تغییر می‌کنند.
- validation برای انتخاب تنظیمات است؛ test را فقط پس از انتخاب نهایی ارزیابی کنید.
- گروه‌بندی AST تضمین نبود برنامه‌های هم‌معنای دارای ساختار متفاوت نیست.
