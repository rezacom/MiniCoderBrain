"""انتخاب شناسه از logits با رتبه‌بندی پایدار و softmax در دقت float64."""

import math

import torch


def validate_sampling_parameters(strategy="greedy", temperature=1.0, top_k=None, top_p=None):
    if strategy not in ("greedy", "sample"):
        raise ValueError("روش انتخاب باید greedy یا sample باشد.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature باید مثبت و متناهی باشد؛ برای انتخاب قطعی از greedy استفاده کنید.")
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k باید عدد صحیح مثبت یا None باشد.")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("top_p باید در بازهٔ صفرِ باز تا یکِ بسته باشد.")
    if strategy == "greedy" and (temperature != 1.0 or top_k is not None or top_p is not None):
        raise ValueError("temperature و top_k و top_p فقط با روش sample قابل تنظیم‌اند.")


def filtered_logits(logits, *, temperature=1.0, top_k=None, top_p=None, forbidden_ids=()):
    """به‌ترتیب دما، Top-K و Top-P؛ دست‌کم یک گزینه باید باقی بماند."""
    validate_sampling_parameters("sample", temperature, top_k, top_p)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits باید تنسور یک‌بعدی و غیرخالی باشد.")
    if logits.device.type != "cpu" or not torch.is_floating_point(logits):
        raise ValueError("logits باید اعشاری و روی CPU باشد.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits نباید NaN یا مثبت بی‌نهایت داشته باشد.")
    scores = logits.detach().to(torch.float64).clone()
    for token_id in forbidden_ids:
        if type(token_id) is not int or not 0 <= token_id < scores.numel():
            raise ValueError("شناسهٔ مسدودشده خارج از واژگان است.")
        scores[token_id] = float("-inf")
    if not torch.isfinite(scores).any():
        raise ValueError("هیچ توکن مجازی برای انتخاب باقی نمانده است.")
    # کم کردن بیشینه پیش از تقسیم، دماهای بسیار کوچک را نیز پایدار نگه می‌دارد.
    scores = (scores - scores.max()) / temperature
    if top_k is not None:
        if top_k > scores.numel():
            raise ValueError("top_k از اندازهٔ واژگان بیشتر است.")
        order = torch.argsort(scores, descending=True, stable=True)
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[order[:top_k]] = True
        scores.masked_fill_(~keep, float("-inf"))
    if top_p is not None and top_p < 1.0:
        order = torch.argsort(scores, descending=True, stable=True)
        probabilities = scores[order].softmax(-1)
        cumulative = probabilities.cumsum(-1)
        # کوچک‌ترین پیشوند با جرم حداقل p؛ گزینهٔ عبور از آستانه حفظ می‌شود.
        remove = torch.cat([torch.zeros(1, dtype=torch.bool), cumulative[:-1] >= top_p])
        scores[order[remove]] = float("-inf")
    return scores


def sample_token(logits, *, strategy="greedy", temperature=1.0, top_k=None, top_p=None,
                 generator=None, forbidden_ids=()):
    validate_sampling_parameters(strategy, temperature, top_k, top_p)
    scores = filtered_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                             forbidden_ids=forbidden_ids)
    if strategy == "greedy":
        return scores.argmax().item()
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("مولد نمونه‌گیری باید روی CPU باشد.")
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).item()
