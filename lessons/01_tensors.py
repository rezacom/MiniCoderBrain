"""درس ۱: ساخت، شکل و دسترسی به داده‌های تنسوری."""

import torch


def main():
    # تنسور صفر بعدی یک عدد منفرد، یعنی اسکالر، است.
    scalar = torch.tensor(7)
    # بردار یک محور و ماتریس دو محور دارد.
    vector = torch.tensor([10, 20, 30])
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print("اسکالر:", scalar, "| تعداد بعدها:", scalar.ndim, "| شکل:", scalar.shape)
    print("بردار:", vector, "| تعداد بعدها:", vector.ndim, "| شکل:", vector.shape)
    print("ماتریس:\n", matrix, "\nشکل:", matrix.shape)
    print("نوع دادهٔ بردار:", vector.dtype, "| نوع دادهٔ ماتریس:", matrix.dtype)

    # اندیس‌ها از صفر شروع می‌شوند. اینجا سطر دوم و ستون سوم را می‌خوانیم.
    print("عضو سطر دوم، ستون سوم:", matrix[1, 2].item())
    print("سطر اول:", matrix[0])

    # تغییر شکل تعداد عنصرها را عوض نمی‌کند: ۲×۳ و ۳×۲ هر دو ۶ عنصر دارند.
    reshaped = matrix.reshape(3, 2)
    print("ماتریس پس از تغییر شکل به ۳×۲:\n", reshaped)


if __name__ == "__main__":
    main()
