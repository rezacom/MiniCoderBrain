"""درس ۸: شناسه‌های ویژه، افزودن اختیاری و ذخیره‌سازی قرارداد."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteTokenizer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tokenizer = ByteTokenizer()
    text = "سلام Python!"
    ordinary = tokenizer.encode(text)
    marked = tokenizer.encode(text, add_bos=True, add_eos=True)

    print("شناسه‌های ثابت: PAD=", PAD_ID, "BOS=", BOS_ID,
          "EOS=", EOS_ID, "UNK=", UNK_ID)
    print("اندازهٔ واژگان:", tokenizer.vocab_size)
    print("بدون افزودن خودکار:", ordinary)
    print("با BOS و EOS به درخواست برنامه:", marked)
    print("نمایش علائم ویژه:", tokenizer.decode(marked))
    print("حذف علائم ویژه هنگام بازسازی متن:", tokenizer.decode(marked, skip_special_tokens=True))

    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        restored = ByteTokenizer.load(path)
        print("بازسازی پس از ذخیره و بارگذاری:", restored.decode(restored.encode(text)))

    print("UNK رزرو شده است، ولی هر متن معتبر UTF-8 را می‌توان با بایت‌ها نوشت؛")
    print("بنابراین در encode معمولی به توکن ناشناخته نیاز نداریم.")


if __name__ == "__main__":
    main()
