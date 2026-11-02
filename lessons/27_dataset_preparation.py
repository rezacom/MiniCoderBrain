"""درس ۲۷: خواندن اسناد صریح و تفکیک پیش از پنجره‌سازی."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data import CausalLanguageModelDataset, TextDataset, split_documents, write_synthetic_corpus


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with TemporaryDirectory() as directory:
        paths = write_synthetic_corpus(Path(directory) / "synthetic")
        documents = TextDataset(paths, add_bos=True, add_eos=True)
        train, validation = split_documents(documents)
        print("این پیکره ساختگی و فقط برای نمایش آموزشی است.")
        for document in documents:
            print(Path(document.source).name, "| نویسه:", len(document.text), "| توکن:", len(document.token_ids))
        print("تعداد اسناد آموزش و اعتبارسنجی:", len(train), len(validation))
        print("اشتراک محتوای سندها:", {doc.content_hash for doc in train} & {doc.content_hash for doc in validation})
        windows = CausalLanguageModelDataset(train, sequence_length=128)
        inputs, targets = windows[0]
        print("شکل ورودی و هدف:", tuple(inputs.shape), tuple(targets.shape))
        print("ده شناسهٔ ورودی:", inputs[:10])
        print("ده هدفِ یک توکن جلوتر:", targets[:10])
        print("برای ۱۲۸ ورودی و هدف کامل، پنجره ۱۲۹ توکن می‌خواند.")


if __name__ == "__main__":
    main()
