"""درس ۲۹: حلقهٔ کوتاه AdamW، اعتبارسنجی و checkpoint کامل."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader, split_documents, write_synthetic_corpus
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig, seed_everything
from minicoder.training.checkpoint import load_checkpoint


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seed_everything()
    with TemporaryDirectory() as directory:
        paths = write_synthetic_corpus(Path(directory) / "synthetic")
        train_docs, val_docs = split_documents(TextDataset(paths, add_bos=True, add_eos=True))
        train_data, val_data = CausalLanguageModelDataset(train_docs, 16), CausalLanguageModelDataset(val_docs, 16)
        model = MiniCoderGPT(ModelConfig(embedding_dim=32, num_heads=4, num_layers=1, hidden_dim=64, max_context_length=16))
        trainer = Trainer(model, create_dataloader(train_data, 2), create_dataloader(val_data, 2, shuffle=False),
                          config=TrainingConfig(max_steps=8, learning_rate=0.01, eval_interval=4))
        print("آموزش کوتاه روی پیکرهٔ ساختگی؛ این داده برای مدل عمومی کافی نیست.")
        for record in trainer.fit():
            print("گام:", record["step"], "| خطای آموزش:", round(record["train_loss"], 4),
                  "| اعتبارسنجی:", record["validation_loss"])
        path = Path(directory) / "step_8.pt"
        trainer.save_checkpoint(path)
        saved = load_checkpoint(path)
        print("گام ذخیره‌شده:", saved["step"], "| حالت‌های AdamW:", len(saved["optimizer_state"]["state"]))
        print("توکن‌های پردازش‌شده:", trainer.tokens_processed, "| زمان فعال:", trainer.duration_seconds)
        print("checkpoint شامل وزن، بهینه‌ساز، تنظیمات، RNG و مکان نمونه‌گیر است.")


if __name__ == "__main__":
    main()
