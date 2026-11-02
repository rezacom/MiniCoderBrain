"""درس ۳۰: رسیدن گرادیان به GPT، محدود کردن نرم و گام AdamW."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig, seed_everything


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seed_everything()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic_repetition.txt"
        path.write_text("AB" * 64, encoding="utf-8")
        data = CausalLanguageModelDataset(TextDataset([path]), 16, short_document_policy="drop")
        model = MiniCoderGPT(ModelConfig(embedding_dim=16, num_heads=2, num_layers=1, hidden_dim=32, max_context_length=16))
        trainer = Trainer(model, create_dataloader(data, 2),
                          config=TrainingConfig(max_steps=1, learning_rate=0.01, max_grad_norm=0.1))
        before = model.lm_head.weight[0, :5].detach().clone()
        record = trainer.train_step()
        print("دادهٔ ساختگی تکراری فقط برای مشاهدهٔ مسیر آموزش است.")
        print("خطای توکن بعدی:", record["train_loss"])
        print("نرم گرادیان پیش از clipping:", record["grad_norm_before"])
        print("نرم گرادیان پس از clipping:", record["grad_norm_after"])
        print("گرادیان جدول توکن:", model.embeddings.token_embedding.embedding.weight.grad.norm().item())
        print("گرادیان Query:", model.blocks[0].attention.query.weight.grad.norm().item())
        print("پنج وزن سر خروجی پیش از گام:", before)
        print("پنج وزن پس از گام AdamW:", model.lm_head.weight[0, :5].detach())
        print("ترتیب گام: پاک‌سازی گرادیان، forward، loss، backward، clipping و optimizer.step")


if __name__ == "__main__":
    main()
