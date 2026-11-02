"""پروفایل ۵۰۰گامی و ماندگاری تاریخچه؛ آزمایش اصلی اجرا نمی‌شود."""

from dataclasses import asdict
import json
import random

import pytest
import torch

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.experiments.python_training import (
    DEFAULT_CONFIG, PROJECT_ROOT, load_python_experiment_config, prepare_python_training,
)
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig, seed_everything
from minicoder.training.checkpoint import load_checkpoint
from scripts import train as existing_train


EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "python_training_v1_1_step500.json"
EXPECTED_TRAINING = {
    "max_steps": 500, "batch_size": 2, "learning_rate": 0.0003,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "eval_interval": 50,
    "seed": 42, "cpu_threads": 4,
}


@pytest.fixture(autouse=True)
def restore_environment():
    python_state, threads = random.getstate(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(python_state)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def test_500_profile_reaches_existing_cli_without_resuming_or_using_test(tmp_path):
    plan = prepare_python_training(EXPERIMENT_CONFIG, checkpoint=tmp_path / "python_v1_1_step500.pt")
    args = existing_train.parser().parse_args(list(plan.cli_arguments))
    assert asdict(plan.training_config) == EXPECTED_TRAINING
    assert plan.checkpoint == tmp_path / "python_v1_1_step500.pt"
    assert load_python_experiment_config(EXPERIMENT_CONFIG)["checkpoint_pattern"] == "checkpoints/python_v1_1_step500.pt"
    assert args.resume is None
    assert args.steps == 500 and args.eval_interval == 50 and args.threads == 4
    assert args.batch_size == 2 and args.sequence_length == args.stride == 128
    assert args.learning_rate == 0.0003 and args.weight_decay == 0.01
    assert args.grad_clip == 1.0 and args.seed == 42
    assert args.embedding_dim is args.num_heads is args.num_layers is args.hidden_dim is None
    assert args.max_context_length is None
    assert len(args.files) == 360
    assert set(args.files) == set(plan.files["train"] + plan.files["validation"])
    assert set(args.files).isdisjoint(plan.files["test"])
    assert load_python_experiment_config(DEFAULT_CONFIG)["training"]["max_steps"] == 5
    assert load_python_experiment_config(DEFAULT_CONFIG)["training"]["cpu_threads"] == 2


def test_500_pipeline_stops_before_fit_and_preserves_cpu_seed_and_splits(tmp_path, monkeypatch):
    plan = prepare_python_training(EXPERIMENT_CONFIG, checkpoint=tmp_path / "step500.pt")
    args = existing_train.parser().parse_args(list(plan.cli_arguments))
    calls = []
    original_seed = existing_train.seed_everything

    class ReachedTrainingBoundary(Exception):
        pass

    def seed_spy(seed, *, cpu_threads):
        calls.append((seed, cpu_threads))
        original_seed(seed, cpu_threads=cpu_threads)

    def reject_checkpoint_load(*args, **kwargs):
        pytest.fail("اجرای تصادفی نباید checkpoint را بارگذاری کند")

    class BoundaryTrainer:
        def __init__(self, model, train_loader, validation_loader, *, config):
            self.model = model
            assert asdict(config) == EXPECTED_TRAINING
            assert model.config == ModelConfig()
            assert model.trainable_parameter_count == 876548
            assert all(parameter.device.type == "cpu" for parameter in model.parameters())
            assert torch.get_num_threads() == 4
            assert torch.are_deterministic_algorithms_enabled()
            for split, loader, count in (("train", train_loader, 320), ("validation", validation_loader, 40)):
                assert len(loader.dataset) == count
                assert {document.source for document in loader.dataset.documents} == {str(path) for path in plan.files[split]}
                assert loader.batch_sampler.batch_size == 2 and loader.batch_sampler.seed == 42
                assert loader.batch_sampler.shuffle is (split == "train")
                inputs, targets = next(iter(loader))
                assert inputs.shape == targets.shape == (2, 128)
                assert inputs.device.type == targets.device.type == "cpu"

        def fit(self, **kwargs):
            raise ReachedTrainingBoundary()

    monkeypatch.setattr(existing_train, "Trainer", BoundaryTrainer)
    monkeypatch.setattr(existing_train, "seed_everything", seed_spy)
    monkeypatch.setattr(existing_train, "load_checkpoint", reject_checkpoint_load)
    with pytest.raises(ReachedTrainingBoundary):
        existing_train.run(args, tmp_path, document_splitter=plan.split_documents)
    assert calls == [(42, 4)]
    assert not plan.checkpoint.exists()


def test_training_history_round_trips_with_periodic_and_final_validation(tmp_path):
    # سه گام روی مدل آزمایشی کوچک؛ نه اجرای پروفایل ۵۰۰گامی یا پیکرهٔ پروژه.
    seed_everything(42, cpu_threads=2)
    train_path, validation_path = tmp_path / "train.txt", tmp_path / "validation.txt"
    train_path.write_text("AB" * 12, encoding="utf-8")
    validation_path.write_text("ABC" * 8, encoding="utf-8")
    train_data = CausalLanguageModelDataset(TextDataset([train_path]), 4)
    validation_data = CausalLanguageModelDataset(TextDataset([validation_path]), 4)
    model = MiniCoderGPT(ModelConfig(embedding_dim=8, num_heads=2, num_layers=1,
                                   hidden_dim=16, max_context_length=4))
    config = TrainingConfig(max_steps=3, eval_interval=2)
    trainer = Trainer(model, create_dataloader(train_data, 2, seed=42),
                      create_dataloader(validation_data, 2, shuffle=False, seed=42), config=config)
    trainer.fit()
    checkpoint = tmp_path / "history.pt"
    trainer.save_checkpoint(checkpoint)
    saved = load_checkpoint(checkpoint)
    assert saved["step"] == 3
    assert saved["history"] == trainer.history
    assert [row["step"] for row in saved["history"]] == [1, 2, 3]
    assert [row["step"] for row in saved["history"] if row["validation_loss"] is not None] == [2, 3]
    assert all(row["train_loss"] > 0 for row in saved["history"])
    assert saved["history"][0]["validation_loss"] is None
    assert json.loads(json.dumps(saved["history"], allow_nan=False)) == saved["history"]
    trainer.load_checkpoint(checkpoint)
    assert trainer.history == saved["history"]


def test_500_profile_refuses_existing_output_without_changing_it(tmp_path):
    checkpoint = tmp_path / "python_v1_1_step500.pt"
    checkpoint.write_bytes(b"keep existing checkpoint")
    with pytest.raises(FileExistsError):
        prepare_python_training(EXPERIMENT_CONFIG, checkpoint=checkpoint)
    assert checkpoint.read_bytes() == b"keep existing checkpoint"
