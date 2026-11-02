"""آموزش، بیش‌برازش ساختگی، clipping و ادامهٔ دقیق مدل و بهینه‌ساز."""

from copy import deepcopy
from dataclasses import replace
import math
import random

import pytest
import torch
from torch.nn import functional as F

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig, perplexity, seed_everything
from minicoder.training.checkpoint import load_checkpoint


@pytest.fixture(autouse=True)
def fixed_environment():
    python_state, threads = random.getstate(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        seed_everything(42, cpu_threads=2)
        yield
    random.setstate(python_state)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def build_trainer(tmp_path, *, validation=True, config=None, seed=42, training_text="AB" * 20):
    seed_everything(seed, cpu_threads=2)
    train_path, val_path = tmp_path / "train.txt", tmp_path / "validation.txt"
    train_path.write_text(training_text, encoding="utf-8")
    val_path.write_text("ABCDEF" * 7, encoding="utf-8")
    train_data = CausalLanguageModelDataset(TextDataset([train_path]), 8, short_document_policy="drop")
    val_data = CausalLanguageModelDataset(TextDataset([val_path]), 8)
    config = config or TrainingConfig(max_steps=6, learning_rate=0.01, weight_decay=0.0)
    model = MiniCoderGPT(ModelConfig(embedding_dim=16, num_heads=2, num_layers=1, hidden_dim=32, max_context_length=8))
    return Trainer(model, create_dataloader(train_data, config.batch_size, seed=config.seed),
                   create_dataloader(val_data, config.batch_size, shuffle=False, seed=config.seed) if validation else None,
                   config=config)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


def test_forward_backward_optimizer_updates_cpu_and_metrics(tmp_path):
    trainer = build_trainer(tmp_path)
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    history = trainer.fit()
    assert trainer.step == 6 and trainer.tokens_processed == 6 * 2 * 8
    assert len(history) == 6 and history[-1]["validation_loss"] is not None
    assert history[-1]["duration_seconds"] > 0
    assert trainer.optimizer.state_dict()["state"]
    assert any(not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters())
    assert all(parameter.device.type == "cpu" for parameter in trainer.model.parameters())
    assert all(math.isfinite(record["train_loss"]) for record in history)


def test_tiny_repetitive_corpus_can_be_overfit(tmp_path):
    config = TrainingConfig(max_steps=60, learning_rate=0.03, weight_decay=0.0)
    trainer = build_trainer(tmp_path, validation=False, config=config)
    inputs, targets = trainer.train_loader.dataset[0]
    _, initial = trainer.model(inputs.unsqueeze(0), targets.unsqueeze(0))
    trainer.fit()
    _, final = trainer.model(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert initial.item() > 1.0
    assert final.item() < 0.1
    assert final.item() < initial.item() * 0.05


def test_gradient_clipping_is_applied_before_optimizer_step(tmp_path):
    config = TrainingConfig(max_grad_norm=1e-4, learning_rate=0.01)
    trainer = build_trainer(tmp_path, config=config)
    record = trainer.train_step()
    assert record["grad_norm_before"] > config.max_grad_norm
    assert record["grad_norm_after"] <= config.max_grad_norm + 1e-8


def test_validation_is_weighted_by_valid_targets_and_restores_state(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.model.train()
    sampler_before = trainer.validation_loader.batch_sampler.state_dict()
    optimizer_before = deepcopy(trainer.optimizer.state_dict())
    total, count = 0.0, 0
    with torch.no_grad():
        for inputs, targets in trainer.validation_loader:
            logits, _ = trainer.model(inputs)
            total += F.cross_entropy(logits.reshape(-1, 260), targets.reshape(-1),
                                     ignore_index=trainer.model.config.ignore_index, reduction="sum").item()
            count += (targets != trainer.model.config.ignore_index).sum().item()
    trainer.validation_loader.batch_sampler.load_state_dict(sampler_before)
    result = trainer.evaluate()
    assert result["loss"] == pytest.approx(total / count, rel=1e-6)
    assert result["tokens"] == count
    assert trainer.model.training
    assert trainer.validation_loader.batch_sampler.state_dict() == sampler_before
    assert_nested_equal(trainer.optimizer.state_dict(), optimizer_before)


@pytest.mark.parametrize("cut", [0, 1, 2, 3])
def test_resume_matches_uninterrupted_training_across_epochs(tmp_path, cut):
    uninterrupted = build_trainer(tmp_path)
    uninterrupted.fit(6)
    expected_model = deepcopy(uninterrupted.model.state_dict())
    expected_optimizer = deepcopy(uninterrupted.optimizer.state_dict())
    expected_loss = [record["train_loss"] for record in uninterrupted.history]
    expected_sampler = uninterrupted.train_loader.batch_sampler.state_dict()
    expected_generator = uninterrupted.train_loader.generator.get_state()
    expected_torch_random, expected_python_random = torch.rand(3), random.random()

    interrupted = build_trainer(tmp_path)
    if cut:
        interrupted.fit(cut)
    path = tmp_path / f"step_{cut}.pt"
    interrupted.save_checkpoint(path)
    saved = load_checkpoint(path)
    assert saved["step"] == cut and saved["tokenizer_config"]["vocab_size"] == 260
    restored = build_trainer(tmp_path, seed=99)
    restored.load_checkpoint(path)
    restored.fit(6)
    assert_nested_equal(restored.model.state_dict(), expected_model)
    assert_nested_equal(restored.optimizer.state_dict(), expected_optimizer)
    assert [record["train_loss"] for record in restored.history] == expected_loss
    assert restored.train_loader.batch_sampler.state_dict() == expected_sampler
    assert torch.equal(restored.train_loader.generator.get_state(), expected_generator)
    assert torch.equal(torch.rand(3), expected_torch_random)
    assert random.random() == expected_python_random
    assert restored.tokens_processed == uninterrupted.tokens_processed


def test_checkpoint_never_overwrites_and_contains_full_optimizer_state(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.train_step()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    original = path.read_bytes()
    saved = load_checkpoint(path)
    assert saved["optimizer_state"]["state"]
    assert "exp_avg" in next(iter(saved["optimizer_state"]["state"].values()))
    assert "torch" in saved["rng"] and "python" in saved["rng"]
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == original


def test_failed_checkpoint_write_leaves_no_partial_destination(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path)
    def broken_save(payload, file):
        file.write(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", broken_save)
    path = tmp_path / "incomplete.pt"
    with pytest.raises(OSError):
        trainer.save_checkpoint(path)
    assert not path.exists()
    assert not list(tmp_path.glob(".checkpoint-*.tmp"))


def test_resume_rejects_changed_data_and_learning_rate(tmp_path):
    trainer = build_trainer(tmp_path)
    path = tmp_path / "saved.pt"
    trainer.save_checkpoint(path)
    changed = build_trainer(tmp_path, training_text="ABC" * 20)
    with pytest.raises(ValueError, match="داده"):
        changed.load_checkpoint(path)
    changed = build_trainer(tmp_path, config=replace(trainer.config, learning_rate=0.02))
    with pytest.raises(ValueError, match="تنظیمات آموزش"):
        changed.load_checkpoint(path)


def test_dataset_targets_are_passed_to_model_without_second_shift(tmp_path):
    trainer = build_trainer(tmp_path, validation=False)
    inputs, targets = trainer.train_loader.dataset[0]
    _, expected_loss = trainer.model(inputs.unsqueeze(0).repeat(2, 1), targets.unsqueeze(0).repeat(2, 1))
    record = trainer.train_step()
    assert record["train_loss"] == pytest.approx(expected_loss.item(), rel=1e-6)


def test_fixed_seed_training_is_reproducible(tmp_path):
    first = build_trainer(tmp_path, validation=False)
    first.fit(3)
    second = build_trainer(tmp_path, validation=False)
    second.fit(3)
    assert_nested_equal(first.model.state_dict(), second.model.state_dict())


def test_train_validation_document_overlap_is_rejected(tmp_path):
    trainer = build_trainer(tmp_path, validation=False)
    with pytest.raises(ValueError, match="سند مشترک"):
        Trainer(trainer.model, trainer.train_loader, trainer.train_loader, config=trainer.config)


def test_perplexity_limits():
    assert perplexity(0) == 1
    assert perplexity(math.log(2)) == pytest.approx(2)
    assert perplexity(1000) is None
    assert perplexity(float("nan")) is None
