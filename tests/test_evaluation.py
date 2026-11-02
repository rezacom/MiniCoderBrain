"""ارزیابی وزن‌دار، گزارش صادقانه، مقایسه و نگهداری حالت فرایند."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import random
import subprocess

import pytest
import torch
from torch.nn import functional as F

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.evaluation import Evaluator, compare_reports, model_information, training_summary
from minicoder.evaluation.benchmarks import generation_benchmarks, memorization_probe, process_memory_usage
from minicoder.evaluation.report import read_test_results, render_markdown, write_report
from minicoder.inference import GenerationConfig, GenerationResult, TextGenerator
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.training import Trainer, TrainingConfig


@pytest.fixture(autouse=True)
def fixed_environment():
    python_state, threads = random.getstate(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        random.seed(42)
        torch.set_num_threads(2)
        yield
    random.setstate(python_state)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def small_config(**changes):
    return replace(ModelConfig(embedding_dim=8, num_heads=2, num_layers=1,
                               hidden_dim=16, max_context_length=8), **changes)


def local_loader(tmp_path, text="ABCDEF" * 7, name="reference.txt", batch_size=2):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    data = CausalLanguageModelDataset(TextDataset([path]), 8)
    return create_dataloader(data, batch_size, shuffle=False, seed=42)


def short_generation():
    return GenerationConfig(max_new_tokens=2, utf8_errors="backslashreplace")


def test_default_model_info_and_tied_unique_parameter_count():
    info = Evaluator.untrained().information()
    assert info["parameters"] == info["trainable_parameters"] == 876_548
    assert (info["transformer_layers"], info["attention_heads"], info["vocab_size"], info["context_length"]) == (4, 4, 260, 128)
    assert info["parameter_bytes"] == 876_548 * 4
    assert info["training_record"]["status"] == "untrained"
    assert info["training_record"]["last_training_batch_loss"] is None
    tied = Evaluator.untrained(ModelConfig(tie_weights=True)).information()
    assert tied["parameters"] == 843_268
    assert sum(tied["component_parameters"].values()) == 843_268
    assert tied["component_parameters"]["lm_head"] == 260


def test_unknown_history_is_not_claimed_to_be_untrained_or_trained():
    info = model_information(MiniCoderGPT(small_config()))
    assert info["training_record"]["status"] == "unknown"
    assert info["training_record"]["configuration"] is None
    assert info["training_record"]["steps"] is None


def test_weighted_next_token_loss_accuracy_and_padding(tmp_path):
    evaluator = Evaluator.untrained(small_config())
    loader = local_loader(tmp_path)
    with torch.no_grad():
        evaluator.model.lm_head.weight.zero_()
        evaluator.model.lm_head.bias.copy_(torch.arange(260) * 0.001)
    result = evaluator.evaluate_loader(loader)
    inputs = torch.stack([loader.dataset[index][0] for index in range(len(loader.dataset))])
    targets = torch.stack([loader.dataset[index][1] for index in range(len(loader.dataset))])
    with torch.inference_mode():
        logits, _ = evaluator.model(inputs)
        expected = F.cross_entropy(logits.reshape(-1, 260), targets.reshape(-1), ignore_index=256)
    assert result["loss"] == pytest.approx(expected.item(), rel=1e-6)
    assert result["perplexity"] == pytest.approx(torch.exp(expected.double()).item(), rel=1e-6)
    assert result["tokens"] == len("ABCDEF" * 7) - 1
    assert result["token_accuracy"] == result["exact_sequence_accuracy"] == 0


def test_evaluation_restores_sampler_rng_training_mode_threads_and_gradients(tmp_path):
    evaluator = Evaluator.untrained(small_config())
    loader = local_loader(tmp_path)
    iterator = iter(loader)
    next(iterator)
    sampler = deepcopy(loader.batch_sampler.state_dict())
    loader_rng = loader.generator.get_state().clone()
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    before = deepcopy(evaluator.model.state_dict())
    evaluator.model.train()
    report = evaluator.run(validation_loader=loader, generation_config=short_generation())
    assert evaluator.model.training and torch.get_num_threads() == 2
    assert loader.batch_sampler.state_dict() == sampler
    assert torch.equal(loader_rng, loader.generator.get_state())
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert all(parameter.grad is None for parameter in evaluator.model.parameters())
    assert all(torch.equal(value, evaluator.model.state_dict()[key]) for key, value in before.items())
    assert report["environment"]["device"] == "cpu"


def test_report_is_reproducible_except_measured_time_and_memory(tmp_path):
    evaluator = Evaluator.untrained(small_config(), seed=8)
    loader = local_loader(tmp_path)
    generation = GenerationConfig(strategy="sample", top_k=10, seed=8, max_new_tokens=3, utf8_errors="backslashreplace")
    first = evaluator.run(validation_loader=loader, generation_config=generation)
    second = evaluator.run(validation_loader=loader, generation_config=generation)
    assert first["model"] == second["model"] and first["protocol"] == second["protocol"]
    assert first["data_evaluation"] == second["data_evaluation"]
    assert first["benchmarks"]["memorization"] == second["benchmarks"]["memorization"]
    assert [row["generated_token_ids"] for row in first["benchmarks"]["generation"]["examples"]] == [
        row["generated_token_ids"] for row in second["benchmarks"]["generation"]["examples"]]


def test_untrained_report_does_not_invent_training_results(tmp_path):
    report = Evaluator.untrained(small_config()).run(validation_loader=local_loader(tmp_path), generation_config=short_generation())
    training = report["model"]["training_record"]
    assert training["status"] == "untrained" and training["configuration"] is None
    assert training["last_training_batch_loss"] is training["last_validation_loss"] is None
    assert report["data_evaluation"]["validation"]["loss"] > 0
    assert report["tests"] is None
    assert report["benchmarks"]["memorization"]["demonstration_only"]
    assert "آموزش‌ندیده" in render_markdown(report)
    assert "تعداد آزمون حدس زده نمی‌شود" in render_markdown(report)


def test_memorization_probe_distinguishes_known_document_from_demo():
    model = MiniCoderGPT(small_config())
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[65] = 30
    engine = TextGenerator(model)
    text = "AAAAAA"
    digest = hashlib.sha256(text.encode()).hexdigest()
    demonstration = memorization_probe(engine, text)
    known = memorization_probe(engine, text, training_hashes=(digest,))
    assert demonstration["token_accuracy"] == 1 and demonstration["exact_token_match"]
    assert demonstration["demonstration_only"] and not demonstration["known_training_document"]
    assert known["known_training_document"] and not known["demonstration_only"]
    assert "teacher forcing" in known["scope"]
    with pytest.raises(ValueError):
        memorization_probe(engine, "")


def test_generation_tasks_speed_and_memory_are_labeled_demonstrations():
    evaluator = Evaluator.untrained(small_config())
    result = generation_benchmarks(evaluator.engine, short_generation())
    assert [example["task"] for example in result["examples"]] == ["english", "persian", "python"]
    assert all(example["demonstration_only"] and not example["code_executed"] for example in result["examples"])
    assert result["speed"]["duration_seconds"] > 0
    assert result["speed"]["new_tokens"] == sum(example["new_tokens"] for example in result["examples"])
    assert result["speed"]["tokens_per_second"] > 0
    memory = process_memory_usage()
    if memory["available"]:
        assert memory["peak_rss_bytes"] > 0
        if memory["rss_bytes"] is not None:
            assert memory["rss_bytes"] > 0
    else:
        assert memory["rss_bytes"] is memory["peak_rss_bytes"] is None


def test_generated_python_is_only_text_and_never_executed(monkeypatch):
    engine = Evaluator.untrained(small_config()).engine
    payload = "```python\n__import__('os').system('untrusted')\n```"
    def prohibited(*args, **kwargs):
        raise AssertionError("کد تولیدشده نباید اجرا شود")
    monkeypatch.setattr(os, "system", prohibited)
    monkeypatch.setattr(subprocess, "run", prohibited)
    monkeypatch.setattr(engine, "generate", lambda prompt, config: GenerationResult(
        prompt, prompt + payload, payload, (), tuple(payload.encode()), "max_new_tokens", True, b""))
    result = generation_benchmarks(engine, short_generation())
    assert result["examples"][2]["generated_text"] == payload
    assert result["examples"][2]["code_executed"] is False


@pytest.fixture
def checkpoints(tmp_path):
    train_loader = local_loader(tmp_path, "AB" * 20, "train.txt")
    validation_loader = local_loader(tmp_path, "CD" * 20, "validation.txt")
    trainer = Trainer(MiniCoderGPT(small_config()), train_loader, validation_loader,
                      config=TrainingConfig(max_steps=3, eval_interval=1))
    trainer.fit(1)
    first = tmp_path / "step1.pt"
    trainer.save_checkpoint(first)
    trainer.fit(3)
    second = tmp_path / "step3.pt"
    trainer.save_checkpoint(second)
    return first, second, train_loader, validation_loader


def test_checkpoint_metadata_reports_recorded_losses_without_rereading_model(checkpoints):
    first, second, _, val_loader = checkpoints
    evaluator = Evaluator.from_checkpoint(second, trusted=True)
    report = evaluator.run(validation_loader=val_loader, generation_config=short_generation())
    training = report["model"]["training_record"]
    saved_history = evaluator.metadata["history"]
    assert training["status"] == "recorded_training" and training["steps"] == 3
    assert training["last_training_batch_loss"] == saved_history[-1]["train_loss"]
    assert training["final_validation_loss"] == saved_history[-1]["validation_loss"]
    assert training["source"] == second.name
    assert report["data_evaluation"]["validation_disjoint_from_checkpoint_training"] is True
    with pytest.raises(ValueError):
        Evaluator.from_checkpoint(first)


def test_old_validation_loss_is_not_reported_as_final(checkpoints):
    _, second, _, _ = checkpoints
    metadata = deepcopy(Evaluator.from_checkpoint(second, trusted=True).metadata)
    metadata["history"][-1]["validation_loss"] = None
    record = training_summary(metadata)
    assert record["last_validation_step"] == 2
    assert record["last_validation_loss"] is not None
    assert record["final_validation_loss"] is None


@pytest.mark.parametrize("mutation", ["missing_history", "nan_loss", "wrong_order", "wrong_config"])
def test_inconsistent_training_metadata_is_rejected(checkpoints, mutation):
    metadata = deepcopy(Evaluator.from_checkpoint(checkpoints[1], trusted=True).metadata)
    if mutation == "missing_history":
        metadata["history"] = []
    elif mutation == "nan_loss":
        metadata["history"][-1]["train_loss"] = float("nan")
    elif mutation == "wrong_order":
        metadata["history"][1]["step"] = 1
    else:
        metadata["training_config"]["learning_rate"] = -1
    with pytest.raises(ValueError):
        training_summary(metadata)


def test_training_overlap_is_rejected_for_heldout_evaluation(checkpoints):
    _, checkpoint, train_loader, validation_loader = checkpoints
    evaluator = Evaluator.from_checkpoint(checkpoint, trusted=True)
    with pytest.raises(ValueError, match="داخل اسناد آموزش"):
        evaluator.run(validation_loader=train_loader, generation_config=short_generation())
    with pytest.raises(ValueError, match="سند مشترک"):
        evaluator.run(train_loader=validation_loader, validation_loader=validation_loader, generation_config=short_generation())


def test_checkpoint_comparison_uses_shared_protocol_and_actual_losses(checkpoints):
    first, second, train_loader, validation_loader = checkpoints
    reports = [Evaluator.from_checkpoint(path, trusted=True).run(
        train_loader=train_loader, validation_loader=validation_loader, generation_config=short_generation())
        for path in (first, second)]
    result = compare_reports(reports)
    assert result["comparable"]
    assert [row["validation_loss"] for row in result["ranking"]] == sorted(
        report["data_evaluation"]["validation"]["loss"] for report in reports)
    assert compare_reports([reports[0], json.loads(json.dumps(reports[1]))])["comparable"]
    altered = deepcopy(reports[1])
    altered["protocol"]["cpu_threads"] += 1
    with pytest.raises(ValueError, match="یکسان"):
        compare_reports([reports[0], altered])


def test_comparison_without_validation_does_not_invent_ranking():
    evaluator = Evaluator.untrained(small_config())
    report = evaluator.run(generation_config=short_generation())
    assert compare_reports([report, deepcopy(report)])["comparable"] is False


def test_junit_counts_and_nonoverwriting_json_markdown_reports(tmp_path):
    junit = tmp_path / "tests.xml"
    junit.write_text('<testsuites><testsuite tests="10" failures="1" errors="1" skipped="1" /></testsuites>', encoding="utf-8")
    results = read_test_results(junit)
    assert results["passed"] == 7 and results["tests"] == 10
    report = Evaluator.untrained(small_config()).run(generation_config=short_generation(), test_results=results)
    json_path, markdown_path = tmp_path / "report.json", tmp_path / "report.md"
    write_report(report, json_path=json_path, markdown_path=markdown_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["tests"] == results
    assert "7 از 10" in markdown_path.read_text(encoding="utf-8")
    original = json_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(report, json_path=json_path)
    assert json_path.read_bytes() == original
    with pytest.raises(ValueError):
        write_report(report, json_path=tmp_path / "same", markdown_path=tmp_path / "same")


@pytest.mark.parametrize("xml", ["<root/>", '<testsuite tests="1" failures="2"/>', '<testsuite tests="-1"/>'])
def test_invalid_test_result_counts_are_rejected(tmp_path, xml):
    path = tmp_path / "bad.xml"
    path.write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError):
        read_test_results(path)


def test_invalid_evaluation_dimensions_and_parameters(tmp_path):
    with pytest.raises(ValueError):
        Evaluator.untrained(small_config(), cpu_threads=0)
    evaluator = Evaluator.untrained(small_config())
    loader = local_loader(tmp_path)
    wrong_context = Evaluator.untrained(small_config(max_context_length=4))
    with pytest.raises(ValueError, match="طول نمونه"):
        wrong_context.evaluate_loader(loader)
    with pytest.raises(ValueError, match="متن آزمون"):
        evaluator.run(memorization_text="", generation_config=short_generation())
