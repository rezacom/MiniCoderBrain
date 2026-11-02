"""اتصال --files موجود، تفکیک ثابت و معیارهای نسل پایتون، بدون آموزش تازه."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minicoder.data import TextDataset
from minicoder.data.python_corpus import generate_python_corpus
from minicoder.evaluation.python_benchmarks import (
    PYTHON_PROMPTS, assess_python_generations, python_syntax_statistics, repetition_statistics,
)
from minicoder.experiments.python_training import prepare_python_training
from minicoder.inference import GenerationResult
from minicoder.tokenizer import ByteTokenizer, EOS_ID
from scripts import train as existing_train


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("python-experiment") / "corpus"
    generate_python_corpus(root)
    return root


def test_plan_uses_existing_files_cli_and_never_passes_test_data(corpus, tmp_path):
    plan = prepare_python_training(data_root=corpus, checkpoint=tmp_path / "fresh.pt")
    args = existing_train.parser().parse_args(list(plan.cli_arguments))
    assert len(args.files) == 360
    assert set(args.files) == set(plan.files["train"] + plan.files["validation"])
    assert set(args.files).isdisjoint(plan.files["test"])
    assert args.resume is None and args.steps == 5
    assert args.embedding_dim is args.num_heads is args.num_layers is args.hidden_dim is None
    assert args.sequence_length == args.stride == 128
    assert args.seed == 42 and args.batch_size == 2
    documents = TextDataset(args.files, add_bos=True, add_eos=True)
    train, validation = plan.split_documents(documents)
    assert len(train) == 320 and len(validation) == 40
    assert {Path(doc.source) for doc in train} == set(plan.files["train"])
    assert {Path(doc.source) for doc in validation} == set(plan.files["validation"])
    assert plan.summary()["test_files_passed_to_training_cli"] == 0
    assert plan.summary()["start_from_scratch"]


def test_splitter_rejects_test_files_even_if_they_are_valid_python(corpus, tmp_path):
    plan = prepare_python_training(data_root=corpus, checkpoint=tmp_path / "new.pt")
    documents = TextDataset(plan.files["train"] + plan.files["validation"] + plan.files["test"])
    with pytest.raises(ValueError, match="test"):
        plan.split_documents(documents)


def test_fixed_split_hook_reaches_training_boundary_without_starting_fit(corpus, tmp_path, monkeypatch):
    plan = prepare_python_training(data_root=corpus, checkpoint=tmp_path / "new.pt")
    args = existing_train.parser().parse_args(list(plan.cli_arguments))
    class ReachedTraining(Exception):
        pass
    class BoundaryTrainer:
        def __init__(self, model, train_loader, validation_loader, *, config):
            self.model = model
            assert len(train_loader.dataset) == 320
            assert len(validation_loader.dataset) == 40
            assert {Path(doc.source) for doc in train_loader.dataset.documents} == set(plan.files["train"])
            assert {Path(doc.source) for doc in validation_loader.dataset.documents} == set(plan.files["validation"])
        def fit(self, **kwargs):
            raise ReachedTraining()
    monkeypatch.setattr(existing_train, "Trainer", BoundaryTrainer)
    monkeypatch.setattr(existing_train, "MiniCoderGPT", lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(existing_train, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(existing_train, "estimate_training_memory_mib", lambda *args: 0.0)
    with pytest.raises(ReachedTraining):
        existing_train.run(args, tmp_path, document_splitter=plan.split_documents)
    assert not plan.checkpoint.exists()


def test_longer_plan_is_explicit_and_existing_checkpoints_are_preserved(corpus, tmp_path):
    destination = tmp_path / "long.pt"
    plan = prepare_python_training(data_root=corpus, steps=1000, checkpoint=destination)
    assert plan.training_config.max_steps == 1000
    assert "--resume" not in plan.cli_arguments
    destination.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError):
        prepare_python_training(data_root=corpus, checkpoint=destination)
    assert destination.read_bytes() == b"existing checkpoint"


def test_repetition_counts_bytes_and_ignores_eos():
    result = repetition_statistics([65, 65, 65, 65, EOS_ID])
    assert result["byte_tokens"] == 4
    assert result["repeated_token_fraction"] == 0.75
    assert result["repeated_trigram_fraction"] == 0.5
    assert result["longest_identical_run"] == 4
    assert repetition_statistics([])["repeated_token_fraction"] == 0


def test_syntax_check_does_not_execute_programs(monkeypatch):
    import os
    monkeypatch.setattr(os, "system", lambda *args: pytest.fail("نباید اجرا شود"))
    assert python_syntax_statistics("import os\nos.system('never run')\n")["valid"]
    assert python_syntax_statistics("def add(a, b):")["valid"] is False
    assert python_syntax_statistics("\\xff", utf8_valid=False)["checked"] is False


def test_fixed_prompts_both_strategies_and_reproducible_seeds():
    continuations = ["\n    return a + b\n", "\n    return x * x\n", "3):\n    print(i)\n",
                     "\n    print(x)\n", "\n    return text[::-1]\n"]
    class Engine:
        calls = []
        def generate(self, prompt, config):
            self.calls.append((prompt, config))
            text = continuations[PYTHON_PROMPTS.index(prompt)]
            return GenerationResult(prompt, prompt + text, text,
                                    tuple(ByteTokenizer().encode(prompt, add_bos=True)),
                                    tuple(text.encode()) + (EOS_ID,), "eos", True, b"")
    engine = Engine()
    result = assess_python_generations(engine, ["def add(a, b):\n    return a + b\n"], seed=42)
    assert len(result["examples"]) == len(engine.calls) == 10
    assert [prompt for prompt, _ in engine.calls[:5]] == list(PYTHON_PROMPTS)
    assert [config.seed for _, config in engine.calls[:5]] == list(range(42, 47))
    assert [config.seed for _, config in engine.calls[5:]] == list(range(42, 47))
    assert all(row["python_syntax"]["valid"] and not row["code_executed"] for row in result["examples"])
    assert result["strategy_summary"]["greedy"]["syntax_valid"] == 5
    assert result["examples"][0]["prompt_exposure"]["configured_training_examples_containing_fragment"] == 1
    assert all(config.top_k is None for _, config in engine.calls[:5])
    assert all(config.top_k == 40 and config.top_p == 0.9 for _, config in engine.calls[5:])
