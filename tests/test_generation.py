"""تولید، فیلتر احتمال، UTF-8 افزایشی، زمینه و checkpoint قابل اعتماد."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader
from minicoder.inference import (
    GenerationConfig, GenerationDecodingError, IncrementalUTF8Decoder, TextGenerator,
    filtered_logits, load_generator_checkpoint, sample_token,
)
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteTokenizer
from minicoder.training import Trainer, TrainingConfig
from minicoder.training.checkpoint import load_checkpoint


@pytest.fixture(autouse=True)
def fixed_environment():
    previous_threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        yield
    torch.set_num_threads(previous_threads)


class ScriptedGPT(nn.Module):
    """مدل کنترل‌شده برای آزمون مستقلِ انتخاب توکن و رمزگشایی."""

    def __init__(self, token_ids, context_length=4):
        super().__init__()
        self.config = ModelConfig(embedding_dim=8, num_heads=2, num_layers=1,
                                  hidden_dim=16, max_context_length=context_length)
        self.anchor = nn.Parameter(torch.zeros(()))
        self.script = token_ids
        self.inputs = []
        self.execution_modes = []

    def forward(self, inputs):
        index = len(self.inputs)
        self.inputs.append(inputs.clone())
        self.execution_modes.append((self.training, torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        logits = torch.full((1, inputs.shape[1], self.config.vocab_size), -1000., device="cpu")
        next_id = self.script[index] if index < len(self.script) else EOS_ID
        logits[:, -1, next_id] = 1000.
        return logits, None


def test_greedy_selection_ties_and_forbidden_tokens():
    logits = torch.tensor([1., 3., 3., 0.])
    original = logits.clone()
    assert sample_token(logits) == 1
    assert sample_token(logits, forbidden_ids=(1,)) == 2
    assert torch.equal(logits, original)


def test_temperature_changes_probabilities_and_is_stable():
    logits = torch.tensor([2., 1., 0.])
    cold = filtered_logits(logits, temperature=0.5).softmax(-1)
    hot = filtered_logits(logits, temperature=2.0).softmax(-1)
    assert cold[0] > hot[0]
    assert -(hot * hot.log()).sum() > -(cold * cold.log()).sum()
    extreme = filtered_logits(torch.tensor([1e30, 0., -1e30]), temperature=1e-300).softmax(-1)
    assert torch.isfinite(extreme).all()
    assert extreme.tolist() == [1., 0., 0.]


def test_top_k_keeps_exact_number_and_stable_tie_order():
    logits = torch.tensor([0., 3., 2., 1.])
    assert torch.isfinite(filtered_logits(logits, top_k=2)).tolist() == [False, True, True, False]
    assert torch.isfinite(filtered_logits(torch.zeros(4), top_k=2)).tolist() == [True, True, False, False]
    rng = torch.Generator().manual_seed(42)
    samples = [sample_token(logits, strategy="sample", top_k=2, generator=rng) for _ in range(50)]
    assert set(samples) == {1, 2}


def test_top_p_keeps_smallest_prefix_including_threshold_crossing():
    logits = torch.tensor([0.4, 0.35, 0.15, 0.1], dtype=torch.float64).log()
    assert torch.isfinite(filtered_logits(logits, top_p=0.5)).tolist() == [True, True, False, False]
    assert torch.isfinite(filtered_logits(logits, top_p=0.01)).tolist() == [True, False, False, False]
    assert torch.isfinite(filtered_logits(logits, top_p=1.0)).all()
    assert torch.isfinite(filtered_logits(torch.zeros(4), top_p=0.5)).sum() == 2
    assert torch.isfinite(filtered_logits(logits, top_k=2, top_p=0.5)).sum() == 1


def test_greedy_generation_stops_at_eos_and_preserves_prompt():
    model = ScriptedGPT([65, 66, EOS_ID, 67])
    result = TextGenerator(model).generate("سلام", GenerationConfig(max_new_tokens=10))
    assert result.text == "سلامAB" and result.generated_text == "AB"
    assert result.generated_token_ids == (65, 66, EOS_ID)
    assert result.prompt_token_ids == tuple(ByteTokenizer().encode("سلام", add_bos=True))
    assert result.finish_reason == "eos" and len(model.inputs) == 3
    assert result.utf8_valid


def test_limits_zero_limit_and_empty_prompt():
    model = ScriptedGPT([65, 66, 67])
    result = TextGenerator(model).generate("", GenerationConfig(max_new_tokens=2))
    assert result.text == "AB" and result.finish_reason == "max_new_tokens"
    assert result.generated_token_ids == (65, 66)
    assert model.inputs[0].tolist() == [[BOS_ID]]
    model = ScriptedGPT([65])
    result = TextGenerator(model).generate("سلام", GenerationConfig(max_new_tokens=0))
    assert result.text == "سلام" and result.generated_token_ids == ()
    assert model.inputs == []
    assert TextGenerator(ScriptedGPT([EOS_ID])).generate("").text == ""


def test_context_truncates_only_model_input_and_keeps_original_persian_prompt():
    prompt = "سلام دنیا"
    model = ScriptedGPT([65, 66, 67], context_length=3)
    result = TextGenerator(model).generate(prompt, GenerationConfig(max_new_tokens=3))
    encoded = ByteTokenizer().encode(prompt, add_bos=True)
    assert model.inputs[0].tolist() == [encoded[-3:]]
    assert model.inputs[1].tolist() == [[*encoded[-2:], 65]]
    assert all(inputs.shape == (1, 3) for inputs in model.inputs)
    assert result.text == prompt + "ABC"
    assert result.token_ids == tuple(encoded + [65, 66, 67])


def test_inference_mode_eval_cpu_and_stream_do_not_leak_global_grad_mode():
    model = ScriptedGPT([65, 66, EOS_ID])
    model.train()
    engine = TextGenerator(model)
    stream = engine.stream("")
    first = next(stream)
    assert first.text == "A"
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    stream.close()
    assert model.training
    assert model.execution_modes == [(False, True, False)]
    assert model.inputs[0].device.type == "cpu" and model.anchor.grad is None


def test_sampling_is_reproducible_without_changing_global_rng():
    model = MiniCoderGPT(ModelConfig(embedding_dim=8, num_heads=2, num_layers=1, hidden_dim=16, max_context_length=4))
    engine = TextGenerator(model)
    config = GenerationConfig(strategy="sample", temperature=0.8, top_k=12, top_p=0.9,
                              seed=7, max_new_tokens=12, utf8_errors="backslashreplace")
    before = torch.get_rng_state().clone()
    first, second = engine.generate("Hi", config), engine.generate("Hi", config)
    assert first == second
    assert torch.equal(before, torch.get_rng_state())
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_reserved_nontext_tokens_are_suppressed_but_eos_is_allowed():
    logits = torch.full((260,), -10.)
    logits[PAD_ID], logits[BOS_ID], logits[UNK_ID], logits[EOS_ID] = 100., 99., 98., 10.
    assert sample_token(logits, forbidden_ids=(PAD_ID, BOS_ID, UNK_ID)) == EOS_ID


def test_incremental_decoder_waits_for_complete_persian_and_emoji():
    decoder = IncrementalUTF8Decoder()
    assert decoder.feed(216) == ""
    assert decoder.pending_bytes == b"\xd8"
    assert decoder.feed(179) == "س"
    emoji_bytes = list("😀".encode())
    assert [decoder.feed(byte) for byte in emoji_bytes] == ["", "", "", "😀"]
    assert decoder.finish() == "" and not decoder.had_errors


def test_streamed_persian_and_emoji_are_exact_and_match_nonstream_output():
    generated = "سلام 😀"
    script = list(generated.encode()) + [EOS_ID]
    events = list(TextGenerator(ScriptedGPT(script)).stream("پرامپت: "))
    result = events[-1].result
    assert "".join(event.text for event in events) == generated
    assert result.text == "پرامپت: " + generated
    assert result.utf8_valid and result.incomplete_bytes == b""
    assert events[0].text == "" and events[-1].token_id == EOS_ID
    assert result == TextGenerator(ScriptedGPT(script)).generate("پرامپت: ")


@pytest.mark.parametrize("script,limit", [([216], 1), ([216, EOS_ID], 2)])
def test_incomplete_utf8_is_explicitly_rejected_or_escaped(script, limit):
    config = GenerationConfig(max_new_tokens=limit)
    model = ScriptedGPT(script)
    with pytest.raises(GenerationDecodingError, match="ناقص") as error:
        TextGenerator(model).generate("سلام", config)
    assert error.value.generated_token_ids == tuple(script)
    assert error.value.partial_text == "سلام" and model.training
    result = TextGenerator(ScriptedGPT(script)).generate("سلام", replace(config, utf8_errors="backslashreplace"))
    assert result.text == "سلام\\xd8"
    assert not result.utf8_valid and result.incomplete_bytes == b"\xd8"


def test_invalid_bytes_never_silently_replace_or_drop_persian():
    with pytest.raises(GenerationDecodingError, match="نامعتبر"):
        TextGenerator(ScriptedGPT([255])).generate("", GenerationConfig(max_new_tokens=1))
    ids = [255, *"سلام".encode(), EOS_ID]
    result = TextGenerator(ScriptedGPT(ids)).generate("", GenerationConfig(utf8_errors="backslashreplace"))
    assert result.text == "\\xffسلام"
    assert not result.utf8_valid and result.incomplete_bytes == b""
    decoder = IncrementalUTF8Decoder("backslashreplace")
    actual = "".join(decoder.feed(byte) for byte in [0xE2, 0xE3, 0x80, 0x81]) + decoder.finish()
    assert actual == "\\xe2、"


def test_literal_escape_text_and_real_replacement_character_are_valid_unicode():
    text = "\\xff �"
    result = TextGenerator(ScriptedGPT([*text.encode(), EOS_ID])).generate("")
    assert result.text == text and result.utf8_valid


@pytest.mark.parametrize("options", [
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"temperature": 0},
    {"strategy": "sample", "temperature": -1}, {"strategy": "sample", "temperature": float("nan")},
    {"strategy": "sample", "temperature": float("inf")}, {"strategy": "sample", "top_k": 0},
    {"strategy": "sample", "top_k": True}, {"strategy": "sample", "top_p": 0},
    {"strategy": "sample", "top_p": 1.1}, {"seed": -1}, {"seed": 2 ** 63},
    {"utf8_errors": "ignore"}, {"utf8_errors": "replace"}, {"strategy": "unknown"},
    {"strategy": "greedy", "top_k": 2},
])
def test_invalid_generation_parameters(options):
    with pytest.raises(ValueError):
        GenerationConfig(**options)


def test_invalid_logits_and_sampling_candidates():
    for logits in (torch.zeros(2, 3), torch.tensor([]), torch.tensor([1, 2]),
                   torch.tensor([float("nan"), 0.]), torch.tensor([float("inf"), 0.]),
                   torch.full((3,), float("-inf"))):
        with pytest.raises(ValueError):
            sample_token(logits)
    with pytest.raises(ValueError):
        sample_token(torch.zeros(3), strategy="sample", top_k=4)
    with pytest.raises(ValueError):
        sample_token(torch.zeros(3), forbidden_ids=(0, 1, 2))
    with pytest.raises(ValueError):
        TextGenerator(ScriptedGPT([65])).generate("", GenerationConfig(strategy="sample", top_k=261))


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "synthetic.txt"
    path.write_text("AB" * 32, encoding="utf-8")
    dataset = CausalLanguageModelDataset(TextDataset([path]), 4)
    model = MiniCoderGPT(ModelConfig(embedding_dim=8, num_heads=2, num_layers=1, hidden_dim=16, max_context_length=4))
    trainer = Trainer(model, create_dataloader(dataset, 2), config=TrainingConfig(max_steps=1))
    trainer.train_step()
    destination = tmp_path / "trusted-local.pt"
    trainer.save_checkpoint(destination)
    return destination, trainer.model


def test_trusted_local_checkpoint_loading_matches_original_model(checkpoint):
    path, original = checkpoint
    rng_before = torch.get_rng_state().clone()
    loaded = load_generator_checkpoint(path, trusted=True)
    assert not loaded.model.training
    assert torch.equal(rng_before, torch.get_rng_state())
    for name, weight in original.state_dict().items():
        assert torch.equal(weight, loaded.model.state_dict()[name])
    config = GenerationConfig(max_new_tokens=8, utf8_errors="backslashreplace")
    assert loaded.generate("سلام", config) == TextGenerator(original).generate("سلام", config)


def test_untrusted_missing_remote_and_incomplete_checkpoints_are_rejected(tmp_path, checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="trusted=True"):
        load_generator_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        load_generator_checkpoint(tmp_path / "missing.pt", trusted=True)
    with pytest.raises(ValueError, match="محلی"):
        load_generator_checkpoint("https://example.invalid/model.pt", trusted=True)
    incomplete = tmp_path / "only-weights.pt"
    torch.save({"model_state": {}}, incomplete)
    with pytest.raises(ValueError, match="قالب"):
        load_generator_checkpoint(incomplete, trusted=True)
    invalid = tmp_path / "broken.pt"
    invalid.write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError):
        load_generator_checkpoint(invalid, trusted=True)


@pytest.mark.parametrize("mutation", ["config", "vocabulary", "tokenizer", "shape", "missing", "nan", "dtype", "version"])
def test_incompatible_checkpoints_are_rejected(checkpoint, tmp_path, mutation):
    path, _ = checkpoint
    saved = load_checkpoint(path)
    if mutation == "config":
        saved["model_config"]["num_heads"] = 3
    elif mutation == "vocabulary":
        saved["model_config"]["vocab_size"] = 259
    elif mutation == "tokenizer":
        saved["tokenizer_config"]["special_tokens"]["EOS"] = 0
    elif mutation == "shape":
        saved["model_state"]["lm_head.weight"] = torch.zeros(1, 1)
    elif mutation == "missing":
        saved["model_state"].pop("lm_head.bias")
    elif mutation == "nan":
        saved["model_state"]["lm_head.bias"][0] = float("nan")
    elif mutation == "dtype":
        saved["model_state"]["lm_head.bias"] = saved["model_state"]["lm_head.bias"].double()
    elif mutation == "version":
        saved["format_version"] = 999
    destination = tmp_path / "incompatible.pt"
    torch.save(saved, destination)
    with pytest.raises(ValueError):
        load_generator_checkpoint(destination, trusted=True)


def test_tied_checkpoint_is_preserved_and_conflicting_aliases_are_rejected(checkpoint, tmp_path):
    path, _ = checkpoint
    saved = load_checkpoint(path)
    saved["model_config"]["tie_weights"] = True
    saved["model_state"]["lm_head.weight"] = saved["model_state"]["embeddings.token_embedding.embedding.weight"].clone()
    tied_path = tmp_path / "tied.pt"
    torch.save(saved, tied_path)
    loaded = load_generator_checkpoint(tied_path, trusted=True)
    assert loaded.model.lm_head.weight is loaded.model.embeddings.token_embedding.embedding.weight
    saved["model_state"]["lm_head.weight"] += 1
    conflict = tmp_path / "conflict.pt"
    torch.save(saved, conflict)
    with pytest.raises(ValueError, match="مشترک"):
        load_generator_checkpoint(conflict, trusted=True)
