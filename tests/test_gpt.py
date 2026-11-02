"""آزمون GPT، قرارداد هدف بیرونی، پارامتر مشترک و ذخیرهٔ وزن‌ها."""

from io import BytesIO

import pytest
import torch

from minicoder.config import SEED
from minicoder.model import MiniCoderGPT, ModelConfig
from minicoder.tokenizer import ByteTokenizer


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


def small_config(**overrides):
    values = dict(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=2,
                  hidden_dim=16, max_context_length=8, ignore_index=-100)
    values.update(overrides)
    return ModelConfig(**values)


def test_default_architecture_finite_forward_and_parameter_count():
    model = MiniCoderGPT()
    assert model.config.vocab_size == ByteTokenizer().vocab_size == 260
    assert (model.config.embedding_dim, model.config.num_heads, len(model.blocks),
            model.config.hidden_dim, model.config.max_context_length) == (128, 4, 4, 512, 128)
    logits, loss = model(torch.tensor([[65, 66, 67], [216, 179, 257]]))
    assert logits.shape == (2, 3, 260)
    assert torch.isfinite(logits).all() and loss is None
    assert logits.device.type == "cpu"
    assert model.lm_head.weight is not model.embeddings.token_embedding.embedding.weight
    assert model.trainable_parameter_count == 876_548


@pytest.mark.parametrize("length", [1, 4, 8])
def test_variable_length_batch_forward(length):
    model = MiniCoderGPT(small_config())
    logits, _ = model(torch.randint(0, 16, (3, length)))
    assert logits.shape == (3, length, 16)
    assert model.trainable_parameter_count == 1552


def test_next_token_loss_uses_externally_shifted_targets_without_double_shift():
    model = MiniCoderGPT(small_config())
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.copy_(torch.arange(16) * 0.1)
    sequence = torch.tensor([[2, 3, 5, 7], [4, 6, 8, 10]])
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    logits, loss = model(inputs, targets)
    bias = model.lm_head.bias
    expected = torch.logsumexp(bias, dim=0) - bias[targets].mean()
    torch.testing.assert_close(loss, expected)
    current_token_loss = torch.logsumexp(bias, dim=0) - bias[inputs].mean()
    assert not torch.isclose(loss, current_token_loss)
    # حتی یک جایگاه، یک هدف بعدی معتبر دارد و نباید درون مدل حذف شود.
    _, one_token_loss = model(torch.tensor([[2]]), torch.tensor([[3]]))
    torch.testing.assert_close(one_token_loss, torch.logsumexp(bias, dim=0) - bias[3])
    assert logits.shape == (2, 3, 16)


@pytest.mark.parametrize("ignore_index", [0, -100, 256])
def test_padding_targets_are_ignored_in_mean_loss(ignore_index):
    model = MiniCoderGPT(small_config(ignore_index=ignore_index))
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, ignore_index], [5, ignore_index, 7]], dtype=torch.int32)
    logits, loss = model(inputs, targets)
    valid = targets != ignore_index
    log_probs = logits.log_softmax(-1)
    expected = -log_probs[valid].gather(1, targets[valid].long().unsqueeze(1)).mean()
    torch.testing.assert_close(loss, expected)


def test_all_ignored_targets_have_finite_differentiable_zero_loss():
    model = MiniCoderGPT(small_config())
    _, loss = model(torch.tensor([[1, 2]]), torch.full((1, 2), -100))
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_causality_and_prefix_consistency():
    model = MiniCoderGPT(small_config()).eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = inputs.clone()
    changed[:, 3:] = torch.tensor([10, 11, 12])
    logits, _ = model(inputs)
    changed_logits, _ = model(changed)
    torch.testing.assert_close(logits[:, :3], changed_logits[:, :3], rtol=0, atol=0)
    prefix_logits, _ = model(inputs[:, :3])
    torch.testing.assert_close(logits[:, :3], prefix_logits, atol=1e-6, rtol=1e-6)


def test_gradients_reach_embeddings_all_blocks_final_norm_and_head():
    model = MiniCoderGPT(small_config())
    sequence = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    _, loss = model(sequence[:, :-1], sequence[:, 1:])
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for weight in (model.embeddings.token_embedding.embedding.weight,
                   model.embeddings.position_embedding.embedding.weight,
                   model.final_norm.weight, model.lm_head.weight,
                   *[block.attention.query.weight for block in model.blocks],
                   *[block.feedforward.linear1.weight for block in model.blocks]):
        assert weight.grad.norm() > 0


def test_optional_weight_tying_is_true_parameter_sharing():
    model = MiniCoderGPT(ModelConfig(tie_weights=True))
    assert model.lm_head.weight is model.embeddings.token_embedding.embedding.weight
    assert model.trainable_parameter_count == 843_268
    assert sum(p.numel() for p in model.parameters()) == 843_268
    logits, loss = model(torch.tensor([[65, 66]]), torch.tensor([[66, 67]]))
    loss.backward()
    assert logits.shape == (1, 2, 260)
    assert model.lm_head.weight.grad is model.embeddings.token_embedding.embedding.weight.grad


@pytest.mark.parametrize("tied", [False, True])
def test_save_load_state_dict_preserves_outputs_and_sharing(tied):
    config = small_config(tie_weights=tied)
    original = MiniCoderGPT(config).eval()
    inputs = torch.tensor([[1, 2, 3]])
    checkpoint = BytesIO()
    torch.save(original.state_dict(), checkpoint)
    checkpoint.seek(0)
    loaded = MiniCoderGPT(config).eval()
    loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert torch.equal(original(inputs)[0], loaded(inputs)[0])
    assert (loaded.lm_head.weight is loaded.embeddings.token_embedding.embedding.weight) is tied


def test_fixed_seed_reproducibility():
    inputs = torch.tensor([[1, 2, 3]])
    torch.manual_seed(SEED)
    first = MiniCoderGPT(small_config())
    torch.manual_seed(SEED)
    second = MiniCoderGPT(small_config())
    assert torch.equal(first(inputs)[0], second(inputs)[0])


@pytest.mark.parametrize("overrides", [
    {"vocab_size": 0}, {"embedding_dim": -1}, {"num_heads": 0}, {"num_heads": 3},
    {"num_layers": 0}, {"hidden_dim": 0}, {"max_context_length": 0},
    {"layer_norm_eps": 0}, {"layer_norm_eps": float("nan")}, {"embedding_dim": True},
])
def test_invalid_configurations(overrides):
    with pytest.raises(ValueError):
        small_config(**overrides)


def test_invalid_configuration_types():
    with pytest.raises(TypeError, match="tie_weights"):
        small_config(tie_weights="yes")
    with pytest.raises(TypeError, match="ignore_index"):
        small_config(ignore_index=0.5)
    with pytest.raises(TypeError, match="ModelConfig"):
        MiniCoderGPT({})


def test_invalid_inputs_context_and_targets():
    model = MiniCoderGPT(small_config())
    for inputs in (torch.ones(3, dtype=torch.long), torch.ones(1, 9, dtype=torch.long),
                   torch.empty(1, 0, dtype=torch.long), torch.tensor([[16]]), torch.tensor([[-1]])):
        with pytest.raises(ValueError):
            model(inputs)
    with pytest.raises(TypeError):
        model(torch.ones(1, 2))
    inputs = torch.tensor([[1, 2]])
    for targets in (torch.tensor([[2]]), torch.tensor([[2, 16]]), torch.tensor([[2, -1]])):
        with pytest.raises(ValueError):
            model(inputs, targets)
    with pytest.raises(TypeError):
        model(inputs, torch.ones(1, 2))
