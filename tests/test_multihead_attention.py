"""آزمون تفکیک سرها، محاسبهٔ مستقل، اتصال و تصویر خروجی."""

import math

import pytest
import torch

from minicoder.config import SEED
from minicoder.model.multihead_attention import MultiHeadCausalAttention


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


def test_head_splitting_and_merging_preserve_order():
    model = MultiHeadCausalAttention()
    inputs = torch.arange(2 * 3 * 128, dtype=torch.float32).reshape(2, 3, 128)
    heads = model.split_heads(inputs)
    assert heads.shape == (2, 4, 3, 32)
    torch.testing.assert_close(heads[1, 2, 1], inputs[1, 1, 64:96])
    assert torch.equal(model.merge_heads(heads), inputs)


@pytest.mark.parametrize("length", [1, 4, 8])
def test_output_shape_and_causal_mask_in_every_head(length):
    model = MultiHeadCausalAttention(8, 4, 8)
    output, weights = model(torch.randn(2, length, 8), return_attention=True)
    assert output.shape == (2, length, 8)
    assert weights.shape == (2, 4, length, length)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, length))
    assert torch.count_nonzero(weights.triu(1)) == 0
    assert model.head_dim == 2


def test_independent_heads_match_scalar_reference_and_output_projection():
    model = MultiHeadCausalAttention(4, 2, 3).double()
    with torch.no_grad():
        for projection in (model.query, model.key, model.value):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
        model.output_projection.weight.copy_(2 * torch.eye(4))
        model.output_projection.bias.fill_(1)
    inputs = torch.tensor([[[1., 0., 2., 1.], [0., 1., 1., -1.], [1., 1., 0., 2.]]], dtype=torch.float64)
    reference = torch.zeros_like(inputs)
    for head in range(2):
        vectors = inputs[0, :, head * 2:(head + 1) * 2]
        for position in range(3):
            scores = torch.stack([torch.dot(vectors[position], vectors[past]) / math.sqrt(2)
                                  for past in range(position + 1)])
            probability = torch.exp(scores - scores.max())
            probability /= probability.sum()
            reference[0, position, head * 2:(head + 1) * 2] = sum(
                probability[past] * vectors[past] for past in range(position + 1)
            )
    torch.testing.assert_close(model(inputs), 2 * reference + 1)


def test_future_information_and_gradients_do_not_leak():
    model = MultiHeadCausalAttention(8, 2, 5)
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    changed = inputs.detach().clone()
    changed[:, 3:] *= -20
    output = model(inputs)
    torch.testing.assert_close(output[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    output[:, :3].square().sum().backward()
    assert torch.count_nonzero(inputs.grad[:, 3:]) == 0


def test_qkv_and_output_projection_gradients_and_parameter_count():
    model = MultiHeadCausalAttention()
    model(torch.randn(2, 3, 128)).square().mean().backward()
    for projection in (model.query, model.key, model.value, model.output_projection):
        assert projection.weight.grad.norm() > 0
        assert torch.isfinite(projection.weight.grad).all()
    assert model.trainable_parameter_count == 66_048


def test_fixed_seed_is_deterministic():
    inputs = torch.randn(2, 3, 8)
    torch.manual_seed(SEED)
    first = MultiHeadCausalAttention(8, 2)
    torch.manual_seed(SEED)
    second = MultiHeadCausalAttention(8, 2)
    assert torch.equal(first(inputs), second(inputs))


@pytest.mark.parametrize("dimension,heads", [(8, 0), (8, 3), (0, 2), (8, -2), (8, True)])
def test_invalid_head_configurations(dimension, heads):
    with pytest.raises(ValueError):
        MultiHeadCausalAttention(dimension, heads)


def test_invalid_input_context_and_head_shapes():
    model = MultiHeadCausalAttention(8, 2, 4)
    for shape in [(2, 8), (1, 2, 7), (1, 5, 8)]:
        with pytest.raises(ValueError):
            model(torch.randn(shape))
    with pytest.raises(ValueError, match="heads must have shape"):
        model.merge_heads(torch.randn(1, 3, 2, 4))


def test_large_inputs_are_stable_in_every_head():
    output, weights = MultiHeadCausalAttention(8, 2, 4)(torch.randn(2, 4, 8) * 1e20, return_attention=True)
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 4))
