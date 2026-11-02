"""آزمون توجه تک‌سر با نمونهٔ دستی، علیت و محاسبهٔ پایدار."""

import math

import pytest
import torch

from minicoder.config import SEED
from minicoder.model.attention import CausalSelfAttention


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


@pytest.mark.parametrize("shape", [(2, 5, 128), (1, 1, 128), (3, 9, 128)])
def test_shapes_normalization_and_future_mask(shape):
    model = CausalSelfAttention()
    output, weights = model(torch.randn(shape), return_attention=True)
    assert output.shape == shape
    assert weights.shape == (shape[0], shape[1], shape[1])
    torch.testing.assert_close(weights.sum(-1), torch.ones(shape[:2]))
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    assert torch.isfinite(output).all()
    assert model.trainable_parameter_count == 3 * (128 * 128 + 128)


def test_hand_calculated_scores_and_value_aggregation():
    model = CausalSelfAttention(embedding_dim=2, max_context_length=3).double()
    with torch.no_grad():
        for projection in (model.query, model.key, model.value):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    inputs = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
    output, weights = model(inputs, return_attention=True)
    exponential = math.exp(1 / math.sqrt(2))
    p = 1 / (1 + exponential)
    r = 1 / (2 + exponential)
    expected_weights = torch.tensor([[[1., 0., 0.], [p, 1-p, 0.], [r, r, 1-2*r]]], dtype=torch.float64)
    expected_output = torch.tensor([[[1., 0.], [p, 1-p], [1-r, 1-r]]], dtype=torch.float64)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_future_changes_and_future_gradients_do_not_leak():
    model = CausalSelfAttention(4, 6)
    inputs = torch.randn(2, 6, 4, requires_grad=True)
    changed = inputs.detach().clone()
    changed[:, 3:] += 10
    output = model(inputs)
    torch.testing.assert_close(output[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    output[:, :3].square().sum().backward()
    assert torch.count_nonzero(inputs.grad[:, 3:]) == 0


def test_qkv_projections_receive_gradients():
    model = CausalSelfAttention(4, 5)
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    model(inputs).square().mean().backward()
    for projection in (model.query, model.key, model.value):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.norm() > 0
        assert projection.bias.grad is not None
    assert inputs.grad.norm() > 0


def test_large_finite_inputs_are_stable():
    model = CausalSelfAttention(4, 4)
    output, weights = model(torch.randn(2, 4, 4) * 1e20, return_attention=True)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 5), (1, 5, 4), (1, 0, 4), (0, 2, 4)])
def test_invalid_dimensions_and_context(shape):
    with pytest.raises(ValueError):
        CausalSelfAttention(4, 4)(torch.randn(shape))


def test_invalid_dtype_and_nonfinite_values():
    model = CausalSelfAttention(4, 4)
    with pytest.raises(TypeError, match="dtype"):
        model(torch.ones(1, 2, 4, dtype=torch.long))
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            model(torch.full((1, 2, 4), value))
    with pytest.raises(TypeError, match="Tensor"):
        model([[[1.0] * 4]])


def test_unrepresentable_scores_raise_instead_of_producing_nan():
    with pytest.raises(ValueError, match="overflowed"):
        CausalSelfAttention(4, 4).double()(torch.full((1, 2, 4), 1e200, dtype=torch.float64))
