"""آزمون نرمال‌سازی، مسیر باقیمانده، علیت و سازگاری بلوک."""

import pytest
import torch

from minicoder.config import SEED
from minicoder.model import CombinedEmbedding, TransformerBlock


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


def test_default_output_shapes_and_parameter_count():
    block = TransformerBlock()
    assert block(torch.randn(2, 5, 128)).shape == (2, 5, 128)
    assert block.trainable_parameter_count == 198_272
    assert sum(p.numel() for p in block.parameters()) == 66_048 + 131_712 + 4 * 128


def test_layernorm_normalizes_features_at_each_position():
    block = TransformerBlock(4, 2, 8, 4).double()
    inputs = torch.tensor([[[1., 2., 4., 8.], [-2., 4., 1., 5.]]], dtype=torch.float64)
    mean = inputs.mean(-1, keepdim=True)
    variance = inputs.var(-1, unbiased=False, keepdim=True)
    expected = (inputs - mean) / torch.sqrt(variance + block.norm1.eps)
    for norm in (block.norm1, block.norm2):
        torch.testing.assert_close(norm(inputs), expected)
        torch.testing.assert_close(norm(inputs).mean(-1), torch.zeros(1, 2, dtype=torch.float64), atol=1e-14, rtol=0)
    changed = inputs.clone()
    changed[:, 1] += 100
    torch.testing.assert_close(block.norm1(changed)[:, 0], block.norm1(inputs)[:, 0])


def test_forward_uses_pre_layernorm_and_two_residual_additions():
    block = TransformerBlock(8, 2, 16, 5)
    inputs = torch.randn(2, 5, 8)
    first_residual = inputs + block.attention(block.norm1(inputs))
    expected = first_residual + block.feedforward(block.norm2(first_residual))
    torch.testing.assert_close(block(inputs), expected, rtol=0, atol=0)


def test_zero_branches_preserve_input_and_identity_gradient():
    block = TransformerBlock(8, 2, 16, 5)
    with torch.no_grad():
        for module in (block.attention, block.feedforward):
            for parameter in module.parameters():
                parameter.zero_()
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    output = block(inputs)
    assert torch.equal(output, inputs)
    output.sum().backward()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))


def test_gradients_reach_both_branches_and_layernorms():
    block = TransformerBlock(8, 2, 16, 4)
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    block(inputs).square().mean().backward()
    for parameter in block.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for module in (block.norm1, block.norm2, block.attention.query, block.feedforward.linear1):
        assert module.weight.grad.norm() > 0
    assert inputs.grad.norm() > 0


def test_causal_outputs_gradients_and_batch_independence():
    block = TransformerBlock(8, 2, 16, 6)
    inputs = torch.randn(2, 6, 8, requires_grad=True)
    changed = inputs.detach().clone()
    changed[:, 3:] *= -30
    output = block(inputs)
    torch.testing.assert_close(output[:, :3], block(changed)[:, :3], rtol=0, atol=0)
    output[:, :3].sum().backward()
    assert torch.count_nonzero(inputs.grad[:, 3:]) == 0
    changed = inputs.detach().clone()
    changed[1] += 20
    torch.testing.assert_close(output[0], block(changed)[0], rtol=0, atol=0)


def test_compatibility_with_embeddings():
    embeddings = CombinedEmbedding(vocab_size=10, embedding_dim=8, max_context_length=4)
    block = TransformerBlock(8, 2, 16, 4)
    output = block(embeddings(torch.tensor([[1, 2, 3], [4, 5, 6]])))
    assert output.shape == (2, 3, 8)
    output.square().mean().backward()
    for parameter in embeddings.parameters():
        assert parameter.grad.norm() > 0


@pytest.mark.parametrize("epsilon", [0, -1, float("inf"), float("nan")])
def test_invalid_layernorm_epsilon(epsilon):
    with pytest.raises(ValueError, match="layer_norm_eps"):
        TransformerBlock(layer_norm_eps=epsilon)
