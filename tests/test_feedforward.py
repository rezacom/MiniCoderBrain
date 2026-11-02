"""آزمون عددی لایهٔ دستی، فعال‌سازی‌ها و شبکهٔ پیش‌خور روی CPU."""

import math

import pytest
import torch
from torch.nn import functional as F

from minicoder.config import EMBEDDING_DIM, HIDDEN_DIM, SEED
from minicoder.model import CombinedEmbedding, FeedForward, GELU, ManualLinear, ReLU


@pytest.fixture(autouse=True)
def reproducible_initialization():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


@pytest.mark.parametrize("shape", [(4, 3), (2, 4, 3)])
def test_manual_linear_output_shapes(shape):
    layer = ManualLinear(3, 5)
    output = layer(torch.randn(shape))
    assert output.shape == (*shape[:-1], 5)
    assert output.device.type == "cpu"
    assert layer.trainable_parameter_count == 3 * 5 + 5
    assert layer.weight.requires_grad and layer.bias.requires_grad


def test_manual_linear_numerical_correctness():
    layer = ManualLinear(2, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 0., -1.], [2., 1., 1.]]))
        layer.bias.copy_(torch.tensor([0.5, 0., 1.]))
    inputs = torch.tensor([[1., 2.], [3., 4.]])
    expected = torch.tensor([[5.5, 2., 2.], [11.5, 4., 2.]])
    torch.testing.assert_close(layer(inputs), expected)
    torch.testing.assert_close(layer(inputs.unsqueeze(0)), expected.unsqueeze(0))


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3)])
def test_manual_linear_analytic_gradients(shape):
    layer = ManualLinear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    inputs = torch.arange(1., 7.).reshape(shape).requires_grad_()
    layer(inputs).sum().backward()
    torch.testing.assert_close(layer.weight.grad, torch.tensor([[5., 5.], [7., 7.], [9., 9.]]))
    torch.testing.assert_close(layer.bias.grad, torch.tensor([2., 2.]))
    expected_input_grad = torch.tensor([[3., 7., 11.], [3., 7., 11.]]).reshape(shape)
    torch.testing.assert_close(inputs.grad, expected_input_grad)


def test_relu_values_and_gradients_including_zero():
    inputs = torch.tensor([-2., -1., 0., 1., 3.], requires_grad=True)
    output = ReLU()(inputs)
    torch.testing.assert_close(output, torch.tensor([0., 0., 0., 1., 3.]))
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.tensor([0., 0., 0., 1., 1.]))


def test_gelu_matches_exact_pytorch_reference_and_analytic_derivative():
    inputs = torch.tensor([-4., -1., 0., 0.5, 2., 4.], dtype=torch.float64, requires_grad=True)
    output = GELU()(inputs)
    torch.testing.assert_close(output, F.gelu(inputs, approximate="none"), rtol=1e-12, atol=1e-12)
    output.sum().backward()
    expected_grad = 0.5 * (1.0 + torch.erf(inputs.detach() / math.sqrt(2.0)))
    expected_grad += inputs.detach() * torch.exp(-inputs.detach().square() / 2) / math.sqrt(2 * math.pi)
    torch.testing.assert_close(inputs.grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_nonlinearity_changes_a_stack_of_linear_layers():
    first, second = ManualLinear(1, 2), ManualLinear(2, 1)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1., -1.]]))
        first.bias.zero_()
        second.weight.fill_(1.)
        second.bias.zero_()
    inputs = torch.tensor([[-2.], [0.], [3.]])
    torch.testing.assert_close(second(first(inputs)), torch.zeros_like(inputs))
    torch.testing.assert_close(second(ReLU()(first(inputs))), torch.tensor([[2.], [0.], [3.]]))


def test_default_feedforward_shape_and_parameter_count():
    model = FeedForward()
    assert model.embedding_dim == EMBEDDING_DIM == 128
    assert model.hidden_dim == HIDDEN_DIM == 512
    assert model(torch.randn(2, 4, EMBEDDING_DIM)).shape == (2, 4, EMBEDDING_DIM)
    assert model.trainable_parameter_count == 131_712
    assert sum(parameter.numel() for parameter in model.parameters()) == 131_712
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_batch_and_sequence_positions_are_processed_independently():
    model = FeedForward(embedding_dim=3, hidden_dim=5)
    inputs = torch.randn(2, 4, 3)
    together = model(inputs)
    for batch in range(2):
        for position in range(4):
            individually = model(inputs[batch:batch + 1, position:position + 1])
            torch.testing.assert_close(together[batch, position], individually[0, 0])
    changed = inputs.clone()
    changed[1, 2] += 10
    torch.testing.assert_close(model(changed)[0], together[0])


def test_gradients_propagate_through_all_feedforward_parameters_and_inputs():
    model = FeedForward(embedding_dim=4, hidden_dim=8)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    model(inputs).square().mean().backward()
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad) > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_feedforward_gradients_reach_previous_phase_embeddings():
    embeddings = CombinedEmbedding(vocab_size=4, embedding_dim=3, max_context_length=4)
    network = FeedForward(embedding_dim=3, hidden_dim=6)
    network(embeddings(torch.tensor([[0, 1, 2]]))).square().mean().backward()
    for parameter in embeddings.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_parameter_updates_reduce_synthetic_training_loss():
    model = FeedForward(embedding_dim=4, hidden_dim=8)
    inputs = torch.randn(2, 3, 4)
    targets = torch.tanh(inputs)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    initial_loss = (model(inputs) - targets).square().mean().item()
    for _ in range(40):
        model.zero_grad()
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= 0.1 * parameter.grad
    final_loss = (model(inputs) - targets).square().mean().item()
    assert final_loss < initial_loss * 0.8
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name])


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3, 4)])
def test_manual_linear_rejects_invalid_input_dimensions(shape):
    with pytest.raises(ValueError, match="2D or 3D"):
        ManualLinear(4, 2)(torch.randn(shape))


@pytest.mark.parametrize("shape", [(2, 4), (1, 2, 3, 4)])
def test_feedforward_rejects_invalid_input_dimensions(shape):
    with pytest.raises(ValueError, match="shape"):
        FeedForward(embedding_dim=4, hidden_dim=8)(torch.randn(shape))


@pytest.mark.parametrize("factory", [lambda: ManualLinear(4, 2), lambda: FeedForward(4, 8)])
def test_feature_width_and_dtype_validation(factory):
    model = factory()
    with pytest.raises(ValueError, match="last input dimension"):
        model(torch.randn(1, 2, 5))
    with pytest.raises(TypeError, match="floating-point dtype"):
        model(torch.ones(1, 2, 4, dtype=torch.long))
    with pytest.raises(TypeError, match="torch.Tensor"):
        model([[[1., 2., 3., 4.]]])


@pytest.mark.parametrize("factory", [
    lambda: ManualLinear(0, 2),
    lambda: ManualLinear(2, -1),
    lambda: FeedForward(embedding_dim=0),
    lambda: FeedForward(hidden_dim=0),
])
def test_invalid_constructor_dimensions(factory):
    with pytest.raises(ValueError, match="positive integer"):
        factory()


def test_fixed_seed_reproduces_parameters_outputs_and_gradients():
    inputs = torch.linspace(-1., 1., 12).reshape(1, 3, 4)
    torch.manual_seed(SEED)
    first = FeedForward(4, 8)
    torch.manual_seed(SEED)
    second = FeedForward(4, 8)
    first_output, second_output = first(inputs), second(inputs)
    torch.testing.assert_close(first_output, second_output, rtol=0, atol=0)
    first_output.square().mean().backward()
    second_output.square().mean().backward()
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)
        assert torch.equal(first_parameter.grad, second_parameter.grad)
