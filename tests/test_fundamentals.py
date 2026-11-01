"""بررسی رفتارهای کلیدی فاز اول، همگی روی CPU."""

import importlib.util
from pathlib import Path

import pytest
import torch


LESSON_PATH = Path(__file__).resolve().parents[1] / "lessons" / "05_gradient_descent.py"
spec = importlib.util.spec_from_file_location("gradient_descent_lesson", LESSON_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


def test_tensor_shapes_and_reshape():
    scalar = torch.tensor(3)
    vector = torch.tensor([1, 2, 3])
    matrix = torch.arange(6).reshape(2, 3)
    assert scalar.ndim == 0
    assert vector.shape == (3,)
    assert matrix.shape == (2, 3)
    assert matrix.reshape(3, 2).shape == (3, 2)


def test_matrix_multiplication_and_broadcasting():
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[5.0], [6.0]])
    assert torch.equal(left @ right, torch.tensor([[17.0], [39.0]]))
    assert torch.equal(left + torch.tensor([10.0, 20.0]),
                       torch.tensor([[11.0, 22.0], [13.0, 24.0]]))


def test_gradient_and_accumulation():
    x = torch.tensor(3.0, requires_grad=True)
    (x * x + 2 * x + 1).backward()
    assert x.grad.item() == pytest.approx(8.0)
    (x * x + 2 * x + 1).backward()
    assert x.grad.item() == pytest.approx(16.0)
    x.grad.zero_()
    assert x.grad.item() == 0.0


def test_manual_parameter_update_and_gradient_reset():
    weight, bias = lesson.make_parameters()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[3.0]])
    loss = lesson.mean_squared_error(lesson.predict(x, weight, bias), y)
    loss.backward()
    old_weight = weight.detach().clone()
    old_bias = bias.detach().clone()
    weight_grad = weight.grad.detach().clone()
    bias_grad = bias.grad.detach().clone()
    lesson.update_parameters((weight, bias), learning_rate=0.1)
    assert torch.allclose(weight, old_weight - 0.1 * weight_grad)
    assert torch.allclose(bias, old_bias - 0.1 * bias_grad)
    assert torch.count_nonzero(weight.grad) == 0
    assert torch.count_nonzero(bias.grad) == 0


def test_linear_regression_converges_reproducibly_on_cpu():
    weight, bias, losses = lesson.train()
    second_weight, second_bias, second_losses = lesson.train()
    assert weight.device.type == bias.device.type == "cpu"
    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-4
    assert weight.item() == pytest.approx(2.0, abs=0.02)
    assert bias.item() == pytest.approx(1.0, abs=0.02)
    assert torch.equal(weight, second_weight)
    assert torch.equal(bias, second_bias)
    assert losses == second_losses
