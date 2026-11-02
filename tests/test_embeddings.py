"""بررسی جدول‌های قابل یادگیری، اعتبار ورودی و مسیر گرادیان روی CPU."""

import pytest
import torch

from minicoder.config import EMBEDDING_DIM, MAX_CONTEXT_LENGTH, SEED
from minicoder.model import CombinedEmbedding, PositionEmbedding, TokenEmbedding
from minicoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ByteTokenizer


@pytest.fixture(autouse=True)
def reproducible_initialization():
    # بذر این آزمون‌ها روی وضعیت تصادفی آزمون‌های فازهای قبلی اثر نمی‌گذارد.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


def test_default_configuration_shapes_and_parameter_counts():
    tokenizer = ByteTokenizer()
    tokens = TokenEmbedding()
    positions = PositionEmbedding()
    combined = CombinedEmbedding()
    input_ids = torch.tensor([tokenizer.encode("سلام"), tokenizer.encode("درود")])
    assert EMBEDDING_DIM == MAX_CONTEXT_LENGTH == 128
    assert tokens.vocab_size == combined.vocab_size == tokenizer.vocab_size == 260
    assert tokens.embedding_dim == positions.embedding_dim == combined.embedding_dim == EMBEDDING_DIM
    assert positions.max_context_length == combined.max_context_length == MAX_CONTEXT_LENGTH
    assert tokens.embedding.weight.shape == (tokenizer.vocab_size, EMBEDDING_DIM)
    assert positions.embedding.weight.shape == (MAX_CONTEXT_LENGTH, EMBEDDING_DIM)
    assert tokens.trainable_parameter_count == tokenizer.vocab_size * EMBEDDING_DIM
    assert positions.trainable_parameter_count == MAX_CONTEXT_LENGTH * EMBEDDING_DIM
    assert combined.trainable_parameter_count == (tokenizer.vocab_size + MAX_CONTEXT_LENGTH) * EMBEDDING_DIM
    for module in (tokens, positions, combined):
        output = module(input_ids)
        assert output.shape == (*input_ids.shape, EMBEDDING_DIM)
        assert output.device.type == "cpu"
        assert all(parameter.requires_grad for parameter in module.parameters())


def test_token_lookup_and_batch_processing():
    module = TokenEmbedding(vocab_size=4, embedding_dim=2)
    with torch.no_grad():
        module.embedding.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]))
    output = module(torch.tensor([[3, 0], [1, 3]]))
    expected = torch.tensor([[[7., 8.], [1., 2.]], [[3., 4.], [7., 8.]]])
    torch.testing.assert_close(output, expected)


def test_position_indices_and_lookup_are_independent_of_token_contents():
    module = PositionEmbedding(max_context_length=3, embedding_dim=2)
    input_ids = torch.tensor([[65, 66, 67], [90, 91, 92]])
    assert torch.equal(module.position_ids(input_ids), torch.tensor([[0, 1, 2], [0, 1, 2]]))
    with torch.no_grad():
        module.embedding.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    expected = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]], [[1., 2.], [3., 4.], [5., 6.]]])
    torch.testing.assert_close(module(input_ids), expected)


def test_positions_have_independently_initialized_vectors():
    # مقداردهی با بذر ثابت انجام می‌شود؛ مقایسه بر تصادف آزاد تکیه ندارد.
    module = PositionEmbedding(max_context_length=3, embedding_dim=4)
    original = module.embedding.weight.detach().clone()
    assert not torch.allclose(original[0], original[1])
    with torch.no_grad():
        module.embedding.weight[0].add_(1.0)
    assert torch.equal(module.embedding.weight[1], original[1])


def test_combined_embedding_is_sum_of_token_and_position_vectors():
    module = CombinedEmbedding(vocab_size=5, embedding_dim=3, max_context_length=4)
    input_ids = torch.tensor([[1, 2, 1], [4, 3, 0]])
    output = module(input_ids)
    assert output.shape == (2, 3, 3)
    torch.testing.assert_close(output, module.token_embedding(input_ids) + module.position_embedding(input_ids))


def test_gradients_accumulate_in_used_rows_of_both_tables():
    module = CombinedEmbedding(vocab_size=4, embedding_dim=2, max_context_length=4)
    input_ids = torch.tensor([[1, 1, 2], [2, 1, 0]])
    module(input_ids).sum().backward()
    # مشتق جمع نسبت به هر مؤلفهٔ خروجی ۱ است؛ تکرار توکن‌ها جمع می‌شود.
    expected_token_grad = torch.tensor([[1., 1.], [3., 3.], [2., 2.], [0., 0.]])
    expected_position_grad = torch.tensor([[2., 2.], [2., 2.], [2., 2.], [0., 0.]])
    torch.testing.assert_close(module.token_embedding.embedding.weight.grad, expected_token_grad)
    torch.testing.assert_close(module.position_embedding.embedding.weight.grad, expected_position_grad)
    assert not input_ids.requires_grad


def test_manual_update_changes_used_rows_and_reduces_synthetic_loss():
    module = CombinedEmbedding(vocab_size=4, embedding_dim=2, max_context_length=4)
    input_ids = torch.tensor([[1, 2]])
    token_before = module.token_embedding.embedding.weight.detach().clone()
    position_before = module.position_embedding.embedding.weight.detach().clone()
    loss = module(input_ids).square().mean()
    loss.backward()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter -= 0.1 * parameter.grad
    module.zero_grad()
    assert module(input_ids).square().mean().item() < loss.item()
    assert not torch.equal(module.token_embedding.embedding.weight[1], token_before[1])
    assert not torch.equal(module.position_embedding.embedding.weight[0], position_before[0])
    assert torch.equal(module.token_embedding.embedding.weight[3], token_before[3])
    assert torch.equal(module.position_embedding.embedding.weight[3], position_before[3])


@pytest.mark.parametrize("module_class", [TokenEmbedding, CombinedEmbedding])
@pytest.mark.parametrize("bad_id", [-1, ByteTokenizer().vocab_size])
def test_invalid_token_ids(module_class, bad_id):
    with pytest.raises(ValueError, match="token IDs"):
        module_class()(torch.tensor([[0, bad_id]]))


def test_special_tokens_are_valid_embedding_indices():
    module = TokenEmbedding()
    assert module(torch.tensor([[0, PAD_ID, BOS_ID, EOS_ID, UNK_ID]])).shape == (1, 5, EMBEDDING_DIM)


@pytest.mark.parametrize("module_class", [PositionEmbedding, CombinedEmbedding])
def test_context_length_boundary_and_overflow(module_class):
    module = module_class(embedding_dim=2, max_context_length=3)
    assert module(torch.zeros((2, 3), dtype=torch.long)).shape == (2, 3, 2)
    with pytest.raises(ValueError, match="max_context_length"):
        module(torch.zeros((2, 4), dtype=torch.long))


@pytest.mark.parametrize("module_class", [TokenEmbedding, PositionEmbedding, CombinedEmbedding])
def test_input_shape_and_dtype_validation(module_class):
    module = module_class()
    for shape in [(3,), (1, 2, 3)]:
        with pytest.raises(ValueError, match="shape"):
            module(torch.zeros(shape, dtype=torch.long))
    for dtype in (torch.float32, torch.bool):
        with pytest.raises(TypeError, match="int32 or torch.int64"):
            module(torch.zeros((1, 2), dtype=dtype))
    with pytest.raises(TypeError, match="torch.Tensor"):
        module([[1, 2]])


@pytest.mark.parametrize("module_class", [TokenEmbedding, PositionEmbedding, CombinedEmbedding])
@pytest.mark.parametrize("shape", [(2, 0), (0, 3)])
def test_empty_inputs_preserve_batch_and_sequence_axes(module_class, shape):
    module = module_class()
    assert module(torch.empty(shape, dtype=torch.long)).shape == (*shape, EMBEDDING_DIM)


def test_int32_token_ids_are_supported():
    assert CombinedEmbedding()(torch.tensor([[0, 1]], dtype=torch.int32)).shape == (1, 2, EMBEDDING_DIM)


@pytest.mark.parametrize("factory", [
    lambda: TokenEmbedding(vocab_size=0),
    lambda: TokenEmbedding(embedding_dim=-1),
    lambda: PositionEmbedding(max_context_length=0),
    lambda: PositionEmbedding(embedding_dim=0),
])
def test_invalid_table_dimensions(factory):
    with pytest.raises(ValueError, match="positive integer"):
        factory()
