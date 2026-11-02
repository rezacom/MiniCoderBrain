"""آزمون دادهٔ محلی، هدف T+1، تفکیک سند و نمونه‌گیری تکرارپذیر."""

import pytest
import torch

from minicoder.data import CausalLanguageModelDataset, TextDataset, create_dataloader, split_documents, write_synthetic_corpus
from minicoder.tokenizer import PAD_ID, ByteTokenizer


def local_document(tmp_path, text, name="sample.txt"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("text", ["سلام دنیا 😀", "Hello World!", "def f():\r\n    return 2\r\n", " \t\n", "Python و پایتون"])
def test_utf8_tokenization_preserves_exact_text(tmp_path, text):
    dataset = TextDataset([local_document(tmp_path, text)])
    document = dataset[0]
    assert document.text == text
    assert list(document.token_ids) == ByteTokenizer().encode(text)
    assert ByteTokenizer().decode(list(document.token_ids)) == text


def test_128_inputs_require_129_tokens_and_correct_shift(tmp_path):
    text = "A" + "BC" * 64
    dataset = CausalLanguageModelDataset(TextDataset([local_document(tmp_path, text)]), 128, short_document_policy="drop")
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (128,)
    assert torch.equal(inputs, torch.tensor(list(text[:-1].encode())))
    assert torch.equal(targets, torch.tensor(list(text[1:].encode())))
    assert inputs[0] != targets[0]
    assert len(dataset) == 1


@pytest.mark.parametrize("stride,starts", [(1, list(range(9))), (3, [0, 3, 6]), (4, [0, 4, 8])])
def test_sliding_and_nonoverlapping_windows(tmp_path, stride, starts):
    text = "ABCDEFGHIJKL"
    dataset = CausalLanguageModelDataset(TextDataset([local_document(tmp_path, text)]), 3,
                                         stride=stride, short_document_policy="drop")
    assert len(dataset) == len(starts)
    for index, start in enumerate(starts):
        inputs, targets = dataset[index]
        assert inputs.tolist() == list(text[start:start + 3].encode())
        assert targets.tolist() == list(text[start + 1:start + 4].encode())


def test_short_document_padding_and_ignore_index(tmp_path):
    dataset = CausalLanguageModelDataset(TextDataset([local_document(tmp_path, "ABC")]), 5, ignore_index=-100)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [65, 66, PAD_ID, PAD_ID, PAD_ID]
    assert targets.tolist() == [66, 67, -100, -100, -100]
    with pytest.raises(ValueError, match="هیچ پنجره"):
        CausalLanguageModelDataset(dataset.documents, 5, short_document_policy="drop")


def test_empty_files_empty_lists_and_insufficient_tokens(tmp_path):
    with pytest.raises(ValueError, match="خالی"):
        TextDataset([local_document(tmp_path, "")])
    with pytest.raises(ValueError, match="خالی"):
        TextDataset([])
    with pytest.raises(ValueError, match="هیچ پنجره"):
        CausalLanguageModelDataset(TextDataset([local_document(tmp_path, "A")]), 4)


def test_only_explicit_files_are_read_and_invalid_utf8_is_rejected(tmp_path):
    bad = tmp_path / "not_configured.txt"
    bad.write_bytes(b"\xff\xfe")
    allowed = local_document(tmp_path, "Only this document.")
    assert len(TextDataset([allowed])) == 1
    with pytest.raises(UnicodeDecodeError):
        TextDataset([bad])
    with pytest.raises(ValueError):
        TextDataset([tmp_path])
    with pytest.raises(TypeError):
        TextDataset(str(allowed))


def test_secret_guard_and_size_limit_do_not_expose_contents(tmp_path):
    suspicious = "api_key = '" + "x" * 24 + "'"
    with pytest.raises(ValueError) as error:
        TextDataset([local_document(tmp_path, suspicious)])
    assert "x" * 24 not in str(error.value)
    with pytest.raises(ValueError, match="اندازه"):
        TextDataset([local_document(tmp_path, "abcdef")], max_file_bytes=3)


@pytest.mark.parametrize("label", ["password", "secret", "client_secret"])
def test_common_secret_assignments_are_rejected(tmp_path, label):
    with pytest.raises(ValueError, match="محرمانه"):
        TextDataset([local_document(tmp_path, label + " = '" + "demonstration" + "'")])


def test_splits_are_deterministic_and_group_duplicate_documents(tmp_path):
    paths = [local_document(tmp_path, text, f"doc_{index}.txt")
             for index, text in enumerate(["ABC" * 10, "DEF" * 10, "GHI" * 10, "ABC" * 10])]
    source = TextDataset(paths)
    train, validation = split_documents(source, 0.5, seed=12)
    second_train, second_validation = split_documents(source, 0.5, seed=12)
    assert [doc.source for doc in train] == [doc.source for doc in second_train]
    assert [doc.source for doc in validation] == [doc.source for doc in second_validation]
    assert {doc.content_hash for doc in train}.isdisjoint(doc.content_hash for doc in validation)
    assert {doc.source for doc in train}.isdisjoint(doc.source for doc in validation)
    assert len(train) + len(validation) == len(source)
    train_windows = CausalLanguageModelDataset(train, 8)
    validation_windows = CausalLanguageModelDataset(validation, 8)
    assert train_windows.fingerprint != validation_windows.fingerprint


def test_one_document_cannot_be_split_into_leaking_windows(tmp_path):
    with pytest.raises(ValueError, match="دو سند"):
        split_documents(TextDataset([local_document(tmp_path, "AB" * 20)]))


def test_batch_shapes_and_shuffle_reproducibility(tmp_path):
    documents = TextDataset([local_document(tmp_path, "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4)])
    windows = CausalLanguageModelDataset(documents, 8, stride=3)
    first = list(create_dataloader(windows, batch_size=3, seed=7))
    second = list(create_dataloader(windows, batch_size=3, seed=7))
    assert first[0][0].shape == first[0][1].shape == (3, 8)
    for (inputs, targets), (other_inputs, other_targets) in zip(first, second):
        assert torch.equal(inputs, other_inputs) and torch.equal(targets, other_targets)


def test_sampler_continues_at_saved_batch(tmp_path):
    windows = CausalLanguageModelDataset(TextDataset([local_document(tmp_path, "ABCDEFGHIJKLMNOP" * 8)]), 4)
    loader = create_dataloader(windows, 2, seed=11)
    iterator = iter(loader)
    next(iterator)
    state = loader.batch_sampler.state_dict()
    expected = next(iterator)
    restored = create_dataloader(windows, 2, seed=11)
    restored.batch_sampler.load_state_dict(state)
    actual = next(iter(restored))
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])


def test_synthetic_corpus_is_local_and_never_overwrites(tmp_path):
    paths = write_synthetic_corpus(tmp_path / "synthetic")
    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    dataset = TextDataset(paths)
    train, validation = split_documents(dataset)
    assert len(train) > 0 and len(validation) > 0
    with pytest.raises(FileExistsError):
        write_synthetic_corpus(tmp_path / "synthetic")


@pytest.mark.parametrize("options", [{"sequence_length": 0}, {"stride": 0}, {"short_document_policy": "unknown"}])
def test_invalid_window_settings(tmp_path, options):
    with pytest.raises(ValueError):
        CausalLanguageModelDataset(TextDataset([local_document(tmp_path, "ABC")]), **options)
