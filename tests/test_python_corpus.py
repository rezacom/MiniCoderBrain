"""پیکرهٔ پایتون: بازتولید، نحو، بودجهٔ زمینه و نشتِ مقاوم به تغییر نام."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import random

import pytest

from minicoder.data import CausalLanguageModelDataset, TextDataset
from minicoder.data.python_corpus import (
    CorpusValidationError, MAX_EXAMPLE_BYTES, build_examples, explicit_corpus_paths,
    generate_python_corpus, parse_python_source, structural_fingerprint,
    validate_python_corpus, whitespace_fingerprint,
)
from minicoder.data.python_templates import TEMPLATES
from minicoder.tokenizer import ByteTokenizer


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("python-corpus") / "corpus"
    generate_python_corpus(root)
    return root


@contextmanager
def edited(path, data):
    original = path.read_bytes()
    path.write_bytes(data)
    try:
        yield
    finally:
        path.write_bytes(original)


def test_generation_is_deterministic_and_does_not_change_global_rng():
    state = random.getstate()
    first, second = build_examples(), build_examples()
    assert first == second
    assert random.getstate() == state
    assert build_examples(seed=1102) != first
    assert Counter(row["split"] for row in first) == {"train": 320, "validation": 40, "test": 40}


def test_all_topics_and_structural_families_are_separated(corpus):
    report = validate_python_corpus(corpus)
    assert report["valid"] and report["syntax_checked"] == 400
    assert report["structural_families"] == 120
    assert report["families_by_split"] == {"train": 80, "validation": 20, "test": 20}
    assert report["max_structural_copies"] == 4
    assert not any(report["issues"].values()) and report["file_errors"] == []
    for split in ("train", "validation", "test"):
        assert set(report["topics"][split]) == set(TEMPLATES)
        assert set(report["topics"][split].values()) == ({32} if split == "train" else {4})


def test_repeated_generation_writes_identical_manifest_and_sources(corpus, tmp_path):
    other = tmp_path / "again"
    generate_python_corpus(other)
    assert (corpus / "manifest.json").read_bytes() == (other / "manifest.json").read_bytes()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["examples"]:
        assert (corpus / row["path"]).read_bytes() == (other / row["path"]).read_bytes()


def test_examples_fit_context_and_have_next_token_targets(corpus):
    paths, _ = explicit_corpus_paths(corpus)
    documents = TextDataset(paths["train"] + paths["validation"] + paths["test"], add_bos=True, add_eos=True)
    dataset = CausalLanguageModelDataset(documents, 128)
    assert len(dataset) == 400
    for index, document in enumerate(documents):
        assert len(document.text.encode("utf-8")) <= MAX_EXAMPLE_BYTES
        assert len(document.token_ids) <= 128
        inputs, targets = dataset[index]
        size = len(document.token_ids) - 1
        assert inputs.shape == targets.shape == (128,)
        assert inputs[:size].tolist() == list(document.token_ids[:-1])
        assert targets[:size].tolist() == list(document.token_ids[1:])
        assert (targets[size:] == dataset.ignore_index).all()
        assert ByteTokenizer().decode(list(document.token_ids), skip_special_tokens=True) == document.text


def test_alpha_renaming_constants_comments_and_spacing_are_not_new_heldout_structure():
    original = "def add(x):\n    return x + 1\n"
    renamed = "# another name\ndef compute(value):\n  return value + 9\n"
    shadowed_builtin = "def alter(sum):\n    return sum + -2\n"
    assert structural_fingerprint(original) == structural_fingerprint(renamed) == structural_fingerprint(shadowed_builtin)
    assert structural_fingerprint(original) != structural_fingerprint("def add(x):\n    return x * 1\n")
    assert whitespace_fingerprint(original) == whitespace_fingerprint("def add ( x ) :\n\treturn x+1 # note\n")


def test_families_are_not_shared_between_any_split():
    families = defaultdict(set)
    structures = defaultdict(set)
    for row in build_examples():
        families[row["family"]].add(row["split"])
        structures[row["structure_sha256"]].add(row["split"])
    assert all(len(splits) == 1 for splits in families.values())
    assert all(len(splits) == 1 for splits in structures.values())


def test_corpus_does_not_overwrite_existing_files(corpus):
    before = (corpus / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_python_corpus(corpus)
    assert (corpus / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("content", [b"", b"\xff", b"def broken(\n", b"x = 1\n" * 30, b"password = 'demonstration_only'\n"])
def test_empty_invalid_utf8_syntax_long_or_sensitive_examples_are_rejected(corpus, content):
    with edited(corpus / "train" / "0001.py", content):
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert not report["valid"] and report["file_errors"]
        with pytest.raises(CorpusValidationError):
            validate_python_corpus(corpus)


def test_duplicate_and_whitespace_clone_leakage_are_detected(corpus):
    training_source = (corpus / "train" / "0001.py").read_bytes()
    with edited(corpus / "validation" / "0001.py", training_source):
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert report["issues"]["exact_duplicates"]
        assert report["issues"]["whitespace_duplicates"]
        assert report["issues"]["structural_leakage"]
    with edited(corpus / "validation" / "0001.py", b"# whitespace variant\n" + training_source):
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert not report["issues"]["exact_duplicates"]
        assert report["issues"]["whitespace_duplicates"] and report["issues"]["structural_leakage"]


def test_changed_file_hash_and_unlisted_file_are_rejected(corpus):
    path = corpus / "train" / "0001.py"
    with edited(path, path.read_bytes() + b"\n"):
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert any(error["error"] == "manifest_mismatch" for error in report["file_errors"])
    extra = corpus / "train" / "unlisted.py"
    extra.write_text("x = 1\n", encoding="utf-8")
    try:
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert any(error["error"] == "unlisted_entry" for error in report["file_errors"])
    finally:
        extra.unlink()


@pytest.mark.parametrize("mutation", ["path", "split", "seed", "missing_field"])
def test_manifest_path_escape_split_and_generation_mismatch_are_rejected(corpus, mutation):
    path = corpus / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "path":
        manifest["examples"][0]["path"] = "../outside.py"
    elif mutation == "split":
        manifest["examples"][0]["split"] = "validation"
    elif mutation == "seed":
        manifest["seed"] = 1102
    else:
        del manifest["examples"][0]["family"]
    with edited(path, json.dumps(manifest).encode("utf-8")):
        report = validate_python_corpus(corpus, raise_on_error=False)
        assert not report["valid"]
        assert any(error["error"] == "generation_manifest_mismatch" for error in report["file_errors"])


@pytest.mark.parametrize("source", ["import os\nos.system('anything')", "eval('1+1')", "open('private.txt')", "# only a comment", '"only a string"'])
def test_source_validation_parses_but_does_not_allow_environmental_examples(source):
    with pytest.raises(ValueError):
        parse_python_source(source)
