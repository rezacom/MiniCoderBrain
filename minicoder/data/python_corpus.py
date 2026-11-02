"""پیکرهٔ قطعی پایتون با جداسازی خانوادهٔ ساختاری و manifest صریح."""

import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
import re
from string import Template
import tokenize

from .preprocessing import reject_obvious_secrets
from .python_templates import TEMPLATES


CORPUS_VERSION = "1.1.0"
DEFAULT_SEED = 1101
MAX_EXAMPLE_BYTES = 126
SPLITS = ("train", "validation", "test")
SAFE_BUILTINS = {"abs", "all", "any", "bool", "dict", "enumerate", "int", "len", "list", "max", "min",
                 "print", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
                 "staticmethod", "property"}
SAFE_METHODS = {"append", "copy", "count", "endswith", "extend", "fromkeys", "get", "isdigit", "items",
                "join", "keys", "lower", "pop", "replace", "setdefault", "sort", "split", "startswith",
                "strip", "title", "upper", "values"}


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def parse_python_source(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("نمونهٔ پایتون خالی است.")
    reject_obvious_secrets(source)
    tree = ast.parse(source, mode="exec")
    if not tree.body or all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, str) for node in tree.body):
        raise ValueError("نمونه فقط فاصله یا توضیح دارد و برنامه نیست.")
    declared = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef)):
            raise ValueError("این پیکرهٔ مقدماتی نباید import یا دسترسی محیطی داشته باشد.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("دسترسی پویا به ویژگی‌های داخلی مجاز نیست.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS | declared or node.func.id in {"eval", "exec", "open", "compile", "__import__"}:
                    raise ValueError("فراخوانی خارج از مجموعهٔ آموزشی مجاز نیست.")
            elif not isinstance(node.func, ast.Attribute) or node.func.attr not in SAFE_METHODS:
                raise ValueError("فراخوانی متد خارج از مجموعهٔ آموزشی مجاز نیست.")
    return tree


class _Canonicalizer(ast.NodeTransformer):
    def __init__(self, tree):
        self.names, self.attributes = {}, {}
        self.bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.bound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
        self.bound.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)))

    def name(self, name):
        if name in SAFE_BUILTINS and name not in self.bound:
            return name
        return self.names.setdefault(name, f"id{len(self.names)}")

    def visit_Name(self, node):
        node.id = self.name(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.name(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = node.name if node.name in {"__init__", "__len__", "__str__"} else self.name(node.name)
        node.body = self._without_docstring(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.name = self.name(node.name)
        node.body = self._without_docstring(node.body)
        return self.generic_visit(node)

    def visit_Module(self, node):
        node.body = self._without_docstring(node.body)
        return self.generic_visit(node)

    @staticmethod
    def _without_docstring(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            return body[1:]
        return body

    def visit_Attribute(self, node):
        if node.attr not in SAFE_METHODS:
            node.attr = self.attributes.setdefault(node.attr, f"attr{len(self.attributes)}")
        return self.generic_visit(node)

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, bool):
            node.value = False
        elif isinstance(value, int):
            node.value = 0
        elif isinstance(value, float):
            node.value = 0.0
        elif isinstance(value, str):
            node.value = ""
        return node

    def visit_UnaryOp(self, node):
        if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float):
            return ast.Constant(value=0 if type(node.operand.value) is int else 0.0)
        return self.generic_visit(node)


@lru_cache(maxsize=4096)
def structural_fingerprint(source):
    tree = parse_python_source(source)
    tree = _Canonicalizer(tree).visit(tree)
    return _sha(ast.dump(tree, annotate_fields=True, include_attributes=False))


@lru_cache(maxsize=4096)
def whitespace_fingerprint(source):
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER):
            continue
        value = "<indent>" if token.type == tokenize.INDENT else token.string
        tokens.append((token.type, value))
    return _sha(json.dumps(tokens, ensure_ascii=False))


def _variants(topic, family, template, count, rng):
    numbers = rng.sample(range(1, 10), count)
    words = rng.sample(["cat", "sun", "book", "tree", "rain", "code", "lamp", "fish"], count)
    names = [("a", "b", "c"), ("x", "y", "z"), ("n", "m", "k"), ("v", "w", "u")]
    for index in range(count):
        v, u, w = names[index]
        if topic == "algorithms" and family == "fib":
            v, u, w = "n", "i", "j"
        mapping = {"v": v, "u": u, "w": w, "n": numbers[index], "m": numbers[index] % 9 + 1,
                   "c": (numbers[index] + 3) % 9 + 1, "seq": ["xs", "ns", "vs", "ys"][index],
                   "obj": ["d", "db", "kv", "dc"][index], "out": ["r", "s", "t", "q"][index],
                   "text": ["s", "txt", "msg", "word"][index], "cls": ["Box", "Cell", "Point", "Meter"][index],
                   "attr": ["x", "val", "size", "num"][index],
                   "word": repr(words[index]), "other": repr(words[(index + 1) % count]),
                   "items": repr([numbers[index], numbers[index] % 9 + 1, 0]),
                   "items2": repr([0, numbers[index], 1]),
                   "fn": family if index == 0 else family + str(index + 1)}
        yield Template(template).substitute(mapping)


def build_examples(seed=DEFAULT_SEED):
    if type(seed) is not int or seed < 0:
        raise ValueError("بذر پیکره باید عدد صحیح نامنفی باشد.")
    rng, records, indices = random.Random(seed), [], Counter()
    for topic, families in TEMPLATES.items():
        if len(families) != 12:
            raise ValueError("هر موضوع باید ۱۲ خانواده داشته باشد.")
        allocation = list(families)
        rng.shuffle(allocation)
        for index, family in enumerate(allocation):
            split = "train" if index < 8 else "validation" if index < 10 else "test"
            variants = 4 if split == "train" else 2
            for variant, source in enumerate(_variants(topic, family, families[family], variants, rng)):
                parse_python_source(source)
                if len(source.encode("utf-8")) > MAX_EXAMPLE_BYTES:
                    raise ValueError(f"نمونهٔ تولیدشده بیش از حد بلند است: {topic}/{family}/{variant}")
                indices[split] += 1
                records.append({"path": f"{split}/{indices[split]:04d}.py", "split": split,
                                "topic": topic, "family": f"{topic}/{family}", "variant": variant,
                                "utf8_bytes": len(source.encode("utf-8")), "sha256": _sha(source),
                                "whitespace_sha256": whitespace_fingerprint(source),
                                "structure_sha256": structural_fingerprint(source), "source": source})
    # اعتبارسنجی در حافظه پیش از ساخت هر فایل؛ شکست تولید پیکرهٔ ناقص نمی‌سازد.
    quality = _quality(records)
    if not quality["valid"]:
        raise ValueError("بانک الگو ساختار تکراری یا نشت دارد: " + json.dumps(quality["issues"], ensure_ascii=False))
    return records


def _quality(records):
    raw, whitespace, structures, families = (defaultdict(list) for _ in range(4))
    for record in records:
        raw[record["sha256"]].append(record)
        whitespace[record["whitespace_sha256"]].append(record)
        structures[record["structure_sha256"]].append(record)
        families[record["family"]].append(record)
    paths = lambda groups: [[record["path"] for record in group] for group in groups]
    issues = {
        "exact_duplicates": paths(group for group in raw.values() if len(group) > 1),
        "whitespace_duplicates": paths(group for group in whitespace.values() if len(group) > 1),
        "structural_leakage": paths(group for group in structures.values() if len({record["split"] for record in group}) > 1),
        "family_leakage": paths(group for group in families.values() if len({record["split"] for record in group}) > 1),
        "excessive_structural_reuse": paths(group for group in structures.values() if len(group) > 4),
    }
    return {"valid": not any(issues.values()), "issues": issues,
            "counts": {split: sum(record["split"] == split for record in records) for split in SPLITS},
            "topics": {split: dict(Counter(record["topic"] for record in records if record["split"] == split)) for split in SPLITS},
            "structural_families": len(structures), "max_structural_copies": max(map(len, structures.values()), default=0),
            "families_by_split": {split: len({record["structure_sha256"] for record in records if record["split"] == split}) for split in SPLITS},
            "min_utf8_bytes": min((record["utf8_bytes"] for record in records), default=0),
            "max_utf8_bytes": max((record["utf8_bytes"] for record in records), default=0)}


def generate_python_corpus(root, *, seed=DEFAULT_SEED):
    root = Path(root)
    if root.exists():
        raise FileExistsError("پوشهٔ پیکره موجود است؛ فقط validate کنید یا مسیر تازه بدهید.")
    examples = build_examples(seed)
    manifest = {"format_version": 1, "corpus_version": CORPUS_VERSION, "seed": seed,
                "max_example_bytes": MAX_EXAMPLE_BYTES, "origin": "locally generated educational Python",
                "split_policy": "structural template families; 8/2/2 per topic; 4/2/2 variants",
                "examples": [{key: value for key, value in record.items() if key != "source"} for record in examples]}
    root.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (root / split).mkdir()
    for record in examples:
        with (root / record["path"]).open("x", encoding="utf-8", newline="\n") as file:
            file.write(record["source"])
    with (root / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
    report = validate_python_corpus(root)
    with (root / "quality_report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
    return report


class CorpusValidationError(ValueError):
    def __init__(self, report):
        super().__init__("پیکرهٔ پایتون معتبر نیست؛ جزئیات خطا در گزارش بررسی آمده است.")
        self.report = report


def validate_python_corpus(root, *, raise_on_error=True):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("manifest محلیِ مجاز پیدا نشد.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1 or manifest.get("corpus_version") != CORPUS_VERSION:
        raise ValueError("نسخهٔ manifest پشتیبانی نمی‌شود.")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0 or manifest.get("max_example_bytes") != MAX_EXAMPLE_BYTES:
        raise ValueError("تنظیمات بذر یا سقف نمونه در manifest معتبر نیست.")
    entries = manifest.get("examples")
    if not isinstance(entries, list) or not entries or len(entries) > 5000:
        raise ValueError("فهرست مثال‌های manifest معتبر نیست.")
    problems, checked, seen_paths = [], [], set()
    reproduced = [{key: value for key, value in record.items() if key != "source"} for record in build_examples(manifest["seed"])]
    if entries != reproduced:
        problems.append({"error": "generation_manifest_mismatch"})
    for entry in entries:
        required = {"path", "split", "topic", "family", "variant", "utf8_bytes", "sha256", "whitespace_sha256", "structure_sha256"}
        if not isinstance(entry, dict) or not required.issubset(entry) or not isinstance(entry.get("topic"), str) or not isinstance(entry.get("family"), str):
            problems.append({"error": "invalid_manifest_entry"})
            continue
        relative = entry.get("path", "")
        split = entry.get("split")
        if not isinstance(relative, str) or not re.fullmatch(r"(train|validation|test)/[0-9]{4}\.py", relative) or relative.split("/")[0] != split:
            problems.append({"path": relative, "error": "invalid_split_path"})
            continue
        path = root / relative
        if relative in seen_paths or not path.resolve().is_relative_to(root) or path.is_symlink():
            problems.append({"path": relative, "error": "duplicate_or_external_path"})
            continue
        seen_paths.add(relative)
        try:
            with path.open("rb") as file:
                data = file.read(MAX_EXAMPLE_BYTES + 1)
            if len(data) > MAX_EXAMPLE_BYTES:
                problems.append({"path": relative, "error": "too_long"})
                continue
            source = data.decode("utf-8", errors="strict")
            parse_python_source(source)
            record = {**entry, "sha256": _sha(data), "utf8_bytes": len(data),
                      "whitespace_sha256": whitespace_fingerprint(source), "structure_sha256": structural_fingerprint(source)}
            for key in ("sha256", "utf8_bytes", "whitespace_sha256", "structure_sha256"):
                if record[key] != entry.get(key):
                    problems.append({"path": relative, "error": "manifest_mismatch", "field": key})
            if entry.get("topic") not in TEMPLATES or entry.get("family") not in {
                f"{entry.get('topic')}/{family}" for family in TEMPLATES.get(entry.get("topic"), {})
            }:
                problems.append({"path": relative, "error": "unknown_topic_or_family"})
            checked.append(record)
        except (OSError, UnicodeError, SyntaxError, ValueError, tokenize.TokenError):
            problems.append({"path": relative, "error": "empty_invalid_utf8_syntax_or_unsafe_source"})
    for split in SPLITS:
        directory = root / split
        if not directory.is_dir() or not directory.resolve().is_relative_to(root):
            problems.append({"path": split, "error": "missing_or_external_directory"})
            continue
        # فقط همین پوشهٔ معرفی‌شده؛ پیمایش بازگشتی یا کشف فایل خصوصی انجام نمی‌شود.
        for path in directory.iterdir():
            relative = f"{split}/{path.name}"
            if relative not in seen_paths:
                problems.append({"path": relative, "error": "unlisted_entry"})
    report = _quality(checked)
    report.update(corpus_version=CORPUS_VERSION, seed=manifest.get("seed"), file_errors=problems,
                  declared_examples=len(entries), syntax_checked=len(checked),
                  corpus_sha256=_sha(json.dumps(manifest, sort_keys=True, ensure_ascii=False)),
                  interpretation="ساختارهای AST مشترک بین بخش‌ها ممنوع‌اند؛ این بررسی اثبات هم‌ارزی معنایی یا کیفیت مدل نیست.")
    if report["counts"] != {"train": 320, "validation": 40, "test": 40}:
        problems.append({"error": "unexpected_split_counts"})
    if any(set(report["topics"][split]) != set(TEMPLATES) for split in SPLITS):
        problems.append({"error": "missing_topic_coverage"})
    if any(any(count != (32 if split == "train" else 4) for count in report["topics"][split].values()) for split in SPLITS):
        problems.append({"error": "unbalanced_topic_coverage"})
    report["valid"] = report["valid"] and not problems
    if not report["valid"] and raise_on_error:
        raise CorpusValidationError(report)
    return report


def explicit_corpus_paths(root):
    report = validate_python_corpus(root)
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    paths = {split: [root / record["path"] for record in manifest["examples"] if record["split"] == split] for split in SPLITS}
    return paths, report
