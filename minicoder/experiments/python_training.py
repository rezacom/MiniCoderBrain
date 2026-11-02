"""اتصال تقسیم ثابت پیکره به همان parser و موتور train.py با --files."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

from minicoder.data.python_corpus import DEFAULT_SEED, explicit_corpus_paths
from minicoder.model import ModelConfig
from minicoder.training import TrainingConfig
from minicoder.inference import GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "python_training_v1_1.json"


def load_python_experiment_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("قالب تنظیمات آزمایش پشتیبانی نمی‌شود.")
    model = ModelConfig()
    if any(getattr(model, key, None) != value for key, value in config["expected_model"].items()):
        raise ValueError("معماری جاری با معماری ثابت آزمایش ۱٫۱ یکسان نیست.")
    if config["corpus_seed"] != DEFAULT_SEED:
        raise ValueError("این پروفایل به بذر ثابت پیکرهٔ ۱٫۱ نیاز دارد.")
    if config["sequence_length"] != 128 or config["stride"] != 128 or config["short_document_policy"] != "pad":
        raise ValueError("پروفایل پایتون از نمونهٔ کامل ۱۲۸تایی و padding راست استفاده می‌کند.")
    TrainingConfig(**config["training"])
    evaluation = config["evaluation"]
    GenerationConfig(strategy="sample", max_new_tokens=evaluation["max_new_tokens"], seed=evaluation["seed"],
                     temperature=evaluation["temperature"], top_k=evaluation["top_k"], top_p=evaluation["top_p"])
    return config


@dataclass(frozen=True)
class PythonTrainingPlan:
    root: Path
    checkpoint: Path
    training_config: TrainingConfig
    files: dict
    quality: dict
    cli_arguments: tuple[str, ...]
    expected_hashes: dict

    def split_documents(self, documents):
        allowed_train = {path.resolve() for path in self.files["train"]}
        allowed_validation = {path.resolve() for path in self.files["validation"]}
        received = {Path(document.source).resolve() for document in documents}
        if received != allowed_train | allowed_validation:
            raise ValueError("فهرست --files باید دقیقاً train و validation باشد؛ test مجاز نیست.")
        for document in documents:
            if document.content_hash != self.expected_hashes[Path(document.source).resolve()]:
                raise ValueError("محتوای فایل از زمان اعتبارسنجی تغییر کرده است.")
        train = [index for index, document in enumerate(documents) if Path(document.source).resolve() in allowed_train]
        validation = [index for index, document in enumerate(documents) if Path(document.source).resolve() in allowed_validation]
        return documents.subset(train), documents.subset(validation)

    def summary(self, *, include_files=False):
        result = {"start_from_scratch": True, "resume": None, "architecture_changes": False,
                  "existing_cli": "scripts/train.py", "files_option": "--files FILES [FILES ...]",
                  "counts": self.quality["counts"], "files_passed_to_training_cli": len(self.files["train"]) + len(self.files["validation"]),
                  "test_files_passed_to_training_cli": 0, "checkpoint": str(self.checkpoint),
                  "training": self.training_config.__dict__, "sequence_length": 128,
                  "corpus_sha256": self.quality["corpus_sha256"],
                  "note": "فهرست صریح در حافظه به parser همان train.py داده می‌شود؛ آموزش در dry-run آغاز نمی‌شود."}
        if include_files:
            result["existing_cli_arguments"] = list(self.cli_arguments)
        return result


def prepare_python_training(config_path=DEFAULT_CONFIG, *, data_root=None, steps=None, checkpoint=None):
    config = load_python_experiment_config(config_path)
    root = Path(data_root) if data_root is not None else PROJECT_ROOT / config["corpus_root"]
    files, quality = explicit_corpus_paths(root)
    if quality["seed"] != config["corpus_seed"]:
        raise ValueError("بذر پیکره با پروفایل آزمایش یکسان نیست.")
    training = TrainingConfig(**config["training"])
    if steps is not None:
        training = replace(training, max_steps=steps)
    destination = Path(checkpoint) if checkpoint is not None else PROJECT_ROOT / config["checkpoint_pattern"].format(steps=training.max_steps)
    if destination.exists():
        raise FileExistsError("checkpoint موجود بازنویسی نمی‌شود؛ مسیر تازه بدهید.")
    arguments = ["--files", *map(str, files["train"] + files["validation"]),
                 "--steps", str(training.max_steps), "--sequence-length", "128", "--stride", "128",
                 "--batch-size", str(training.batch_size), "--learning-rate", str(training.learning_rate),
                 "--weight-decay", str(training.weight_decay), "--grad-clip", str(training.max_grad_norm),
                 "--eval-interval", str(training.eval_interval), "--seed", str(training.seed),
                 "--threads", str(training.cpu_threads), "--short-documents", "pad", "--checkpoint", str(destination)]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {(root / record["path"]).resolve(): record["sha256"] for record in manifest["examples"]}
    return PythonTrainingPlan(root.resolve(), destination, training, files, quality, tuple(arguments), expected)
