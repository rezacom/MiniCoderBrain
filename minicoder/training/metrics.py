"""معیارهای عددی، تکرارپذیری و برآورد تقریبی حافظهٔ CPU."""

import math
import random
import sys

import torch

from minicoder.config import SEED


def seed_everything(seed=SEED, *, cpu_threads=2):
    if type(seed) is not int or seed < 0 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("بذر نامنفی و تعداد رشتهٔ مثبت لازم است.")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def perplexity(loss):
    if loss is None or not math.isfinite(loss) or loss < 0 or loss > math.log(sys.float_info.max):
        return None
    return math.exp(loss)


def estimate_training_memory_mib(model, batch_size, sequence_length):
    parameters = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    config = model.config
    # وزن، گرادیان و دو حالت AdamW؛ همراه چند ماتریس توجه و فعال‌سازی.
    attention = batch_size * config.num_heads * sequence_length ** 2 * 8 * config.num_layers * 4
    activations = batch_size * sequence_length * (config.hidden_dim + config.embedding_dim) * 4 * config.num_layers * 8
    return (4 * parameters + attention + activations) / 1024 ** 2
