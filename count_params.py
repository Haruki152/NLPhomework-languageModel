"""按实际张量统计参数，并用公式独立核对；不分配大模型权重。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from config import ModelConfig, make_config
from data import WordTokenizer
from model import DecoderLanguageModel


def count_parameters(model) -> dict:
    groups = defaultdict(int)
    total = 0
    trainable = 0
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        parts = name.split(".")
        group_name = ".".join(parts[:2]) if parts[0] == "decoder" else parts[0]
        groups[group_name] += count
    return {"total": total, "trainable": trainable, "groups": dict(groups)}


def expected_parameters(config: ModelConfig) -> int:
    width = config.width
    hidden = config.hidden_width
    # Q/K/V/输出映射: 4D²+4D；前馈: 2DF+F+D；两个归一化: 4D。
    per_layer = 4 * width * width + 2 * width * hidden + 9 * width + hidden
    embedding = (config.vocab_size + config.context_length) * width
    return embedding + config.layers * per_layer + 2 * width


def main():
    parser = argparse.ArgumentParser(description="统计 small 或 300m 配置的参数量")
    parser.add_argument("--size", choices=["small", "300m"], default="300m")
    parser.add_argument("--tokenizer", type=Path, default=Path("runs/small/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, help="没有词表文件时可显式指定")
    parser.add_argument("--output", type=Path, default=Path("runs/parameter_counts.json"))
    args = parser.parse_args()
    vocab_size = args.vocab_size or len(WordTokenizer.load(args.tokenizer))
    config = make_config(args.size, vocab_size)

    # meta 设备只有形状，没有数值存储；此处仅统计，真实推理另见 demo_large.py。
    with torch.device("meta"):
        model = DecoderLanguageModel(config)
    counts = count_parameters(model)
    expected = expected_parameters(config)
    if counts["total"] != expected:
        raise RuntimeError(f"参数公式不一致: {counts['total']} != {expected}")
    result = {"size": args.size, "config": config.to_dict(), **counts, "formula_total": expected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
