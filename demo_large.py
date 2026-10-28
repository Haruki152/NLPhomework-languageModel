"""约 300M 模型只做实际推理，不训练，不加载小模型权重。"""

import argparse
import json
import time
from pathlib import Path

import torch

from config import make_config
from count_params import count_parameters, expected_parameters
from data import WordTokenizer
from generate import generate_tokens
from model import DecoderLanguageModel


def main():
    parser = argparse.ArgumentParser(description="随机初始化 300M 模型，执行前向计算和短文本生成")
    parser.add_argument("--tokenizer", type=Path, default=Path("runs/small/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/large_demo.json"))
    parser.add_argument("--prompt", default="alice was")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer = WordTokenizer.load(args.tokenizer)
    config = make_config("300m", len(tokenizer))
    started = time.perf_counter()
    print("正在分配并初始化真实的 300M 模型权重（float32 / CPU）...", flush=True)
    model = DecoderLanguageModel(config).eval()
    counts = count_parameters(model)
    if counts["total"] != expected_parameters(config):
        raise RuntimeError("参数统计与公式不一致")
    initialization_seconds = time.perf_counter() - started
    print(f"实际参数量: {counts['total']:,}；开始推理...", flush=True)
    prompt_ids = tokenizer.encode(args.prompt) or [tokenizer.begin_id]
    inputs = torch.tensor([prompt_ids[-config.context_length:]], dtype=torch.long)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs)
        finite = torch.isfinite(logits).all().item()
        if not finite:
            raise RuntimeError("300M 模型产生了非有限 logits")
        ids = generate_tokens(model, prompt_ids, args.new_tokens, seed=args.seed)
    result = {
        "weights": "random_initialization",
        "trained": False,
        "note": "仅展示实际 300M 模型的参数统计与推理流程。未训练权重，输出不代表语言能力。",
        "config": config.to_dict(), **counts,
        "formula_total": expected_parameters(config),
        "dtype": str(next(model.parameters()).dtype), "device": "cpu",
        "torch_version": str(torch.__version__), "seed": args.seed, "threads": args.threads,
        "weight_storage_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
        "logits_shape": list(logits.shape), "logits_all_finite": finite,
        "prompt": args.prompt, "new_tokens": args.new_tokens,
        "token_ids": ids, "text": tokenizer.decode(ids),
        "initialization_seconds": round(initialization_seconds, 3),
        "inference_seconds": round(time.perf_counter() - inference_started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
