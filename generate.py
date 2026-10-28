"""加载小模型检查点，并逐个词元生成文本。"""

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig
from data import WordTokenizer
from layers import stable_softmax
from model import DecoderLanguageModel


@torch.inference_mode()
def generate_tokens(model, prompt_ids: list[int], new_tokens: int, temperature: float = 0.8,
                    top_k: int = 30, seed: int = 42, device: str = "cpu") -> list[int]:
    if new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("new_tokens、temperature 和 top_k 均不能为负数")
    if not prompt_ids:
        raise ValueError("提示词元序列不能为空")
    generator = torch.Generator(device=device).manual_seed(seed)
    previous_mode = model.training
    model.eval()
    generated = list(prompt_ids)
    try:
        for _ in range(new_tokens):
            context = generated[-model.config.context_length:]
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            scores = model(inputs)[0, -1].float()
            # 不生成训练时只用作边界或占位的特殊词元。
            scores[:3] = float("-inf")
            if temperature == 0:
                next_id = scores.argmax().item()
            else:
                scores = scores / temperature
                if top_k:
                    keep = min(top_k, scores.numel() - 3)
                    if keep <= 0:
                        raise ValueError("词表中没有可生成的普通词元")
                    threshold = scores.topk(keep).values[-1]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                probabilities = stable_softmax(scores)
                next_id = torch.multinomial(probabilities, 1, generator=generator).item()
            generated.append(next_id)
    finally:
        model.train(previous_mode)
    return generated


def load_checkpoint(path: Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = DecoderLanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = WordTokenizer(checkpoint["vocabulary"])
    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="使用训练好的小模型续写")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/small/checkpoint.pt"))
    parser.add_argument("--prompt", default="alice was")
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 表示贪心生成")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, args.device)
    prompt = tokenizer.encode(args.prompt) or [tokenizer.begin_id]
    tokens = generate_tokens(model, prompt, args.new_tokens, args.temperature, args.top_k,
                             args.seed, args.device)
    result = {
        "prompt": args.prompt,
        "text": tokenizer.decode(tokens),
        "token_ids": tokens,
        "checkpoint_step": checkpoint["step"],
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "unknown_prompt_tokens": prompt.count(tokenizer.unknown_id),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
