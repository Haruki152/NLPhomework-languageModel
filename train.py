"""只训练小模型：记录损失、保存检查点，以及训练前后的固定样本评估。"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from config import make_config
from count_params import count_parameters, expected_parameters
from data import TokenStream, WordTokenizer, split_text
from generate import generate_tokens
from model import DecoderLanguageModel, next_token_loss


@torch.inference_mode()
def evaluate(model, batches) -> float:
    previous_mode = model.training
    model.eval()
    losses = [next_token_loss(model(inputs), targets).item() for inputs, targets in batches]
    model.train(previous_mode)
    return sum(losses) / len(losses)


def save_json(path: Path, contents):
    path.write_text(json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="训练四层小模型，300M 模型不在此脚本中训练")
    parser.add_argument("--data", type=Path, default=Path("data/alice.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/small"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.threads) <= 0:
        parser.error("步数、批大小、评估间隔和线程数必须大于 0")
    if args.learning_rate <= 0:
        parser.error("学习率必须大于 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空。请换一个 --output，防止覆盖已有实验")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    raw_bytes = args.data.read_bytes()
    training_text, validation_text = split_text(raw_bytes.decode("utf-8"))
    # 仅用训练集建词表，验证集的新词统一映射到 <unk>。
    tokenizer = WordTokenizer.build(training_text)
    tokenizer.save(args.output / "tokenizer.json")
    config = make_config("small", len(tokenizer))
    training_ids = tokenizer.encode(training_text, add_boundaries=True)
    validation_ids = tokenizer.encode(validation_text, add_boundaries=True)
    training = TokenStream(training_ids, config.context_length)
    validation = TokenStream(validation_ids, config.context_length)

    model = DecoderLanguageModel(config).to(args.device)
    counts = count_parameters(model)
    if counts["total"] != expected_parameters(config):
        raise RuntimeError("参数统计与公式不一致")
    save_json(args.output / "config.json", config.to_dict())
    save_json(args.output / "parameters.json", counts)
    # 训练和评估使用不同随机数流。评估批次固定，避免采样变化干扰损失对比。
    train_random = torch.Generator().manual_seed(args.seed + 1)
    evaluation_random = torch.Generator().manual_seed(args.seed + 2)
    train_batches = [training.batch(args.batch_size, evaluation_random, args.device)
                     for _ in range(args.eval_batches)]
    validation_batches = [validation.batch(args.batch_size, evaluation_random, args.device)
                          for _ in range(args.eval_batches)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    settings = {
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": 0.01,
        "gradient_clip": 1.0, "seed": args.seed, "threads": args.threads,
        "device": args.device, "torch_version": str(torch.__version__),
        "data_path": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "training_tokens": len(training_ids), "validation_tokens": len(validation_ids),
        "validation_unknown_fraction": validation_ids.count(0) / len(validation_ids),
        "eval_batches": args.eval_batches, "eval_every": args.eval_every,
        "split": "按段落划分，前 90% 训练，后 10% 验证；仅训练集建词表",
    }
    save_json(args.output / "training_settings.json", settings)

    started = time.perf_counter()
    history = []
    last_batch_loss = None
    fields = ["step", "batch_loss", "train_loss", "validation_loss", "elapsed_seconds"]
    with (args.output / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps + 1):
            if step:
                model.train()
                inputs, targets = training.batch(args.batch_size, train_random, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = next_token_loss(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"第 {step} 步损失非有限值")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                last_batch_loss = loss.item()

            if step % args.eval_every == 0 or step == args.steps:
                row = {
                    "step": step, "batch_loss": last_batch_loss,
                    "train_loss": evaluate(model, train_batches),
                    "validation_loss": evaluate(model, validation_batches),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                history.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"step {step:4d} | train {row['train_loss']:.4f} | "
                      f"validation {row['validation_loss']:.4f} | {row['elapsed_seconds']:.1f}s", flush=True)

    # 检查点包含词表和配置，重新加载时无需依赖外部词表文件。
    torch.save({
        "config": config.to_dict(), "vocabulary": tokenizer.words,
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "step": args.steps, "training_settings": settings,
    }, args.output / "checkpoint.pt")
    summary = {
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_validation_loss": history[0]["validation_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "train_loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
        "validation_loss_decreased": history[-1]["validation_loss"] < history[0]["validation_loss"],
        "steps": args.steps, "parameters": counts["total"],
        "elapsed_seconds": history[-1]["elapsed_seconds"],
    }
    save_json(args.output / "summary.json", summary)
    examples = []
    for index, prompt in enumerate(["alice was", "the rabbit", "she said"]):
        ids = generate_tokens(model, tokenizer.encode(prompt), 40, seed=args.seed + index, device=args.device)
        examples.append({"prompt": prompt, "text": tokenizer.decode(ids), "seed": args.seed + index})
    save_json(args.output / "samples.json", examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
