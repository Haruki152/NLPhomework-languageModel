"""执行关键数值检查。参考接口只用于核对，不参与模型实现。"""

import io
import json
import time
from pathlib import Path

import torch

from config import ModelConfig, make_config
from count_params import count_parameters, expected_parameters
from data import TokenStream, WordTokenizer
from generate import generate_tokens
from layers import CausalAttention, LayerNormalization, stable_softmax
from model import DecoderLanguageModel, next_token_loss


def tiny_config():
    return ModelConfig(vocab_size=13, context_length=8, width=16, heads=4, layers=2, hidden_width=32)


def check_softmax():
    scores = torch.tensor([[1000., 999., float("-inf")], [-1000., -999., -1001.]])
    actual = stable_softmax(scores)
    torch.testing.assert_close(actual, torch.softmax(scores, dim=-1))
    assert actual[0, 2].item() == 0
    torch.testing.assert_close(actual.sum(-1), torch.ones(2))


def check_normalization():
    layer = LayerNormalization(5).double()
    values = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.layer_norm(values, (5,), layer.scale, layer.offset, layer.epsilon)
    torch.testing.assert_close(layer(values), expected)
    assert torch.autograd.gradcheck(layer, (values,))
    assert torch.isfinite(layer(torch.ones(2, 3, 5, dtype=torch.float64))).all()


def check_attention():
    attention = CausalAttention(tiny_config())
    values = torch.randn(2, 6, 16)
    output, weights = attention(values, return_weights=True)
    assert output.shape == (2, 6, 16)
    assert weights.shape == (2, 4, 6, 6)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 6))
    modified = values.clone()
    modified[:, 3:] += torch.randn_like(modified[:, 3:]) * 10
    torch.testing.assert_close(output[:, :3], attention(modified)[:, :3])


def check_model_causality():
    model = DecoderLanguageModel(tiny_config()).eval()
    inputs = torch.tensor([[3, 4, 5, 6, 7, 8]])
    changed = inputs.clone()
    changed[:, 3:] = torch.tensor([9, 10, 11])
    with torch.inference_mode():
        original = model(inputs)
        altered = model(changed)
    assert original.shape == (1, 6, 13)
    torch.testing.assert_close(original[:, :3], altered[:, :3])


def check_label_shift():
    stream = TokenStream(list(range(30)), context_length=6)
    inputs, targets = stream.batch(3, torch.Generator().manual_seed(1), "cpu")
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
    torch.testing.assert_close(targets, inputs + 1)


def check_loss_and_gradients():
    model = DecoderLanguageModel(tiny_config())
    inputs = torch.randint(0, 13, (2, 6))
    targets = torch.randint(0, 13, (2, 6))
    logits = model(inputs)
    loss = next_token_loss(logits, targets)
    reference = torch.nn.functional.cross_entropy(logits.reshape(-1, 13), targets.reshape(-1))
    torch.testing.assert_close(loss, reference)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def check_parameter_formula():
    for size in ("small", "300m"):
        config = make_config(size, 2500)
        with torch.device("meta"):
            model = DecoderLanguageModel(config)
        actual = count_parameters(model)["total"]
        assert actual == expected_parameters(config)
        if size == "300m":
            assert 290_000_000 <= actual <= 320_000_000


def check_checkpoint_and_generation():
    model = DecoderLanguageModel(tiny_config()).eval()
    inputs = torch.tensor([[3, 4, 5]])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DecoderLanguageModel(tiny_config()).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    first = generate_tokens(model, [3, 4], 10, seed=7)
    second = generate_tokens(restored, [3, 4], 10, seed=7)
    assert first == second
    assert len(first) == 12  # 超过上下文长度，验证滑动窗口仍能继续生成。
    assert all(3 <= token < 13 for token in first[2:])


def check_tokenizer():
    tokenizer = WordTokenizer.build("Alice said: 'hello'. Alice smiled.")
    assert tokenizer.encode("ALICE") == tokenizer.encode("alice")
    assert tokenizer.encode("unseenword") == [tokenizer.unknown_id]
    assert tokenizer.decode(tokenizer.encode("Alice smiled.")) == "alice smiled."
    assert tokenizer.encode("alice’s") == tokenizer.encode("alice's")


def check_tiny_overfit():
    model = DecoderLanguageModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]]).repeat(2, 1)
    targets = torch.tensor([[4, 5, 6, 3, 4, 5, 6, 3]]).repeat(2, 1)
    initial = next_token_loss(model(inputs), targets).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final = next_token_loss(model(inputs), targets).item()
    assert final < initial * 0.3, (initial, final)
    return {"initial_loss": initial, "final_loss": final}


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    checks = [check_softmax, check_normalization, check_attention, check_model_causality,
              check_label_shift, check_loss_and_gradients, check_parameter_formula,
              check_checkpoint_and_generation, check_tokenizer, check_tiny_overfit]
    results = []
    for check in checks:
        started = time.perf_counter()
        details = check()
        result = {"check": check.__name__, "passed": True,
                  "seconds": round(time.perf_counter() - started, 3)}
        if details is not None:
            result["details"] = details
        results.append(result)
        print(f"PASS {check.__name__}", flush=True)
    output = Path("runs/checks.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(results)} 项检查全部通过。")


if __name__ == "__main__":
    main()
