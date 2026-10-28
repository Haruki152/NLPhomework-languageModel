"""用 Parameter 和张量运算实现核心层，不使用现成的注意力或归一化层。"""

import math

import torch
from torch import nn


class LinearMap(nn.Module):
    """最后一维的线性变换，权重形状为 [输入维度, 输出维度]。"""

    def __init__(self, input_width: int, output_width: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_width, output_width))
        self.bias = nn.Parameter(torch.zeros(output_width))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weight + self.bias


class EmbeddingTable(nn.Module):
    def __init__(self, entries: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(entries, width))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]


class LayerNormalization(nn.Module):
    """每个词元独立地沿特征维归一化，方差除以 d，而不是 d-1。"""

    def __init__(self, width: int, epsilon: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.offset = nn.Parameter(torch.zeros(width))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # 半精度推理时也用 float32 累加，避免方差计算精度不足。
        working = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
        mean = working.mean(dim=-1, keepdim=True)
        variance = (working - mean).square().mean(dim=-1, keepdim=True)
        normalized = (working - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized.to(values.dtype) * self.scale + self.offset


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    """沿最后一维计算 softmax，先减最大值以防 exp 溢出。"""
    working = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    shifted = working - working.amax(dim=-1, keepdim=True)
    exponentials = shifted.exp()
    return (exponentials / exponentials.sum(dim=-1, keepdim=True)).to(scores.dtype)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        # 分开保留 Q/K/V，便于对应公式和检查每个投影的作用。
        self.query = LinearMap(config.width, config.width)
        self.key = LinearMap(config.width, config.width)
        self.value = LinearMap(config.width, config.width)
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        self.output = LinearMap(config.width, config.width, residual_std)

    def split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, width = values.shape
        # [B, T, D] -> [B, H, T, D/H]
        return values.reshape(batch_size, length, self.heads, self.head_width).transpose(1, 2)

    def forward(self, values: torch.Tensor, return_weights: bool = False):
        batch_size, length, width = values.shape
        queries = self.split_heads(self.query(values))
        keys = self.split_heads(self.key(values))
        contents = self.split_heads(self.value(values))

        # 每个位置只能看到自身和之前的位置；True 对应需要遮住的未来位置。
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_width)
        positions = torch.arange(length, device=values.device)
        future = positions[None, :] > positions[:, None]
        scores = scores.masked_fill(future, float("-inf"))
        weights = stable_softmax(scores)
        attended = weights @ contents

        # [B, H, T, D/H] -> [B, T, D]
        combined = attended.transpose(1, 2).contiguous().reshape(batch_size, length, width)
        result = self.output(combined)
        if return_weights:
            return result, weights
        return result


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = LinearMap(config.width, config.hidden_width)
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        self.contract = LinearMap(config.hidden_width, config.width, residual_std)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(values)
        # GELU 的常用 tanh 近似式，直接按公式计算。
        curved = math.sqrt(2 / math.pi) * (expanded + 0.044715 * expanded.pow(3))
        activated = 0.5 * expanded * (1 + torch.tanh(curved))
        return self.contract(activated)
