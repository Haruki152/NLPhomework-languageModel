"""解码器语言模型：词元嵌入 -> 多层解码器 -> 词表分数。"""

import torch
from torch import nn

from config import ModelConfig
from layers import CausalAttention, EmbeddingTable, FeedForward, LayerNormalization


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = LayerNormalization(config.width, config.norm_epsilon)
        self.attention = CausalAttention(config)
        self.feedforward_norm = LayerNormalization(config.width, config.norm_epsilon)
        self.feedforward = FeedForward(config)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states + self.attention(self.attention_norm(states))
        states = states + self.feedforward(self.feedforward_norm(states))
        return states


class DecoderLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.words = EmbeddingTable(config.vocab_size, config.width)
        self.positions = EmbeddingTable(config.context_length, config.width)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])
        self.final_norm = LayerNormalization(config.width, config.norm_epsilon)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("输入必须是 [batch_size, sequence_length]")
        length = token_ids.shape[1]
        if not 1 <= length <= self.config.context_length:
            raise ValueError(f"序列长度必须在 1 到 {self.config.context_length} 之间")

        position_ids = torch.arange(length, device=token_ids.device)
        states = self.words(token_ids) + self.positions(position_ids)
        for layer in self.decoder:
            states = layer(states)
        states = self.final_norm(states)

        # 输入嵌入和输出映射共享一张词表，统计参数时只算一次。
        # 输出是 [B, T, V] 的 logits，不是注意力权重。
        return states @ self.words.weight.transpose(0, 1)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """标签已在 data.py 中右移；这里手写数值稳定的交叉熵。"""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits 的前两维必须与 targets 一致")
    scores = logits.float()
    log_normalizer = torch.logsumexp(scores, dim=-1)
    correct_scores = scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - correct_scores).mean()
