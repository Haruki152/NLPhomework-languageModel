"""单词分词、固定词表和连续文本的下词预测样本。"""

import json
import re
from collections import Counter
from pathlib import Path

import torch


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+|[^\w\s]", re.IGNORECASE)
SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>"]


def split_words(text: str) -> list[str]:
    text = text.replace("’", "'").replace("‘", "'")
    return TOKEN_PATTERN.findall(text.lower())


class WordTokenizer:
    def __init__(self, words: list[str]):
        if words[:3] != SPECIAL_TOKENS or len(words) != len(set(words)):
            raise ValueError("词表必须以三个特殊词元开头，且不能有重复项")
        self.words = words
        self.word_to_id = {word: index for index, word in enumerate(words)}
        self.unknown_id = 0
        self.begin_id = 1
        self.end_id = 2

    def __len__(self):
        return len(self.words)

    @classmethod
    def build(cls, training_text: str, max_words: int = 8192):
        if max_words < len(SPECIAL_TOKENS):
            raise ValueError("词表大小不足以容纳特殊词元")
        counts = Counter(split_words(training_text))
        # 频率相同时按词元排序，使词表构建结果可重复。
        ordered = sorted(counts, key=lambda word: (-counts[word], word))
        words = SPECIAL_TOKENS + [word for word in ordered if word not in SPECIAL_TOKENS]
        return cls(words[:max_words])

    def encode(self, text: str, add_boundaries: bool = False) -> list[int]:
        ids = [self.word_to_id.get(word, self.unknown_id) for word in split_words(text)]
        if add_boundaries:
            ids = [self.begin_id] + ids + [self.end_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        words = [self.words[index] for index in ids if index not in (self.begin_id, self.end_id)]
        text = " ".join(words)
        text = re.sub(r"\s+([,.;:!?\)\]])", r"\1", text)
        text = re.sub(r"([\(\[])\s+", r"\1", text)
        return text

    def save(self, path):
        Path(path).write_text(json.dumps({"words": self.words}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(saved["words"])


def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    """按段落切开连续文本，避免训练和验证窗口在边界处重叠。"""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction 必须在 0 和 1 之间")
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) < 2:
        raise ValueError("文本至少需要两个段落")
    boundary = max(1, min(len(paragraphs) - 1, int(len(paragraphs) * train_fraction)))
    return "\n\n".join(paragraphs[:boundary]), "\n\n".join(paragraphs[boundary:])


class TokenStream:
    def __init__(self, ids: list[int], context_length: int):
        if len(ids) <= context_length:
            raise ValueError("文本长度必须超过 context_length")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def batch(self, batch_size: int, generator: torch.Generator, device: str):
        starts = torch.randint(len(self.ids) - self.context_length, (batch_size,), generator=generator)
        indices = starts[:, None] + torch.arange(self.context_length)[None, :]
        inputs = self.ids[indices]
        # 同一窗口后移一位：输入第 t 个词，预测第 t+1 个词。
        targets = self.ids[indices + 1]
        return inputs.to(device), targets.to(device)
