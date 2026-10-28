"""模型尺寸集中放在这里，训练和推理共用同一份定义。"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 64
    width: int = 128
    heads: int = 4
    layers: int = 4
    hidden_width: int = 512
    norm_epsilon: float = 1e-5

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "width", "heads", "layers", "hidden_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.width % self.heads != 0:
            raise ValueError("width 必须能被 heads 整除")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon 必须大于 0")

    def to_dict(self):
        return asdict(self)


def make_config(size: str, vocab_size: int) -> ModelConfig:
    if size == "small":
        # CPU 上使用较窄的四层模型，便于实际完成训练。
        return ModelConfig(vocab_size=vocab_size)
    if size == "300m":
        # 24 层的主体约有 302M 参数，小词表下总量约为 305M。
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=128,
            width=1024,
            heads=16,
            layers=24,
            hidden_width=4096,
        )
    raise ValueError(f"未知的模型尺寸: {size}")
