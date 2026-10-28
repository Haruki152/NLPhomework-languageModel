# 已记录的实验结果

此目录保存报告对应的那次实验，供阅读结果和直接加载小模型使用。新运行的输出写入根目录的 `runs/`，不会覆盖这里。

## 小模型

| 文件 | 内容 |
| --- | --- |
| `small/checkpoint.pt` | 第 400 步模型权重、配置、词表和训练设置 |
| `small/config.json` | 4 层、128 维、4 头的模型配置 |
| `small/tokenizer.json` | 与检查点对应的 2510 项词表 |
| `small/training_settings.json` | 训练设置、数据摘要及原实验路径 |
| `small/parameters.json` | 各模块参数量，共 1,122,816 |
| `small/loss.csv` | 每 50 步记录的固定批次损失 |
| `small/loss.png`、`small/loss.svg` | 根据同一份 CSV 绘制的曲线 |
| `small/summary.json` | 训练前后损失与步数摘要 |
| `small/samples.json` | 训练后生成的三组样例，种子为 2026、2027、2028 |
| `small/reloaded_sample.json` | 重新加载检查点后生成的样例，种子为 42 |

检查点内包含词表，不要替换成其他数据重新建立的词表。训练输出没有保存优化器状态，附带检查点用于推理及检查权重加载。

## 大模型及检查记录

| 文件 | 内容 |
| --- | --- |
| `parameter_counts.json` | 约 300M 模型的分组统计与公式核对结果 |
| `large_demo.json` | 随机初始化大模型的实际推理记录 |
| `checks.json` | 10 项核心检查的原始结果 |

大模型共 305,012,736 个参数，使用 CPU / float32。`large_demo.json` 中的 `trained` 为 `false`：没有训练大模型。记录中的生成文本只用于展示真实模型能完成前向计算和生成。

耗时来自原实验的一次运行，会受到硬件、线程数和系统负载影响。完整实验分析见[实验报告](../docs/experiment_report.pdf)。
