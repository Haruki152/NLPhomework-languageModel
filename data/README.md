# 数据说明

训练文本为 Lewis Carroll 的 *Alice's Adventures in Wonderland*。

- [Project Gutenberg 书籍页](https://www.gutenberg.org/ebooks/11)
- [本次实际下载的 GITenberg 镜像](https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt)

| 文件 | 内容 |
| --- | --- |
| `alice_original.txt` | 下载的完整原文，包含书籍信息和 Project Gutenberg 许可说明 |
| `alice.txt` | 去掉正文外说明后的文本，用于训练 |
| `source.json` | 书名、来源、实际下载地址和原文 SHA-256 |

原文附带的许可说明保留在 `alice_original.txt` 中。`alice.txt` 来自同一份原文，处理步骤见根目录的 `prepare_data.py`。

原文 SHA-256：

```text
49a0b2726606e1290ac03a63978fa1dd1bd38a8d805704d98265f393533ea094
```

训练时统一英文大小写与弯引号，再按单词和标点分词。词表只使用训练集构建；按段落划分训练与验证部分后，分别取连续窗口，不跨越两部分的边界。

本次词表为 2510 项，训练与验证分别有 31,305 和 3,061 个词元，数量包含边界标记。验证集未知词元占比约为 5.42%。
