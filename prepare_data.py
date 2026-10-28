"""下载并提取小说正文；原始文件保留 Gutenberg 的来源和许可说明。"""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


SOURCE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
MIRROR_URL = "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt"


def main():
    parser = argparse.ArgumentParser(description="准备 Alice 英文单词级训练文本")
    parser.add_argument("--directory", type=Path, default=Path("data"))
    parser.add_argument("--source-file", type=Path, help="可选：用已下载的同一 Gutenberg 文本代替联网")
    parser.add_argument("--source-url", default=SOURCE_URL, help="本地文件对应的网址，用于保存来源")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.source_file:
        raw = args.source_file.read_bytes()
        downloaded_from = args.source_url
    else:
        for downloaded_from in [SOURCE_URL, MIRROR_URL]:
            try:
                request = urllib.request.Request(downloaded_from, headers={"User-Agent": "NLPHomework/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
                break
            except OSError:
                if downloaded_from == MIRROR_URL:
                    raise
                print("主站下载失败，尝试同一书籍的 GITenberg 镜像。", flush=True)
    text = raw.decode("utf-8-sig")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text)
    if start is None or end is None:
        raise ValueError("没有找到 Gutenberg 正文标记，请检查下载内容")
    body = text[start.end():end.start()]
    # 跳过书名和目录，从第一章正文开始。
    chapter = re.search(r"^CHAPTER I\.", body, re.MULTILINE)
    if chapter is None:
        raise ValueError("没有找到第一章")
    body = body[chapter.start():].strip()
    if len(body) < 100_000:
        raise ValueError("正文过短，可能下载不完整")
    (args.directory / "alice_original.txt").write_bytes(raw)
    (args.directory / "alice.txt").write_text(body, encoding="utf-8")
    metadata = {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "source": SOURCE_URL,
        "downloaded_from": downloaded_from,
        "source_page": "https://www.gutenberg.org/ebooks/11",
        "sha256_original": hashlib.sha256(raw).hexdigest(),
        "body_characters": len(body),
        "processing": "去掉正文前后说明及目录；训练分词时转小写，统一弯引号。",
    }
    (args.directory / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
