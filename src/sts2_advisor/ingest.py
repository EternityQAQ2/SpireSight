from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = paragraph if not current else current + "\n\n" + paragraph
    if current:
        chunks.append(current)
    return chunks


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="将 Markdown/TXT 攻略转换为可检索 JSONL")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/knowledge.jsonl"))
    parser.add_argument("--title")
    parser.add_argument("--url", default="")
    parser.add_argument("--character", default="all")
    parser.add_argument("--decision-type", action="append", default=[])
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--store", choices=("json", "chroma", "both"), default="json")
    parser.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="strategy_chunks")
    args = parser.parse_args(argv)

    text = args.source.read_text(encoding="utf-8")
    title = args.title or args.source.stem
    records = []
    for index, chunk in enumerate(chunk_text(text)):
        digest = hashlib.sha256((str(args.source) + str(index) + chunk).encode("utf-8")).hexdigest()[:16]
        records.append(
            {
                "evidence_id": f"guide-{digest}",
                "text": chunk,
                "source_title": title,
                "source_url": args.url,
                "metadata": {
                    "character": args.character,
                    "decision_types": args.decision_type,
                    "license": args.license,
                    "review_status": "unreviewed",
                },
            }
        )
    if args.store in {"json", "both"}:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"已写入 {len(records)} 个 JSONL 知识块：{args.output}")

    if args.store in {"chroma", "both"}:
        try:
            import chromadb
        except ImportError as exc:
            raise SystemExit("未安装 chromadb；请先安装 rag 可选依赖") from exc
        collection = chromadb.PersistentClient(path=str(args.chroma_path)).get_or_create_collection(
            name=args.collection
        )
        collection.upsert(
            ids=[record["evidence_id"] for record in records],
            documents=[record["text"] for record in records],
            metadatas=[
                {
                    "source_title": record["source_title"],
                    "source_url": record["source_url"],
                    "character": record["metadata"]["character"],
                    "decision_type": (
                        record["metadata"]["decision_types"][0]
                        if len(record["metadata"]["decision_types"]) == 1
                        else "all"
                    ),
                    "decision_types": ",".join(record["metadata"]["decision_types"]),
                    "license": record["metadata"]["license"],
                    "review_status": record["metadata"]["review_status"],
                }
                for record in records
            ],
        )
        print(f"已写入 {len(records)} 个 Chroma 知识块：{args.chroma_path}/{args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
