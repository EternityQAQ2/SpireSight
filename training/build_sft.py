from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM = "你是《杀戮尖塔2》的只读决策建议器。只能从 candidates 中选择，不得创造动作；仅输出合法 JSON。"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_name(record: dict) -> str:
    context = record["context"]
    run = context.get("run", {})
    group = str(run.get("run_id") or run.get("seed") or context["decision_id"])
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="从审核标注构建 SFT JSONL")
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sft"))
    args = parser.parse_args()

    annotations = {
        item["decision_id"]: item
        for item in read_jsonl(args.annotations)
        if item.get("approved") is True and isinstance(item.get("advice"), dict)
    }
    output: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    for path in args.recordings:
        for record in read_jsonl(path):
            context = record.get("context", {})
            decision_id = context.get("decision_id")
            annotation = annotations.get(decision_id)
            if not annotation or decision_id in seen:
                continue
            seen.add(decision_id)
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"context": context, "evidence": record.get("evidence", [])},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(annotation["advice"], ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "metadata": {
                    "decision_id": decision_id,
                    "decision_type": context.get("decision_type"),
                    "state_type": context.get("state_type"),
                    "reviewer": annotation.get("reviewer", "unknown"),
                },
            }
            output[split_name(record)].append(sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in output.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"{split}: {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

