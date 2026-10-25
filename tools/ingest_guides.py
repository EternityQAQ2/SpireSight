"""遍历 guides/ 下的攻略并入库，从属关系由目录结构与正文自动推导。

用法（项目根目录执行，无需设置 PYTHONPATH）：
  python tools/ingest_guides.py                 # 增量 upsert
  python tools/ingest_guides.py --rebuild       # 清空 Chroma 与 JSONL 后全量重入（修订攻略后使用）

从属规则：
  guides/<类别>/<X流派>/<名>.md  -> character=X（如 铁甲战士流派 -> 铁甲战士）
  其余路径                      -> character=all（避免元数据过滤误伤非流派攻略）
来源 URL 取正文中的第一个 Markdown 链接；license 按域名映射，未收录域名记为 unknown。
每次入库后把从属关系索引写入 data/manifests/guides_index.json。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_advisor.ingest import main as ingest_main  # noqa: E402


LICENSE_BY_DOMAIN = (
    ("slaythespire2.net", "slaythespire2.net，未标注"),
    ("huijiwiki.com", "CC BY-NC-SA 灰机wiki"),
)
DEFAULT_DECISION_TYPES = ("card_reward", "shop")


def source_url(text: str) -> str:
    match = re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", text)
    return match.group(1) if match else ""


def license_for(url: str) -> str:
    return next((license_ for domain, license_ in LICENSE_BY_DOMAIN if domain in url), "unknown")


def plan_guides(root: Path) -> list[dict]:
    """guides/<类别>/<组>/<名>.md -> 从属关系描述。"""
    plans: list[dict] = []
    for path in sorted(root.rglob("*.md")):
        parts = path.relative_to(root).parts
        category = parts[0] if len(parts) > 1 else ""
        group = parts[-2] if len(parts) >= 3 else ""
        build_name = path.stem
        if group.endswith("流派"):
            character = group[: -len("流派")]
            title = f"{character}：{build_name}"
        else:
            # 非流派目录（boss/事件等）不猜角色，避免元数据过滤把攻略挡掉
            character = "all"
            group = group or category
            title = build_name
        text = path.read_text(encoding="utf-8")
        url = source_url(text)
        plans.append(
            {
                "path": path,
                "category": category or "guides",
                "group": group,
                "character": character,
                "title": title,
                "url": url,
                "license": license_for(url),
            }
        )
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="遍历 guides/ 并把攻略入库（自动推导从属关系）")
    parser.add_argument("--guides-dir", type=Path, default=Path("guides"))
    parser.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--jsonl", type=Path, default=Path("data/knowledge.jsonl"))
    parser.add_argument("--store", choices=("json", "chroma", "both"), default="both")
    parser.add_argument(
        "--decision-type", action="append", default=[],
        help="覆盖默认决策类型（默认 card_reward + shop），可重复",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="先清空 Chroma 与 JSONL 再全量入库；chunk id 是内容哈希，修订后必须用本选项去陈旧副本",
    )
    args = parser.parse_args(argv)

    if args.rebuild:
        shutil.rmtree(args.chroma_path, ignore_errors=True)
        if args.jsonl.exists():
            args.jsonl.write_text("", encoding="utf-8")
        print(f"已清空 {args.chroma_path} 与 {args.jsonl}")

    decision_types = args.decision_type or list(DEFAULT_DECISION_TYPES)
    plans = plan_guides(args.guides_dir)
    if not plans:
        print(f"{args.guides_dir} 下没有找到 .md 文件")
        return 1

    failed = 0
    for item in plans:
        print(f"[入库] {item['path'].relative_to(args.guides_dir)}")
        print(f"        从属: 类别={item['category']} 组={item['group']} 角色={item['character']}")
        print(f"        来源: {item['url'] or '(未找到链接)'} | 许可: {item['license']}")
        if not item["url"]:
            print("        [警告] 正文没有来源链接，溯源信息不完整")
        argv_out = [
            str(item["path"]),
            "--store", args.store,
            "--output", str(args.jsonl),
            "--title", item["title"],
            "--url", item["url"],
            "--character", item["character"],
            *[flag for decision_type in decision_types for flag in ("--decision-type", decision_type)],
            "--chroma-path", str(args.chroma_path),
            "--license", item["license"],
        ]
        if ingest_main(argv_out) != 0:
            failed += 1

    manifest_path = Path("data/manifests/guides_index.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "guides_dir": str(args.guides_dir),
                "decision_types": decision_types,
                "entries": [
                    {key: (str(value) if isinstance(value, Path) else value) for key, value in item.items()}
                    for item in plans
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n完成：{len(plans) - failed}/{len(plans)} 个文件入库，从属索引 -> {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
