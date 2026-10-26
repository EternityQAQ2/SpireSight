"""遍历 guides/ 下的攻略并入库，从属关系由目录结构与正文自动推导。

用法（项目根目录执行，无需设置 PYTHONPATH）：
  python tools/ingest_guides.py                 # 增量 upsert
  python tools/ingest_guides.py --dry-run       # 只打印入库计划，不写库
  python tools/ingest_guides.py --rebuild       # 清空 Chroma 与 JSONL 后全量重入（修订攻略后使用）

从属规则：
  guides/<类别>/<X流派>/<名>.md  -> character=X（如 铁甲战士流派 -> 铁甲战士）
  其余路径                      -> character=all（避免元数据过滤误伤非流派攻略）

决策类型（决定攻略会被注入哪种决策）按三级优先级解析：
  1. 命令行 --decision-type（全局覆盖，测试用）
  2. DECISION_TYPES_BY_FILE 单文件覆盖（相对 guides/ 的 posix 路径）
  3. DECISION_TYPES_BY_CATEGORY 目录类别默认值

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
    ("spirebuilds.com", "spirebuilds.com，未标注"),
)
DEFAULT_DECISION_TYPES = ("card_reward", "shop")
# 目录类别决定默认投放的决策界面。
#   builds：构筑攻略，只在选牌和商店有用。
#   bosses：目前放的是“怪物机制”，只在战斗决策（combat_next_action）里有意义。
#           如果以后放回“Boss 备战”这类战略攻略，用下面的单文件覆盖指回
#           map/rest_site/shop/card_reward，而不是改这里。
DECISION_TYPES_BY_CATEGORY = {
    "builds": ("card_reward", "shop"),
    "bosses": ("combat_next_action",),
    # mechanics：通用规则（格挡清零、能量、多段攻击），每次战斗都要参考。
    "mechanics": ("combat_next_action",),
}
# 单文件覆盖：key 是相对 guides/ 的 posix 路径。
# 例：想让某份 Boss 备战攻略重新进入地图与篝火决策：
#   "bosses/通用备战清单.md": ("map", "rest_site", "shop", "card_reward"),
DECISION_TYPES_BY_FILE: dict[str, tuple[str, ...]] = {}

# Boss 攻略只在“当前战斗里真的出现这个敌人”时才有用。默认用文件名作为敌人名，
# 多实体 Boss 在这里补别名（战斗状态里出现的是爪牙/信徒等实体名）。
ENEMY_ALIASES_BY_FILE: dict[str, tuple[str, ...]] = {
    "bosses/第一幕/同族小队.md": ("同族小队", "同族神官", "同族信徒"),
    "bosses/第二幕/帝皇蟹.md": ("帝皇蟹", "碾碎爪", "火箭"),
    "bosses/第三幕/女王.md": ("女王", "火炬头聚合体"),
}


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
                "rel_path": path.relative_to(root).as_posix(),
                "category": category or "guides",
                "group": group,
                "character": character,
                "title": title,
                "enemies": resolve_enemies(
                    path.relative_to(root).as_posix(), category, title
                ),
                "url": url,
                "license": license_for(url),
            }
        )
    return plans


def resolve_enemies(rel_path: str, category: str, title: str) -> list[str]:
    """Boss 攻略适用的敌人名；构筑/事件等非战斗攻略返回空列表。"""
    if category != "bosses":
        return []
    if rel_path in ENEMY_ALIASES_BY_FILE:
        return list(ENEMY_ALIASES_BY_FILE[rel_path])
    return [title]


def resolve_decision_types(rel_path: str, category: str, cli_override: list[str]) -> list[str]:
    """按 CLI > 单文件 > 类别 > 默认的顺序解析决策类型。"""
    if cli_override:
        return list(cli_override)
    if rel_path in DECISION_TYPES_BY_FILE:
        return list(DECISION_TYPES_BY_FILE[rel_path])
    return list(DECISION_TYPES_BY_CATEGORY.get(category, DEFAULT_DECISION_TYPES))


def stale_file_overrides(plans: list[dict]) -> list[str]:
    """返回写在 DECISION_TYPES_BY_FILE 里、但磁盘上已不存在的路径。"""
    known = {item["rel_path"] for item in plans}
    return sorted(path for path in DECISION_TYPES_BY_FILE if path not in known)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="遍历 guides/ 并把攻略入库（自动推导从属关系）")
    parser.add_argument("--guides-dir", type=Path, default=Path("guides"))
    parser.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--jsonl", type=Path, default=Path("data/knowledge.jsonl"))
    parser.add_argument("--store", choices=("json", "chroma", "both"), default="both")
    parser.add_argument(
        "--decision-type", action="append", default=[],
        help="全局覆盖默认决策类型（默认按目录类别），可重复",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="先清空 Chroma 与 JSONL 再全量入库；chunk id 是内容哈希，修订后必须用本选项去陈旧副本",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印每个文件的从属关系与决策类型，不写入 Chroma 或 JSONL",
    )
    args = parser.parse_args(argv)

    plans = plan_guides(args.guides_dir)
    if not plans:
        print(f"{args.guides_dir} 下没有找到 .md 文件")
        return 1

    for stale in stale_file_overrides(plans):
        print(f"[警告] DECISION_TYPES_BY_FILE 里的路径不存在：{stale}")

    if args.rebuild and not args.dry_run:
        shutil.rmtree(args.chroma_path, ignore_errors=True)
        if args.jsonl.exists():
            args.jsonl.write_text("", encoding="utf-8")
        print(f"已清空 {args.chroma_path} 与 {args.jsonl}")
    elif args.rebuild and args.dry_run:
        print("[dry-run] 跳过清空步骤")

    failed = 0
    entries: list[dict] = []
    for item in plans:
        effective_types = resolve_decision_types(
            item["rel_path"], item["category"], args.decision_type
        )
        item["decision_types"] = effective_types
        entries.append(
            {
                "path": item["rel_path"],
                "category": item["category"],
                "group": item["group"],
                "character": item["character"],
                "title": item["title"],
                "url": item["url"],
                "license": item["license"],
                "decision_types": effective_types,
                "enemies": item["enemies"],
            }
        )

        print(f"[{'计划' if args.dry_run else '入库'}] {item['rel_path']}")
        print(
            f"        从属: 类别={item['category']} 组={item['group']} "
            f"角色={item['character']} 决策={','.join(effective_types)} "
            f"敌人={','.join(item['enemies']) or '-'}"
        )
        print(f"        来源: {item['url'] or '(未找到链接)'} | 许可: {item['license']}")
        if not item["url"]:
            print("        [警告] 正文没有来源链接，溯源信息不完整")
        if args.dry_run:
            continue

        argv_out = [
            str(item["path"]),
            "--store", args.store,
            "--output", str(args.jsonl),
            "--title", item["title"],
            "--url", item["url"],
            "--character", item["character"],
            *[flag for decision_type in effective_types for flag in ("--decision-type", decision_type)],
            *[flag for enemy in item["enemies"] for flag in ("--enemy", enemy)],
            "--chroma-path", str(args.chroma_path),
            "--license", item["license"],
        ]
        if ingest_main(argv_out) != 0:
            failed += 1

    if args.dry_run:
        print(f"\n[dry-run] 共 {len(entries)} 个文件，未写入任何数据")
        return 0

    manifest_path = Path("data/manifests/guides_index.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "guides_dir": str(args.guides_dir),
                "default_decision_types": list(DEFAULT_DECISION_TYPES),
                "category_decision_types": {
                    key: list(value) for key, value in DECISION_TYPES_BY_CATEGORY.items()
                },
                "file_decision_types": {
                    key: list(value) for key, value in DECISION_TYPES_BY_FILE.items()
                },
                "enemy_aliases": {
                    key: list(value) for key, value in ENEMY_ALIASES_BY_FILE.items()
                },
                "cli_decision_types": list(args.decision_type),
                "entries": entries,
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
