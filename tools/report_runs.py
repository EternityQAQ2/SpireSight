"""每局战绩统计的手动入口（advisor 退出时也会自动刷新）。

用法（项目根目录执行）：
  python tools/report_runs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_advisor.runstats import DEFAULT_DB, DEFAULT_JSON, DEFAULT_MD, refresh  # noqa: E402


def main() -> int:
    count = refresh(DEFAULT_DB, DEFAULT_JSON, DEFAULT_MD)
    print(f"{count} 局 -> {DEFAULT_JSON} 与 {DEFAULT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
