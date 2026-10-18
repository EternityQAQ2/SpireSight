from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.build_wiki_db import clean_wiki, load_export


class WikiDatabaseTests(unittest.TestCase):
    def test_loads_embedded_tabx_jsonl(self) -> None:
        document = {
            "schema": {"fields": [{"name": "id"}, {"name": "name"}]},
            "data": [["strike_ironclad", "打击"]],
        }
        wrapper = {"raw_line": json.dumps(document, ensure_ascii=False), "source_file": "Card.tabx"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Card.jsonl"
            path.write_text(json.dumps(wrapper, ensure_ascii=False) + "\n", encoding="utf-8")
            _, records = load_export(path)
        self.assertEqual(records, [{"id": "strike_ironclad", "name": "打击"}])

    def test_cleans_wiki_markup_without_losing_game_values(self) -> None:
        source = "造成{{颜色|blue|8}}点伤害。给予{{颜色|gold|2}}层[[易伤]]。"
        self.assertEqual(clean_wiki(source), "造成8点伤害。给予2层易伤。")


if __name__ == "__main__":
    unittest.main()
