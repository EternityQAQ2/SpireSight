# Wiki 数据工具


## 拉取 TabX

```powershell
python tools/pull.py
```

## 构建 SQLite 事实库

```powershell
python tools/build_wiki_db.py
```

默认输出：`data/sts2_facts.sqlite3`。

默认导入卡牌、遗物、药水、状态、附魔、怪物、事件、效果交互、侵蚀、
关键词和意图。以下数据默认排除：

- `Badge`：成就/徽章，不参与单局决策。
- `Modifier`：仅每日挑战使用；需要时添加 `--include-modifiers`。
- `Trivia`：趣闻，不参与策略建议。
- `Version`：适合作为数据批次元数据，不作为游戏实体检索。

导入过程会删除图片名、百科页面名、图鉴排序、风味文本和编辑备注等展示字段，
同时保留清洗后的说明与原始 Wiki 说明，便于检索和追溯。

注意：当前 `Event.tabx` 只有事件名称、出现阶段和前置条件，没有事件选项及结果；
事件选择建议仍需要事件页面或攻略数据补充。
