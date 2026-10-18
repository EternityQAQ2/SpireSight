# 后训练数据约定

训练不会直接使用未审核的会话记录。

## SFT 标注

`annotations.jsonl` 每行：

```json
{"decision_id":"sha256:...","approved":true,"reviewer":"name","advice":{"decision_id":"sha256:...","candidate_id":"card:2","reason":"...","confidence":0.9,"alternative_ids":[],"evidence_ids":[],"warnings":[]}}
```

构建数据：

macOS（zsh / bash）：

```bash
.venv/bin/python training/build_sft.py recordings/*.jsonl --annotations data/annotations.jsonl
```

Windows（PowerShell）：

```powershell
.venv\Scripts\python.exe training\build_sft.py recordings\*.jsonl --annotations data\annotations.jsonl
```

## DPO 标注

`preferences.jsonl` 每行：

```json
{"decision_id":"sha256:...","approved":true,"reviewer":"name","chosen":{"candidate_id":"card:2","reason":"..."},"rejected":{"candidate_id":"card:0:NIBBIT_0","reason":"..."},"preference_reason":"避免不必要掉血"}
```

chosen 和 rejected 必须都是当前局面的合法候选。明显非法答案不适合作为主要 rejected。

## 训练

macOS 的推理与数据准备见 [macOS 使用说明](../docs/macos.md)。本次适配未验证 macOS 上的 SFT/DPO 或 QLoRA；以下训练命令保留原有服务器/Windows 工作流。

安装训练依赖后先 SFT，再 DPO：

```powershell
uv sync --extra train
.venv\Scripts\python.exe training\train_sft.py
.venv\Scripts\python.exe training\train_dpo.py --adapter artifacts\adapters\sft
```

6GB 显存可能不足以用当前脚本直接训练 2B 模型。正式训练前应加入与你的 GPU 匹配的 4-bit QLoRA 配置，或使用更大显存设备；不要在未验证显存预算时直接启动长训练。

