# macOS 适配测试记录

日期：2026-10-18（Asia/Shanghai）

## 环境

- macOS 26.5，arm64
- Python 3.13.11
- PyTorch 2.14.1，MPS built / available 均为 True
- Transformers 5.20.0
- 使用项目原有 `uv.lock`，安装 `model`、`dev` 可选依赖

## 修改

- 模型设备自动选择：CUDA → MPS → CPU；新增 `--device` 覆盖。
- MPS 显式移动模型至 GPU；CPU float32，MPS/CUDA float16；输入保持跟随模型设备。
- 向模型和 tokenizer 显式传入 Hugging Face 缓存目录，保留离线加载及原有缓存环境变量；已有环境变量不被覆盖。
- 补充 macOS 安装、运行、入库、测试说明及 `.pth` 隐藏标记问题的修复方法。

## 验证结果

```text
python -m pytest -q
25 passed, 2 subtests passed

PYTHONPATH=src python -m unittest discover -s tests -v
Ran 25 tests
OK
```

25 项测试包含原有 5 项规范化/合法性测试，以及设备选择、模拟 HTTP 服务到 CLI 的完整调用、中文和空格路径、中文录制、HTTP 409、无效 JSON、监听去重和 SIGINT 退出。

为保持 Windows 功能，新增模型加载边界测试，验证 CUDA 自动映射、float16、离线加载参数不变，强制 CPU 能正确关闭 CUDA 映射，原有 HF_HOME / HF_HUB_CACHE / HF_XET_CACHE 默认设置和已有环境变量覆盖关系保留。上述边界测试模拟依赖加载，不代表 Windows 或 NVIDIA GPU 实机验证。README 和训练说明保留 Windows PowerShell 示例；POSIX SIGINT 测试在 Windows 上跳过。

模型测试使用本地随机初始化的微型 GPT-2 权重和 tokenizer，无外部模型下载：

- 实际 CPU float32 加载、前向计算、生成通过。
- 实际 MPS float16 加载、前向计算、生成通过；检查输出数值有限。
- Transformers 已导入后从显式 Hugging Face hub 缓存离线加载通过。
- 实际随机模型生成非法答案后，服务降级到合法规则建议通过。

其他检查：

- 安装后的 `sts2-advisor --help`、`sts2-ingest` 从其他工作目录运行通过。
- 中文且含空格的攻略文件入库通过，来源、许可证、`unreviewed` 标记保留。
- `uv build --offline` 构建 sdist 和 wheel 通过。
- 运行时模拟服务观察到的请求均为 `GET /api/v1/singleplayer`。

## 未通过或未验证的范围

真实游戏 API 契约检查：

```text
python test_api.py --timeout 1
0 passed, 0 skipped, 6 failed
Connection refused: localhost:15526
```

单人、多人、profile、compendium、profiles、wiki 六个端点均因本机 API 未启动而连接拒绝。此结果不能当作真实游戏联调通过；启动游戏及 STS2MCP 后需重跑。

未下载或运行 MiniCPM5-2B 完整权重；微型 GPT-2 测试只证明通用加载与生成链路，不能证明 MiniCPM 自定义代码、算子或策略质量。未验证 Windows 实机、Chroma embedding、Intel Mac、CUDA 硬件及 SFT/DPO 长训练。CUDA 设备选择和加载只做边界回归验证。

unittest 模型测试会显示 PyTorch 关于 macOS 不支持 distributed multiprocessing redirect 的提示；单进程推理测试通过，不使用该能力。
