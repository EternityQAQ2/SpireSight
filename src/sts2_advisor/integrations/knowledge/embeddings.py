"""Optional embedding model configuration shared by ingestion and retrieval."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager


EMBEDDING_MODEL = os.environ.get("STS2_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def chroma_embedding_function():
    """ingest 与查询共用的 sentence-transformers 嵌入函数（懒加载，rag 可选依赖）。

    transformers 5.x 在线检查 Hub 时对部分仓库会误入 AutoProcessor 加载路径而报
    "Unrecognized processing class"；模型已在本地缓存时直接解析快照路径加载，
    并临时置 HF_HUB_OFFLINE=1 杜绝一切在线检查（main 版本解析、
    additional_chat_templates 列举等，网络不通时各要经历约 2 分钟重试）。
    缓存解析失败时回退在线加载并给出警告，不再静默降级。
    """
    from chromadb.utils import embedding_functions

    local_path = cached_model_path(EMBEDDING_MODEL)
    if local_path != EMBEDDING_MODEL:
        with hub_offline_scope():
            # chromadb 的 EF 在构造函数内即刻实例化 SentenceTransformer，
            # 离线作用域需覆盖整个构造过程。
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=local_path
            )
    print(
        f"警告：嵌入模型 {EMBEDDING_MODEL} 的本地缓存解析失败，回退在线加载；"
        "若网络不通，先经历约 2 分钟连接重试后才回退缓存读取。",
        file=sys.stderr,
    )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


@contextmanager
def hub_offline_scope():
    """调用时临时开启 HF_HUB_OFFLINE（huggingface_hub 2.x 每次调用动态读取）。"""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def cached_model_path(model_name: str) -> str:
    """Return the local snapshot directory when the model is cached, else the name."""
    try:
        from huggingface_hub import snapshot_download

        return str(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception:
        return model_name
