"""
Milvus / langchain-milvus 兼容补丁
==================================
为什么需要这个文件（本机实测的真实问题）：

  langchain-milvus 0.3.x 在内部用 `MilvusClient` 建连接，拿到一个随机连接别名
  （例如 `cm-4333650912`），但仍用 **ORM 风格**的 `pymilvus.Collection(name, using=alias)`
  去读集合 schema 和索引。pymilvus 2.6.x 里 `MilvusClient` 不会再把这个别名注册进
  ORM 的全局连接表，于是抛：

      pymilvus.exceptions.ConnectionNotExistException:
          <ConnectionNotExistException: (code=1, message: should create connection first.)>

  结果就是：**上传文档（入库）直接失败**，而搜索能跑但拿不到分数。

补丁做的事：
  `patch_milvus_orm_alias()`：包装 `pymilvus.Collection.__init__`——当 langchain_milvus
  用未注册的别名取集合时，按 .env / `connection_args` 里的 Milvus 地址现场补一个 ORM
  连接再重试（幂等，失败不影响主流程）。

用法（在任何用到 langchain_milvus 的模块顶部、import 之后调用一次）：
    from milvus_compat import patch_langchain_milvus
    patch_langchain_milvus()
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def _default_uri() -> str:
    """从环境变量拼出 Milvus URI，供补连接时使用。"""
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    # 默认 19630：与 demos/lesson5/docker-compose.yml 保持一致（避让本机其它 Milvus）
    port = os.environ.get("MILVUS_PORT", "19630")
    return os.environ.get("MILVUS_URI", f"http://{host}:{port}")


def patch_milvus_orm_alias() -> bool:
    """给 langchain_milvus 的 Milvus 实例补上 ORM 连接别名。

    Returns:
        True 表示补丁已生效（或早已生效），False 表示当前环境不需要/不支持该补丁。
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from langchain_milvus.vectorstores import milvus as lc_milvus
        from pymilvus import Collection, connections
    except Exception as e:  # pragma: no cover - 依赖缺失时静默降级
        logger.warning("milvus_compat: 无法导入 langchain_milvus/pymilvus (%s)，跳过补丁", e)
        return False

    # 修补 pymilvus.Collection：langchain_milvus 传入的 using 别名若没在 ORM
    # 连接表里注册，就按 .env / connection_args 里的 Milvus 地址现场补一个连接再重试。
    original_collection_init = Collection.__init__
    if not getattr(original_collection_init, "_dsh_alias_patched", False):

        def patched_collection_init(self, name, schema=None, using="default", **kwargs):
            try:
                return original_collection_init(self, name, schema, using, **kwargs)
            except Exception as e:  # noqa: BLE001
                if "should create connection first" not in str(e):
                    raise
                uri = os.environ.get("MILVUS_URI") or _default_uri()
                connections.connect(alias=using, uri=uri)
                logger.debug("milvus_compat: 为别名 %s 现场补连接 -> %s", using, uri)
                return original_collection_init(self, name, schema, using, **kwargs)

        patched_collection_init._dsh_alias_patched = True
        Collection.__init__ = patched_collection_init

    # langchain_milvus 在自己模块里绑定了 Collection 名字，确保它指向修补后的类
    lc_milvus.Collection = Collection

    _PATCHED = True
    logger.info("milvus_compat: 已打补丁（pymilvus.Collection 别名自动补连接）")
    return True


def patch_langchain_milvus() -> None:
    """对外统一入口：应用所有可用的兼容补丁。"""
    patch_milvus_orm_alias()


def cosine_index_params() -> dict:
    """返回 COSINE 度量 + AUTOINDEX 的索引参数。

    为什么必须显式指定 COSINE：
      langchain-milvus 默认 `AUTOINDEX` 的度量是 **L2（欧式距离，越小越好）**，
      而 `vector_retriever.py` 里的相似度阈值（默认 0.5）是按 **0~1 相似度** 设的。
      不指定 COSINE 的话，分数会是几千量级的距离值或反向相似度，
      `score >= 0.5` 过滤逻辑就失去意义（要么全留要么全丢）。
    """
    return {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
