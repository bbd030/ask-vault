# AskVault

一个逐步构建的 RAG（检索增强生成）私有知识库问答学习项目。

## 当前流程

TXT 文档 → 重叠切分 → Embedding → 本地向量缓存 → Milvus 入库与检索 → 根据资料生成回答 → 显示参考来源。

## 已实现与后续目标

- [x] 读取 UTF-8 TXT 文档并记录来源
- [x] 固定长度切分与片段重叠
- [x] 使用 DashScope 生成文档和问题向量
- [x] 余弦相似度与 Top-K 检索
- [x] 根据检索资料生成回答并标注参考资料编号
- [x] JSON 缓存：文档、模型或切分配置改变时重新生成向量
- [x] Milvus 持久化存储、版本隔离与数据库检索
- [ ] 按段落、句子切分及更多文档格式
- [x] Flask 健康检查和问答接口
- [ ] 文件上传接口与 Vue 页面

## 本地启动

需要 Python 3.13 和已运行的 Docker Desktop（WSL 2 后端、Linux 容器）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

第一次配置时，复制 `.env.example` 为 `.env`，然后在 `.env` 中填写自己的 `DASHSCOPE_API_KEY`。已有 `.env` 时直接编辑它。

```powershell
Copy-Item .env.example .env
```

启动本项目的 Milvus，再运行问答程序：

```powershell
docker compose up -d --wait --wait-timeout 180
.\.venv\Scripts\python.exe main.py
```

输入问题，例如：`报销需要哪些材料？`

## 关键配置

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `text-embedding-v1`（配置示例） | 文本向量模型 |
| `LLM_MODEL` | `qwen-plus` | 回答生成模型 |
| `VECTOR_BACKEND` | `milvus` | 选择 Milvus；设为 `json` 可使用本地余弦检索 |
| `MILVUS_URI` | `http://127.0.0.1:19630` | Milvus 本机连接地址 |
| `MILVUS_COLLECTION` | `ask_vault` | 集合名前缀 |

`.env`、`.venv/` 和生成的 JSON 向量缓存不会提交到 Git。模型服务密钥只保存在自己的 `.env` 中。

## 存储与检索

- `data/vector_index.json` 缓存文档向量，避免每次运行重新请求文档 Embedding；问题向量化与回答生成仍会请求模型服务。
- Milvus 数据保存在本项目独立的 Docker 命名卷中。服务端和 Python SDK 均固定使用 2.6.9。
- Milvus 使用 COSINE 度量。不同模型或向量维度进入不同集合，当前文档版本通过索引标识过滤，重复运行不会重复追加同一批片段。
- 当前阶段只处理 `data/sample.txt`。历史文档版本保留在数据库中，检索只选择当前版本；后续再增加文档管理和历史版本清理。
- 相似度阈值 `0.5` 是学习用初始值，需要通过更多问题评估，不能当作答案正确率。

暂停数据库可以执行 `docker compose stop`，再次使用时执行启动命令。

## 代码阅读顺序

| 文件 | 学习内容 |
| --- | --- |
| `main.py` | 命令行提问入口 |
| `knowledge_base.py` | 命令行与接口共用的读取、切分、检索和问答流程 |
| `server.py` | Flask 健康检查、问答接口和请求校验 |
| `embedding.py` | 将文字转换为向量 |
| `retriever.py` | 手写余弦相似度与本地检索 |
| `vector_store.py` | JSON 缓存读取、校验和保存 |
| `milvus_store.py` | 集合、入库、文档版本过滤和向量检索 |
| `generator.py` | 组织参考资料、调用回答模型和处理错误 |
| `docker-compose.yml` | 本机 Milvus 服务和数据卷 |

## Flask 本机接口

启动 Milvus 后，可以使用同一套问答逻辑启动 HTTP 接口：

```powershell
.\.venv\Scripts\python.exe server.py
```

服务只监听本机 `127.0.0.1:5000`，关闭调试和自动重载，当前顺序处理请求。端口可通过 `.env` 中的 `FLASK_PORT` 调整。

| 请求 | 用途 |
| --- | --- |
| `GET /api/health` | 检查 HTTP 服务存活，不触发模型或数据库调用 |
| `POST /api/query` | 传入问题，返回回答、来源编号和检索片段 |

问答请求示例：

```json
{"question": "报销需要哪些材料？", "top_k": 2}
```

请求头使用 `Content-Type: application/json`。成功返回 `success`、`question`、`answer`、`sources`、`matches` 和 `backend`；不返回密钥或完整向量。

问题限 2000 个字符，`top_k` 为 1 到 10 的整数。输入错误返回 400，错误内容类型返回 415，数据库或模型调用失败返回 503。该阶段处理固定的 `data/sample.txt`，尚未提供上传或网页界面。

普通接口与共用逻辑检查不调用模型，也不要求 Milvus 已启动：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 数据库集成检查

Milvus 启动后，可运行以下检查。测试使用独立的 `ask_vault_test` 集合前缀和二维样例向量，不调用模型服务，也不修改业务集合。

```powershell
$env:ASKVAULT_MILVUS_TEST = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖重复入库、分数与来源映射，以及新旧文档版本隔离。测试数据保留在独立测试集合中，可用于重启后的再次验证。

## 项目来源

参考第 5 课 RAG 私有知识库课程，逐步实现并记录学习过程。
Milvus 部署配置参考官方 `standalone_embed.sh` 的嵌入式 etcd 与本地存储方案，只绑定本机端口。
