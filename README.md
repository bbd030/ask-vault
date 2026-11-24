# AskVault

一个逐步构建的 RAG（检索增强生成）私有知识库问答学习项目。

## 当前流程

TXT / PDF / DOCX 文档 → 重叠切分 → Embedding → 本地向量缓存 → Milvus 入库与检索 → 根据资料生成回答 → 显示参考来源。

## 已实现与后续目标

- [x] 读取 UTF-8 TXT 文档并记录来源
- [x] 段落、句子边界优先切分与片段重叠
- [x] 使用 DashScope 生成文档和问题向量
- [x] 余弦相似度与 Top-K 检索
- [x] 根据检索资料生成回答并标注参考资料编号
- [x] JSON 缓存：文档、模型或切分配置改变时重新生成向量
- [x] Milvus 持久化存储、版本隔离与数据库检索
- [x] 读取有文字层的 PDF 并接入问答
- [x] 读取 Word (.docx) 的正文和表格并接入问答
- [x] Flask 健康检查和问答接口
- [x] Vue 问答页面：提问、回答、参考来源与检索片段
- [x] TXT 上传接口与网页上传（支持指定上传文档问答）
- [x] 已上传文档列表与单文档切换
- [x] 多文档联合检索与原文件下载
- [x] 替换和移除已上传资料

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
- 默认处理 `data/sample.txt`，也支持通过上传编号查询一份 TXT、PDF 或 DOCX。历史文档版本保留在数据库中，检索只选择当前版本；后续再增加文档管理和历史版本清理。
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

## Vue 问答页面

在 Docker Desktop、Milvus 和 Flask 后端已启动的情况下，另开终端启动页面：

```powershell
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。页面支持输入问题、选择参考片段数量、查看回答与原文来源、失败重试和新建对话。

默认读取 `data/sample.txt`；网页上传成功后切换到该文档，首次提问时建立索引。对话仅保存在当前页面内存中，刷新或新建对话会清空显示。

前端通过 Vite 的 `/api` 代理访问本机 Flask，密钥只在 Python 后端读取。默认后端地址为 `http://127.0.0.1:5000`；如调整后端端口，在启动前端前设置 `ASKVAULT_API_TARGET` 为对应地址。

前端检查和构建：

```powershell
cd frontend
npm test
npm run build
```

`frontend/src/App.vue` 管理界面和交互，`frontend/src/api.js` 封装问答请求，`frontend/vite.config.js` 配置本机服务与代理。

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
| `POST /api/upload` | 接收并保存一个 UTF-8 TXT、PDF 或 DOCX，暂不入库 |
| `GET /api/documents` | 列出已上传资料 的编号、原始名称和字节数 |

问答请求示例：

```json
{"question": "报销需要哪些材料？", "top_k": 2}
```

请求头使用 `Content-Type: application/json`。成功返回 `success`、`question`、`answer`、`sources`、`matches` 和 `backend`；不返回密钥或完整向量。

问题限 2000 个字符，`top_k` 为 1 到 10 的整数。输入错误返回 400，错误内容类型返回 415，数据库或模型调用失败返回 503。支持默认 sample.txt 或通过 document_id 指定一份上传资料。

普通接口与共用逻辑检查不调用模型，也不要求 Milvus 已启动：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 网页上传资料

打开 Vue 页面，在侧栏“上传资料”中选择文件，再点击“上传文件”。上传期间禁用选择和提交，成功后显示原始文件名及“已切换为当前资料，可开始提问”。失败时保留所选文件并显示错误，可修正文件或重试。

浏览器先检查扩展名、空文件和 1 MiB 大小限制；UTF-8 编码及空白内容等由后端检查。`frontend/src/api.js` 使用 `FormData` 的 `file` 字段发送文件，不手动设置 `Content-Type`，由浏览器附带 multipart 分隔信息。

上传成功后切换当前问答资料；失败时保持原资料。上传提示只保存在当前页面，刷新会清空提示，已保存的文件仍在本地目录中。上传超时或断网时，文件可能已在服务端保存，重新上传会生成另一个文件。

## 文档上传接口

`POST /api/upload` 使用 `multipart/form-data`，文件字段名为 `file`。
每次接收一个 `.txt`、`.pdf` 或 `.docx`，扩展名不区分大小写。TXT 要求 UTF-8（兼容 BOM）、非空白且无空字符。PDF 使用 pypdf 逐页提取文本，加密文件和无法提取文字的扫描件会在上传时被拒绝。Word 使用 python-docx 按顺序提取正文段落和普通表格，支持 .docx。旧版 .doc 可先转存为 .docx。DOCX 解压后的内容上限为 32 MiB。上传文件最多 1 MiB（1,048,576 字节）。上传请求总大小最多 1 MiB + 64 KiB，其他接口仍限制为 64 KiB。

文件按随机生成的唯一名称保存到 `data/uploads/`，同名上传不会覆盖旧文件。该目录已被 Git 忽略。成功返回 HTTP 201，例如：

```json
{"success": true, "document": {"id": "生成的唯一编号", "filename": "制度.txt", "stored_name": "生成的唯一编号.txt", "size_bytes": 123, "indexed": false}, "message": "文件已保存，可在提问时指定文档编号；首次提问时建立索引"}
```

缺少文件、文件数量或格式不符、编码错误、空白内容返回 400；内容类型错误返回 415；超限返回 413；保存失败返回 503。错误响应为 JSON。

后端关闭了自动重载，更新代码后需在原 Flask 终端按 Ctrl+C，再用项目虚拟环境重启 `server.py`，不要重复启动第二个实例。然后在项目目录用 PowerShell 测试（只使用测试资料）：

```powershell
curl.exe -F "file=@data/sample.txt" http://127.0.0.1:5000/api/upload
```

`curl -F` 会自动设置 multipart 请求头。检查返回的 `stored_name`，确认 `data/uploads/` 下对应文件内容与原文件一致。

上传动作只保存原文件和名称元数据；首次对它提问时才切分、Embedding 并在 Milvus 模式下入库。

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

## 指定上传文档问答

网页上传成功后，自动将它选为当前问答资料并清空旧对话。选择其他文件但未上传、或上传失败时，当前资料不变。刷新页面恢复 sample.txt；文件仍保存在本地。每条问题记录当时的文档编号，失败重试不会悄悄改用另一份资料。

上传仍只保存文件（indexed: false）。首次针对该文档提问时，后端提取文本、切分、生成向量缓存，并在 Milvus 模式下入库后检索。后续提问复用该文档的缓存。来源显示上传时的原始文件名；不同上传编号分别缓存和过滤，也可联合检索多份资料。

POST /api/query 可增加 document_id（上传返回的 32 位编号），例如：

```json
{"question":"这份资料规定了什么？","top_k":2,"document_id":"上传返回的32位编号"}
```

省略 document_id 继续查询 sample.txt；非法编号返回 400，找不到资料返回 404。data/uploads/ 中的同名 .json 保存原始名称，.index.json 是该文档的向量缓存，均不提交 Git。首次提问会调用模型并产生用量，较大文件耗时可能更长。Flask 关闭自动重载，后端修改后需重启原服务。

旧阶段上传的文件如果缺少名称元数据，请重新上传后再用于问答。

## 文档列表与切换

页面加载时通过 GET /api/documents 读取上传目录，刷新页面后仍可在“文档列表”中找到先前上传的资料。选择器同时提供默认 sample.txt。同名文件用编号前六位辅助区分，内部仍使用完整编号。

上传成功会自动更新列表并选中该文档。切换文档会清空当前对话；上传或问答期间暂时禁止切换。刷新页面后当前选择恢复 sample.txt，可从列表重新选择其他资料。列表加载失败会保留已有选项并显示错误，可点击“刷新列表”重试。

列表仅返回 id、filename、size_bytes，按原文件修改时间从新到旧排列；不返回原文、绝对路径或向量，不调用模型、不访问 Milvus。目录尚未创建时返回空列表；缺少原文件、名称信息损坏的条目会跳过，目录无法读取时返回 503。仅有旧 TXT 而无名称信息的文件需要重新上传。可在选择器中切换到全部上传资料。

## 真实模型与 Milvus 联调

需先启动 Docker Desktop 并确认 Milvus 健康。以下命令会真实调用 DashScope，产生少量模型用量；普通单元测试不会执行它：

```powershell
.\.venv\Scripts\python.exe -B scripts/check_live_rag.py --run
```

脚本通过 Flask 测试客户端走上传、列表和问答接口，真实调用向量模型、回答模型和 Milvus；不经过 Vue 或 HTTP 网络服务。测试使用临时目录与独立 ask_vault_live_test 集合前缀，不读取业务资料。仅发送两句自造文本，检查两份同名文档隔离、回答与引用、重复提问复用缓存，以及重复入库不增加片段。结束时删除本次文档版本对应的测试向量，临时文件和缓存自动清理，保留测试集合本身。

2026-11-24 首次实测通过：两份资料均回答各自的口令并引用原始文件名，第三次查询第一份资料时新增文档向量化调用为 0，数据库片段无重复。本检查只验证简单事实场景的完整流程，不代表已经完成问答质量评估。

## 当前切分方式

默认片段上限为 500 个字符，重叠 50 个字符，与课程项目的默认大小保持一致。切点优先在当前窗口后半段查找：空行、换行、中文句末标点、分号、逗号、英文句末标点及空格。都找不到时，才在字符上限处切开。标点保留在前一片段，纯空白片段不进入向量化。

这一步参考课程的分隔符优先级，在现有函数中实现边界查找，没有引入 LangChain。它不是完整的语义识别：超长句仍可能被切断，下一片段也可能从上片段末尾的 50 个字符中间开始。500/50 是初始配置，最终结合整体问答效果再调整。

索引元数据增加 splitter=boundary_v1，并记录片段大小与重叠配置，旧缓存不再匹配；下一次提问会自动重建文档向量。Milvus 继续通过索引标识隔离新旧版本，历史向量暂时保留。

按当前学习节奏，构造阶段只做必要语法或构建检查；切分边界、缓存重建以及各格式问答在功能组装后统一验证。


## 联合检索与下载

选择「全部上传资料」可联合检索本次列表中的上传文档。后端只向量化一次问题，分别检索各文档的当前版本，再按相似度合并取全局前 top_k，最后生成一次回答。同名文件的引用通过文档编号区分。

POST /api/query 可传 document_ids 数组指定多份上传资料，与 document_id 二选一。默认 sample.txt 仍可单独查询。选中一份上传资料后可点击「下载原文件」。GET /api/documents/<document_id>/download 按原始文件名返回上传时保存的内容。


## 替换和移除资料

选中一份上传资料，再选择一个相同格式的文件，点击「替换当前资料」。PUT /api/documents/<document_id> 使用 multipart 的 file 字段，校验成功后一次替换原文件，保留原名称和编号。内容变化会使旧缓存失效，下次提问会重新向量化。默认 sample.txt 作为示例保留。

「移除资料」需在页面确认。DELETE /api/documents/<document_id> 将名称记录改成 \`<id>.deleted.json\`，资料立即退出列表、下载和检索范围。原文件、缓存和历史 Milvus 向量保留。需要恢复时，可停止 Flask 后在上传目录把该名称记录改回 \`<id>.json\`。
