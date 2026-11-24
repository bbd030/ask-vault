"""
向量数据库管理模块
基于LangChain和Milvus实现文档切分、向量化存储和检索功能
- 1.
MinIO (对象存储服务) : Milvus 使用 MinIO 来存储数据。它提供了一个网页管理界面。

- 登录页面 : http://localhost:9001
- 账号 (Access Key) : minioadmin
- 密码 (Secret Key) : minioadmin
- 2.
Milvus (向量数据库) : 在我们当前的配置中，Milvus 本身 没有 提供一个用于登录的网页界面。您可以通过代码（例如使用 pymilvus 库）连接到 Milvus 服务来进行操作。

- 连接地址 : localhost:19630（见 demos/lesson5/docker-compose.yml）
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load env vars from explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# LangChain and Milvus imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from pymilvus import utility, connections, Collection

# 兼容补丁：pymilvus 2.6 + langchain-milvus 0.3.x 下，入库会报
# "should create connection first."（详见 milvus_compat.py 顶部说明）
from milvus_compat import patch_langchain_milvus, cosine_index_params
patch_langchain_milvus()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """向量数据库管理器 (Milvus后端)"""
    
    def __init__(self, 
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        初始化向量数据库管理器
        
        Args:
            milvus_host: Milvus 服务主机
            milvus_port: Milvus 服务端口
            collection_name: Milvus 集合名称
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        # Ensure port is a string as pymilvus might expect it, or handle int gracefully
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19630"))
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(chunk_overlap or os.getenv("CHUNK_OVERLAP", "50"))
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化文档切分器
        # 说明：chunk_size / chunk_overlap 可用环境变量覆盖（默认 500 / 50）。
        # 演示数据是"一份文件里塞多条制度"的短文档，500 会把多条制度混进同一块，
        # 拉低相似度；这类小语料用 CHUNK_SIZE=100 检索更准（见第 6 课调优）。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 连接到Milvus
        self._connect_to_milvus()
        
        # 向量数据库实例
        self.vectorstore = None
        # 延迟加载：不要在 __init__ 中调用 _load_existing_db，避免启动时连接未就绪或报错
        # self._load_existing_db() 

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 确保 API Key 存在
            if not self.dashscope_api_key:
                 logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                 self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            # 简单测试 embedding 是否工作
            try:
                self.embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
            except Exception as e:
                 logger.error(f"DashScope 模型验证失败: {e}")
                 raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def _connect_to_milvus(self):
        """连接到Milvus服务"""
        try:
            logger.info(f"Connecting to Milvus: host={self.milvus_host}, port={self.milvus_port}")
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"成功连接到Milvus: {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _load_existing_db(self):
        """加载已存在的Milvus集合"""
        try:
            if utility.has_collection(self.collection_name):
                try:
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=self.collection_name,
                        connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"}
                    )
                    logger.info(f"成功加载现有Milvus集合: {self.collection_name}")
                except Exception as e:
                    logger.error(f"加载现有集合失败: {e}")
                    # 不再自动删除，防止误删数据。
                    # 如果 Schema 不兼容，将在写入时处理。
                    self.vectorstore = None
            else:
                logger.info(f"未找到集合 {self.collection_name}，将在添加文档时创建")
        except Exception as e:
            logger.error(f"加载Milvus集合失败: {e}")

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        # ... (代码与之前版本相同)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            loader = None
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                # .doc 是老式二进制格式，LangChain / python-docx 都读不了：
                # 先用 textutil(macOS) 或 LibreOffice 转成 .docx 再走统一下面的路径。
                if file_extension == '.doc':
                    converted = self._convert_doc_to_docx(file_path)
                    if converted:
                        file_path = str(converted)
                        file_extension = '.docx'
                    else:
                        logger.error(f".doc 无法转换（缺 textutil/LibreOffice）: {file_path}")
                        return []
                # 首选 LangChain 官方加载器（依赖 docx2txt / unstructured）
                try:
                    loader = Docx2txtLoader(file_path)
                except Exception as e:
                    logger.warning(f"Docx2txtLoader 不可用（{e}），改用内置 python-docx 解析")
                    loader = None
            elif file_extension in ['.xlsx', '.xls']:
                # 首选 LangChain 官方加载器（依赖 unstructured）
                try:
                    loader = UnstructuredExcelLoader(file_path)
                except Exception as e:
                    logger.warning(f"UnstructuredExcelLoader 不可用（{e}），改用内置 openpyxl 解析")
                    loader = None
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")

            if loader is not None:
                try:
                    documents = loader.load()
                except ModuleNotFoundError as e:
                    # LangChain 的 Docx2txtLoader / UnstructuredExcelLoader 在
                    # **load() 时**才 import docx2txt / unstructured，构造时不报错；
                    # 这里捕获后走内置兜底解析，避免出现"上传成功但 0 个块"。
                    logger.warning(f"加载器缺少可选依赖（{e}），改用内置解析: {file_extension}")
                    documents = self._fallback_load(file_path, file_extension)
            else:
                # 兜底解析：不依赖 unstructured / docx2txt 等可选包
                documents = self._fallback_load(file_path, file_extension)

            if not documents:
                logger.warning(f"文件解析为空: {file_path}（{file_extension}）")
                return []

            # 文本清洗：去控制字符/零宽字符、U+FFFD 占位、归一空白、合并被硬换行切断的中文句子
            # 对应课件 §5.1 备注"真实项目还要做清洗"——脏文本会同时拉低检索质量和引用可读性
            try:
                from text_cleaner import clean_documents, quality_report
                before_text = "\n".join(d.page_content or "" for d in documents)
                before_len = len(before_text)
                documents = clean_documents(documents)
                after_len = sum(len(d.page_content or "") for d in documents)
                # 质量体检要在**清洗前**做，否则控制字符已被清掉、统计不到
                rep = quality_report(os.path.basename(file_path), before_text)
                if rep["replacement_chars"] or rep["control_chars"]:
                    logger.warning(
                        f"⚠️ 文本质量提示 [{os.path.basename(file_path)}]："
                        f"抽不出的字 {rep['replacement_chars']} 个（已标成 □）、"
                        f"控制字符 {rep['control_chars']} 个（已清除）"
                    )
                if before_len != after_len:
                    logger.info(f"清洗：{before_len:,} → {after_len:,} 字符")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"文本清洗跳过（{e}）")

            if not documents:
                logger.warning(f"清洗后无有效内容: {file_path}")
                return []

            # 统一 source 为文件名（而非完整路径），便于前端列表展示与删除/下载定位
            source_name = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source"] = source_name
                doc.metadata["file_name"] = source_name
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    def _convert_doc_to_docx(self, path: str):
        """把老式 .doc 转成 .docx（python-docx / Docx2txtLoader 都读不了 .doc）。

        转换器优先级：macOS 自带 textutil → LibreOffice 无头模式。
        转换失败返回 None，由调用方明确报错，而不是静默产出 0 个块。
        产物落在临时目录，不污染语料目录。
        """
        import shutil
        import subprocess
        import tempfile

        src = Path(path)
        out_dir = Path(tempfile.gettempdir()) / "lesson5_doc_conv"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (src.stem + ".docx")
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out

        textutil = shutil.which("textutil")
        if textutil:
            try:
                subprocess.run([textutil, "-convert", "docx", "-output", str(out), str(src)],
                               check=True, capture_output=True, timeout=120)
                if out.exists() and out.stat().st_size > 0:
                    logger.info(f".doc 已用 textutil 转为 .docx: {out.name}")
                    return out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"textutil 转换失败，改用 LibreOffice：{e}")

        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            try:
                subprocess.run([soffice, "-env:UserInstallation=file://" + str(out_dir / "lo-profile"),
                                "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(src)],
                               check=True, capture_output=True, timeout=300)
                if out.exists() and out.stat().st_size > 0:
                    logger.info(f".doc 已用 LibreOffice 转为 .docx: {out.name}")
                    return out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LibreOffice 转换失败：{e}")
        return None

    def _fallback_load(self, file_path: str, ext: str) -> List[Document]:
        """内置兜底解析器：docx 用 python-docx，xlsx/xls 用 openpyxl。

        为什么需要它（也是课件 §5.1 的真实坑）：
          LangChain 的 Docx2txtLoader / UnstructuredExcelLoader 依赖 docx2txt、
          unstructured 等可选包，环境里没装就直接报 ModuleNotFoundError，
          学生看到的现象就是「上传了但问不到」。这里给一条零外部依赖的兜底路径，
          顺便演示「表格结构丢失」这件事——表格会被拍平成 "表头: 值" 的文本行。
        """
        docs: List[Document] = []

        if ext in ('.docx', '.doc'):
            try:
                import docx  # python-docx
                d = docx.Document(file_path)
                parts = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]
                # 表格也拍平成文本（表格结构本身会丢失，这正是课件提到的坑）
                for ti, table in enumerate(getattr(d, "tables", []) or [], 1):
                    parts.append(f"[表格{ti}]")
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        if any(cells):
                            parts.append(" | ".join(cells))
                text = "\n".join(parts)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"parser": "python-docx"}))
            except Exception as e:
                logger.error(f"python-docx 兜底解析失败 {file_path}: {e}")
            return docs

        if ext in ('.xlsx', '.xls'):
            try:
                from openpyxl import load_workbook
                wb = load_workbook(file_path, read_only=True, data_only=True)
                for ws in wb.worksheets:
                    lines = [f"[工作表] {ws.title}"]
                    for row in ws.iter_rows(values_only=True):
                        cells = ["" if c is None else str(c) for c in row]
                        if any(c.strip() for c in cells):
                            lines.append(" | ".join(cells).rstrip(" |"))
                    text = "\n".join(lines)
                    if len(text) > len(f"[工作表] {ws.title}"):
                        docs.append(Document(page_content=text,
                                             metadata={"parser": "openpyxl", "sheet": ws.title}))
                wb.close()
            except Exception as e:
                logger.error(f"openpyxl 兜底解析失败 {file_path}: {e}")
            return docs

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        # ... (代码与之前版本相同)
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """
        将文档添加到Milvus数据库
        
        Args:
            documents: 文档列表
            collection_name: 集合名称（可选，覆盖默认）
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return
        
        target_collection = collection_name or self.collection_name
        
        try:
            # 检查集合是否存在
            collection_exists = utility.has_collection(target_collection)
            
            if self.vectorstore is None or self.collection_name != target_collection:
                # 初始化 vectorstore
                if collection_exists:
                    logger.info(f"加载现有集合: {target_collection}")
                else:
                    logger.info(f"集合不存在，将创建新集合: {target_collection}")
                
                # Milvus.from_documents 和 Milvus(...) 的区别：
                # from_documents: 会根据文档内容创建集合（如果不存在），并插入数据。
                # Milvus(...): 仅初始化客户端，不插入数据，通常用于检索或追加。
                
                # 策略：始终使用 Milvus() 初始化，然后调用 add_documents。
                # 如果是首次创建，我们需要先确保集合存在，或者让 add_documents 处理？
                # LangChain 的 Milvus.from_documents 是最方便的初始化+插入入口。
                # 但为了避免重复创建/Schema冲突，我们应该：
                # 1. 如果集合存在，用 Milvus() 加载，然后 add_documents()
                # 2. 如果集合不存在，用 Milvus.from_documents()
                
                if collection_exists:
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"},
                        # 关键：auto_id=True 通常是默认的，但确保配置一致
                    )
                    self.collection_name = target_collection
                    # 追加数据
                    self.vectorstore.add_documents(documents)
                    logger.info(f"成功向现有集合 '{target_collection}' 追加 {len(documents)} 条文档")
                else:
                    # 集合不存在，创建并插入
                    self.vectorstore = Milvus.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"},
                        # metric_type：决定"像不像"怎么算，建索引时定死。
                        #   默认 AUTOINDEX 走 L2（欧式距离 √Σ(a-b)²，越小越像，
                        #   分数是 6247 这种量级）→ 与 vector_retriever 里按相似度
                        #   设的阈值（score >= 0.5）不在同一量纲，过滤会失效。
                        #   指定 COSINE 后分数是 0~1 余弦相似度，阈值才真正起作用。
                        index_params=cosine_index_params(),
                        # drop_old=False：建表前**不要**删掉同名集合（这里是"集合不存在"
                        #   分支，本就不会冲突，写 False 是防御性的）。
                        #   ⚠️ 若改成 True，就变成"每次上传新文件都先清空整个知识库"——
                        #   学生传 3 份只会剩最后 1 份。上传接口必须保持 False。
                        drop_old=False
                    )
                    self.collection_name = target_collection
                    logger.info(f"成功创建集合 '{target_collection}' 并插入 {len(documents)} 条文档")
            else:
                # vectorstore 已初始化且集合名称匹配，直接追加
                self.vectorstore.add_documents(documents)
                logger.info(f"成功向当前集合 '{target_collection}' 追加 {len(documents)} 条文档")
            
        except Exception as e:
            # 捕获 Schema 不兼容错误并尝试自动修复（重建）
            msg = str(e)
            if "non-exist field" in msg or "inconsistent with defined schema" in msg:
                 logger.warning(f"检测到 Schema 不兼容 ({e})，尝试重建集合...")
                 try:
                     if utility.has_collection(target_collection):
                         utility.drop_collection(target_collection)
                     
                     self.vectorstore = Milvus.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"},
                        index_params=cosine_index_params(),
                        # drop_old=True：这里是【主动重建】场景（老集合字段与代码不一致，
                        # 上面已 drop 过，这里再删一次也无妨）。重建会连 Schema + 索引
                        # 一起重来，所以比单纯清数据慢；另外 Milvus 的 drop 是异步回收，
                        # 刚删完查 num_entities 可能还没降下来，属正常现象。
                        drop_old=True
                     )
                     self.collection_name = target_collection
                     logger.info(f"集合 '{target_collection}' 已重建并写入数据")
                 except Exception as re:
                     logger.error(f"重建集合失败: {re}")
                     raise re
            else:
                logger.error(f"添加文档到Milvus失败: {e}")
                raise e

    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """
        处理单个文件：加载、切分、存储
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            documents = self.load_document(file_path)
            if not documents:
                return False
            
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs, collection_name)
            
            logger.info(f"文件处理完成: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """
        处理CSV数据文件
        
        Args:
            csv_path: CSV文件路径
            text_columns: 需要向量化的文本列名列表
            
        Returns:
            处理是否成功
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]

            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]

            documents = []
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {"source": csv_path, "row_index": idx}
                
                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            content_parts.append(f"{col}: {text}")

                for col in metadata_columns:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[str(col)] = str(val)
                
                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs)
            
            return True
            
        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

    def search(self, query: str, k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: Milvus目前不支持直接的元数据过滤，此参数保留但暂不使用
            collection_name: 指定要搜索的集合名称（可选）
        
        Returns:
            (文档, 相似度分数) 列表
        """
        target_collection = collection_name or self.collection_name

        if self.vectorstore is None or (target_collection and self.collection_name != target_collection):
            try:
                if target_collection and utility.has_collection(target_collection):
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"}
                    )
                    self.collection_name = target_collection
                    logger.info(f"加载集合用于搜索: {target_collection}")
                else:
                    logger.warning("向量数据库未初始化")
                    return []
            except Exception as e:
                logger.error(f"加载Milvus集合失败: {e}")
                return []
        
        try:
            if filter_dict:
                logger.warning("Milvus集成当前不支持直接的元数据过滤，该过滤器将被忽略。")
            
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取数据库信息
        """
        target_collection = collection_name or self.collection_name
        
        info = {
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "collection_name": target_collection,
            "is_initialized": self.vectorstore is not None
        }
        
        try:
            # 如果 vectorstore 未初始化，尝试临时连接检查
            if utility.has_collection(target_collection):
                # 使用 Collection 对象获取统计信息
                col = Collection(target_collection)
                # 刷新以确保获取最新数据量（刚插入的数据可能还在内存中）
                col.flush()
                info["document_count"] = col.num_entities
                
                # 如果 self.vectorstore 为空但集合存在，尝试初始化它（如果还没做过）
                if self.vectorstore is None:
                     try:
                        self.vectorstore = Milvus(
                            embedding_function=self.embeddings,
                            collection_name=target_collection,
                            connection_args={"uri": f"http://{self.milvus_host}:{self.milvus_port}"}
                        )
                        info["is_initialized"] = True
                     except:
                        pass # 忽略加载错误，只返回统计
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取Milvus集合信息失败: {e}")
            info["error"] = str(e)
        
        return info

    def clear_database(self):
        """清空Milvus集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.vectorstore = None
                logger.info(f"Milvus集合 '{self.collection_name}' 已被删除")
        except Exception as e:
            logger.error(f"清空Milvus集合失败: {e}")

    def list_documents(self, collection_name: str = None) -> List[Dict]:
        """
        列出已入库的文档（按 source 分组统计 chunk 数）

        Args:
            collection_name: 集合名称

        Returns:
            文档列表 [{"source": 文件名, "chunk_count": 块数}, ...]
        """
        from collections import Counter
        target_collection = collection_name or self.collection_name
        try:
            col = Collection(target_collection)
            col.load()
            rows = col.query(expr="pk >= 0", output_fields=["source"], limit=16384)
            counter = Counter(r.get("source", "未知文件") for r in rows)
            return [{"source": src, "chunk_count": n} for src, n in counter.items()]
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []

    def delete_document(self, source: str, collection_name: str = None) -> bool:
        """
        按来源文件名删除该文档的所有 chunk

        Args:
            source: 来源文件名（对应 metadata 的 source 字段）
            collection_name: 集合名称

        Returns:
            是否删除成功
        """
        target_collection = collection_name or self.collection_name
        try:
            col = Collection(target_collection)
            col.load()
            expr = f'source == "{source}"'
            col.delete(expr)
            col.flush()
            logger.info(f"已删除文档 '{source}' 的所有分块（集合 {target_collection}）")
            # 让 langchain vectorstore 下次搜索时重新加载，避免陈旧引用
            if target_collection == self.collection_name:
                self.vectorstore = None
            return True
        except Exception as e:
            logger.error(f"删除文档 '{source}' 失败: {e}")
            raise

    def get_document_text(self, source: str, collection_name: str = None) -> str:
        """
        按来源文件名取出该文档所有 chunk 文本，按 pk 升序拼回（下载兜底用）

        Args:
            source: 来源文件名
            collection_name: 集合名称

        Returns:
            拼接后的完整文本
        """
        target_collection = collection_name or self.collection_name
        try:
            col = Collection(target_collection)
            col.load()
            rows = col.query(expr=f'source == "{source}"', output_fields=["pk", "text"], limit=16384)
            rows.sort(key=lambda r: r.get("pk", 0))
            return "\n".join(r.get("text", "") for r in rows)
        except Exception as e:
            logger.error(f"读取文档 '{source}' 文本失败: {e}")
            raise

def main():
    """测试函数"""
    # 确保Milvus服务正在运行
    try:
        connections.connect("default", host="127.0.0.1", port=int(os.getenv("MILVUS_PORT", "19630")))
        utility.get_collection_stats("agent_rag")
        connections.disconnect("default")
    except Exception as e:
        logger.error("无法连接到Milvus服务，请确保您已通过 docker compose up -d 启动了Milvus（默认 19630）。")
        logger.error(f"错误: {e}")
        return

    # 创建向量数据库管理器
    db_manager = VectorDatabaseManager()
    
    # 清空现有数据
    print("清空现有数据库...")
    db_manager.clear_database()

    # 处理学科数据CSV文件
    csv_path = "../data.csv"
    if os.path.exists(csv_path):
        print("处理CSV数据...")
        success = db_manager.process_csv_data(csv_path)
        if success:
            print("CSV数据处理成功！")
            
            # 测试搜索
            print("\n测试搜索功能:")
            results = db_manager.search("示例查询", k=3)
            for i, (doc, score) in enumerate(results):
                print(f"\n结果 {i+1} (相似度: {score:.4f}):")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"元数据: {doc.metadata}")
        else:
            print("CSV数据处理失败！")
    else:
        print(f"未找到数据文件: {csv_path}")
    
    # 显示数据库信息
    print("\n数据库信息:")
    info = db_manager.get_database_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
