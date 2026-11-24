import json
import re
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from document_loader import document_suffix, extract_document_text
from embedding import embed_text
from generator import answer_question
from milvus_store import MilvusVectorStore
from retriever import search_chunks
from vector_store import load_index, save_index


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8-sig")

def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """优先在窗口后半段的段落或句子边界切分，保留尾部字符重叠。"""
    if chunk_size <= 0:
        raise ValueError("片段大小必须大于 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("重叠长度必须大于等于 0，并且小于片段大小")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 参考课程的分隔符顺序：先段落、换行，再句子、短句和空格。
    separators = ("\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 保证切点超过重叠区，让下一次 start 严格前进。
            # 优先在后半段寻找边界，避免短段落造成过小片段。
            search_start = start + max(chunk_size // 2, overlap + 1)
            for separator in separators:
                position = text.rfind(separator, search_start, end)
                if position != -1:
                    end = position + len(separator)
                    break
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


class KnowledgeBase:
    """命令行和 HTTP 接口共用的单文档问答流程。"""

    def __init__(self, project_dir: Path | None = None):
        self.project_dir = project_dir or Path(__file__).resolve().parent
        load_dotenv(self.project_dir / ".env")

    def prepare(self, document_id: str | None = None) -> tuple[list[dict], dict]:
        file_path = self.project_dir / "data" / "sample.txt"
        source = file_path.name
        index_path = self.project_dir / "data" / "vector_index.json"
        if document_id is not None:
            if not isinstance(document_id, str) or not re.fullmatch(r"[0-9a-f]{32}", document_id):
                raise ValueError("文档编号格式不正确")
            folder = self.project_dir / "data" / "uploads"
            info = json.loads((folder / f"{document_id}.json").read_text(encoding="utf-8"))
            source = info["filename"]
            file_path = folder / f"{document_id}{document_suffix(source)}"
            index_path = folder / f"{document_id}.index.json"
        if file_path.suffix != ".txt":
            text = extract_document_text(file_path.read_bytes(), file_path.suffix)
        else:
            text = load_text(file_path)
        if not text.strip():
            raise RuntimeError("知识库文档为空")
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise RuntimeError("请在 .env 中配置 EMBEDDING_MODEL")
        chunk_size, overlap = 500, 50
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        metadata = {
            "source": source,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding_model": model,
            "splitter": "boundary_v1",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": len(chunks),
        }
        if document_id is not None:
            metadata["document_id"] = document_id
        records = load_index(index_path, metadata)
        if records is None:
            records = []
            for index, chunk in enumerate(chunks, start=1):
                records.append({
                    "chunk_id": index, "source": source,
                    "text": chunk, "embedding": embed_text(chunk),
                })
            save_index(index_path, metadata, records)
        return records, metadata

    def ask(self, question: str, top_k: int = 2, document_id: str | None = None,
            document_ids: list[str] | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        question = question.strip()
        if len(question) > 2000:
            raise ValueError("问题不能超过 2000 个字符")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k 必须是 1 到 10 之间的整数")
        backend = os.getenv("VECTOR_BACKEND", "milvus").strip().lower()
        if backend not in {"milvus", "json"}:
            raise RuntimeError("VECTOR_BACKEND 只能设置为 milvus 或 json")
        selected_ids = document_ids if document_ids is not None else [document_id]
        if not selected_ids:
            raise ValueError("请至少指定一份资料")
        prepared = [self.prepare(document_id=selected_id) for selected_id in selected_ids]
        query_vector = embed_text(question, text_type="query")
        results = []
        for selected_id, (records, metadata) in zip(selected_ids, prepared):
            if backend == "milvus":
                store = MilvusVectorStore(
                    uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19630"),
                    metadata=metadata, dimension=len(query_vector),
                    collection_prefix=os.getenv("MILVUS_COLLECTION", "ask_vault"),
                )
                try:
                    store.sync_records(records)
                    document_results = store.search(query_vector, top_k=top_k)
                finally:
                    store.close()
            else:
                document_results = search_chunks(records, query_vector, top_k=top_k)
            for result in document_results:
                record = {**result["record"]}
                if selected_id is not None:
                    record["document_id"] = selected_id
                results.append({"score": result["score"], "record": record})
        results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:top_k]
        answer = answer_question(question, results)
        matches = [{
            "score": result["score"],
            "source": result["record"]["source"],
            "chunk_id": result["record"]["chunk_id"],
            "text": result["record"]["text"],
            **({"document_id": result["record"]["document_id"]} if "document_id" in result["record"] else {}),
        } for result in results]
        return {
            "question": question, "answer": answer["answer"],
            "sources": answer["sources"], "matches": matches, "backend": backend,
        }
