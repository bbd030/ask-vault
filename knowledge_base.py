import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from embedding import embed_text
from generator import answer_question
from milvus_store import MilvusVectorStore
from retriever import search_chunks
from vector_store import load_index, save_index


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")

def split_text(
    text: str,
    chunk_size: int = 30,
    overlap: int = 5,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("片段大小必须大于 0")

    if not 0 <= overlap < chunk_size:
        raise ValueError("重叠长度必须大于等于 0，并且小于片段大小")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

    return chunks


class KnowledgeBase:
    """命令行和 HTTP 接口共用的单文档问答流程。"""

    def __init__(self, project_dir: Path | None = None):
        self.project_dir = project_dir or Path(__file__).resolve().parent
        load_dotenv(self.project_dir / ".env")

    def prepare(self) -> tuple[list[dict], dict]:
        file_path = self.project_dir / "data" / "sample.txt"
        text = load_text(file_path)
        if not text.strip():
            raise RuntimeError("知识库文档为空")
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise RuntimeError("请在 .env 中配置 EMBEDDING_MODEL")
        chunk_size, overlap = 30, 5
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        metadata = {
            "source": file_path.name,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding_model": model,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": len(chunks),
        }
        index_path = self.project_dir / "data" / "vector_index.json"
        records = load_index(index_path, metadata)
        if records is None:
            records = []
            for index, chunk in enumerate(chunks, start=1):
                records.append({
                    "chunk_id": index, "source": file_path.name,
                    "text": chunk, "embedding": embed_text(chunk),
                })
            save_index(index_path, metadata, records)
        return records, metadata

    def ask(self, question: str, top_k: int = 2) -> dict:
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
        records, metadata = self.prepare()
        query_vector = embed_text(question, text_type="query")
        if backend == "milvus":
            store = MilvusVectorStore(
                uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19630"),
                metadata=metadata, dimension=len(query_vector),
                collection_prefix=os.getenv("MILVUS_COLLECTION", "ask_vault"),
            )
            try:
                store.sync_records(records)
                results = store.search(query_vector, top_k=top_k)
            finally:
                store.close()
        else:
            results = search_chunks(records, query_vector, top_k=top_k)
        answer = answer_question(question, results)
        matches = [{
            "score": result["score"],
            "source": result["record"]["source"],
            "chunk_id": result["record"]["chunk_id"],
            "text": result["record"]["text"],
        } for result in results]
        return {
            "question": question, "answer": answer["answer"],
            "sources": answer["sources"], "matches": matches, "backend": backend,
        }
