import hashlib
import json
import re

from pymilvus import DataType, MilvusClient


class MilvusVectorStore:
    """保存文档片段，并使用 Milvus 计算余弦相似度。"""

    def __init__(
        self, uri: str, metadata: dict, dimension: int,
        collection_prefix: str = "ask_vault",
    ):
        if dimension <= 0:
            raise ValueError("向量维度必须大于 0")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", collection_prefix):
            raise ValueError("集合前缀只能包含英文字母、数字和下划线")
        self.metadata = metadata
        self.dimension = dimension
        self.index_key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        model_key = hashlib.sha256(
            str(metadata["embedding_model"]).encode("utf-8")
        ).hexdigest()[:12]
        # 不同模型和维度使用不同集合，避免混用不兼容的向量。
        self.collection_name = f"{collection_prefix}_{model_key}_{dimension}"
        self.client = MilvusClient(uri=uri, timeout=30)
        try:
            self._ensure_collection()
        except Exception:
            self.client.close()
            raise

    def _ensure_collection(self):
        if not self.client.has_collection(collection_name=self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)
            schema.add_field("source", DataType.VARCHAR, max_length=2048)
            schema.add_field("chunk_id", DataType.INT64)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("index_key", DataType.VARCHAR, max_length=64)
            indexes = self.client.prepare_index_params()
            indexes.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
            self.client.create_collection(
                collection_name=self.collection_name, schema=schema,
                index_params=indexes, consistency_level="Strong",
            )
        self.client.load_collection(collection_name=self.collection_name)

    def sync_records(self, records: list[dict]) -> bool:
        if len(records) != self.metadata["chunk_count"]:
            raise ValueError("片段数量与当前索引信息不一致")
        if any(len(record["embedding"]) != self.dimension for record in records):
            raise ValueError("片段向量维度不一致")
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'index_key == "{self.index_key}"',
            output_fields=["count(*)"], consistency_level="Strong",
        )
        if rows and rows[0]["count(*)"] == len(records):
            return False

        data = []
        for record in records:
            record_id = hashlib.sha256(
                f"{self.index_key}:{record['chunk_id']}".encode("utf-8")
            ).hexdigest()
            data.append({"id": record_id, "index_key": self.index_key, **record})
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            self.client.flush(collection_name=self.collection_name)
        return True

    def search(self, query_vector: list[float], top_k: int = 2) -> list[dict]:
        if len(query_vector) != self.dimension or top_k <= 0:
            raise ValueError("问题向量维度或检索数量不正确")
        hits = self.client.search(
            collection_name=self.collection_name, data=[query_vector],
            anns_field="embedding", limit=top_k,
            filter=f'index_key == "{self.index_key}"',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["source", "chunk_id", "text"],
            consistency_level="Strong",
        )
        return [{"score": float(hit["distance"]), "record": hit["entity"]}
                for hit in hits[0]]

    def close(self):
        self.client.close()
