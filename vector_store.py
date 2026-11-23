import json
import math
from pathlib import Path


def load_index(path: Path, metadata: dict) -> list[dict] | None:
    """读取与当前文档、模型和切分配置匹配的本地索引。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    if data.get("version") != 1 or data.get("metadata") != metadata:
        return None
    records = data.get("records")
    if not isinstance(records, list) or len(records) != metadata["chunk_count"]:
        return None

    dimension = None
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            return None
        if record.get("chunk_id") != index or record.get("source") != metadata["source"]:
            return None
        if not isinstance(record.get("text"), str):
            return None
        vector = record.get("embedding")
        if not isinstance(vector, list) or not vector:
            return None
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            return None
        if not any(vector):
            return None
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            return None

    return records


def save_index(path: Path, metadata: dict, records: list[dict]) -> None:
    """完整写入临时文件后再替换索引，避免留下半写入的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "metadata": metadata, "records": records}
    temporary_path = path.with_suffix(".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
