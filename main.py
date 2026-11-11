import os
from dotenv import load_dotenv
from pathlib import Path


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

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")

    embedding_model = os.getenv("EMBEDDING_MODEL")
    api_key = os.getenv("DASHSCOPE_API_KEY")

    print(f"Embedding 模型：{embedding_model}")
    print("API Key 状态：", "已配置" if api_key else "未配置")
    file_path = project_dir / "data" / "sample.txt"

    text = load_text(file_path)

    print(f"文档来源：{file_path.name}")
    print(f"字符数量：{len(text)}")
    print("文档内容：")
    print(text)

    chunks = split_text(text)
    chunk_records = []

    for index, chunk in enumerate(chunks, start=1):
        record = {
            "chunk_id": index,
            "source": file_path.name,
            "text": chunk,
        }
        chunk_records.append(record)

    print(f"\n切分结果：共 {len(chunk_records)} 个片段")

    for record in chunk_records:
        print(f"\n--- 片段 {record['chunk_id']} ---")
        print(f"来源：{record['source']}")
        print(f"字符数量：{len(record['text'])}")
        print(record["text"])