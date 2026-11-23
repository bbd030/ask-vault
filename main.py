from generator import answer_question
from embedding import embed_text
import os
from dotenv import load_dotenv
from pathlib import Path
from retriever import search_chunks

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
        print(f"正在向量化片段 {index}/{len(chunks)}……")
        vector = embed_text(chunk)

        record = {
            "chunk_id": index,
            "source": file_path.name,
            "text": chunk,
            "embedding": vector,
        }
        chunk_records.append(record)

    print(f"\n切分结果：共 {len(chunk_records)} 个片段")

    for record in chunk_records:
        print(f"\n--- 片段 {record['chunk_id']} ---")
        print(f"来源：{record['source']}")
        print(f"字符数量：{len(record['text'])}")
        print(f"向量维度：{len(record['embedding'])}")
        print(record["text"])

    question = input("\n请输入你的问题：").strip()

    if not question:
        raise ValueError("问题不能为空")

    query_vector = embed_text(question, text_type="query")

    results = search_chunks(
        records=chunk_records,
        query_vector=query_vector,
        top_k=2,
    )

    print("\n检索结果：")

    for rank, result in enumerate(results, start=1):
        record = result["record"]

        print(f"\n--- 第 {rank} 名 ---")
        print(f"相似度：{result['score']:.4f}")
        print(f"来源：{record['source']}，片段 {record['chunk_id']}")
        print(record["text"])

    print("\n正在根据检索资料生成回答……")
    try:
        answer_result = answer_question(question, results)
    except RuntimeError as exc:
        print(f"\n回答生成失败：{exc}")
    else:
        print("\n回答：")
        print(answer_result["answer"])

        if answer_result["sources"]:
            print("\n提供给模型的参考资料：")
            for source in answer_result["sources"]:
                print(
                    f"[{source['id']}] {source['source']}，"
                    f"片段 {source['chunk_id']}"
                )
