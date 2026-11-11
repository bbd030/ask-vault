import os
from pathlib import Path

import dashscope
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


def embed_text(text: str) -> list[float]:
    response = dashscope.TextEmbedding.call(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=os.environ["DASHSCOPE_API_KEY"],
        input=[text],
        text_type="document",
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"向量化失败：{response.code} - {response.message}"
        )

    return response.output["embeddings"][0]["embedding"]


if __name__ == "__main__":
    text = "报销材料包括发票、审批单和付款凭证。"
    vector = embed_text(text)

    print(f"原始文字：{text}")
    print(f"向量维度：{len(vector)}")
    print(f"前 5 个数值：{vector[:5]}")