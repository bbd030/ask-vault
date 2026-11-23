import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("两个向量的维度必须相同")

    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("不能计算零向量的余弦相似度")

    return dot_product / (norm_a * norm_b)

def search_chunks(
    records: list[dict],
    query_vector: list[float],
    top_k: int = 2,
) -> list[dict]:
    results = []

    for record in records:
        score = cosine_similarity(query_vector, record["embedding"])

        results.append({
            "score": score,
            "record": record,
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    print("方向相同：", cosine_similarity([1, 0], [1, 0]))
    print("方向垂直：", cosine_similarity([1, 0], [0, 1]))
    print("方向相反：", cosine_similarity([1, 0], [-1, 0]))