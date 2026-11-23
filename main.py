from knowledge_base import KnowledgeBase, load_text, split_text


if __name__ == "__main__":
    question = input("请输入你的问题：").strip()
    try:
        result = KnowledgeBase().ask(question)
    except Exception as exc:
        print(f"问答失败：{exc}")
        raise SystemExit(1)

    print(f"\n检索方式：{result['backend']}")
    print("\n检索结果：")
    for rank, match in enumerate(result["matches"], start=1):
        print(f"\n--- 第 {rank} 名 ---")
        print(f"相似度：{match['score']:.4f}")
        print(f"来源：{match['source']}，片段 {match['chunk_id']}")
        print(match["text"])
    print("\n回答：")
    print(result["answer"])
    if result["sources"]:
        print("\n提供给模型的参考资料：")
        for source in result["sources"]:
            print(f"[{source['id']}] {source['source']}，片段 {source['chunk_id']}")
