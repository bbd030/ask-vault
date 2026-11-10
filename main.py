from pathlib import Path


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    file_path = project_dir / "data" / "sample.txt"

    text = load_text(file_path)

    print(f"文档来源：{file_path.name}")
    print(f"字符数量：{len(text)}")
    print("文档内容：")
    print(text)