from io import BytesIO
from zipfile import BadZipFile, ZipFile


def document_suffix(filename: str) -> str:
    """将显示名称映射到允许的磁盘后缀。"""
    lower_name = filename.lower()
    for suffix in (".txt", ".pdf", ".docx"):
        if lower_name.endswith(suffix):
            return suffix
    raise ValueError("目前只支持 .txt、.pdf 和 .docx 文件")


def extract_pdf_text(content: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("请先解除 PDF 加密再上传")
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, NotImplementedError) as exc:
        raise ValueError("无法读取 PDF，请确认文件完整且未加密") from exc

    text = "\n\n".join(pages).replace("\x00", "").strip()
    if not text:
        raise ValueError("PDF 中没有可提取的文字，请先对扫描件做 OCR")
    return text


def extract_docx_text(content: bytes) -> str:
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError
    from docx.text.paragraph import Paragraph
    from lxml.etree import XMLSyntaxError

    stream = BytesIO(content)
    try:
        with ZipFile(stream) as archive:
            expanded_size = sum(entry.file_size for entry in archive.infolist())
    except BadZipFile as exc:
        raise ValueError("无法读取 Word 文档，请上传完整的 .docx 文件") from exc

    if expanded_size > 32 * 1024 * 1024:
        raise ValueError("Word 文档解压后的大小不能超过 32 MiB")
    try:
        stream.seek(0)
        document = Document(stream)
    except (BadZipFile, PackageNotFoundError, XMLSyntaxError, KeyError, RuntimeError, ValueError) as exc:
        raise ValueError("无法读取 Word 文档，请上传完整的 .docx 文件") from exc

    blocks = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            blocks.append(block.text)
        else:
            rows = []
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            blocks.append("\n".join(rows))
    text = "\n\n".join(block for block in blocks if block.strip()).strip()
    if not text:
        raise ValueError("Word 文档中没有可提取的正文或表格文字")
    return text


def extract_document_text(content: bytes, suffix: str) -> str:
    if suffix == ".pdf":
        return extract_pdf_text(content)
    if suffix == ".docx":
        return extract_docx_text(content)
    return content.decode("utf-8-sig")
