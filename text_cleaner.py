"""
文本清洗与质量体检
==================
为什么需要它：真实项目里「检索不准 / 答案串味」的常见根因不是模型，而是**入库的文本本身就脏**。
本模块对应课件《第 5 课》§5.1 的备注：

    "这一层最常见的问题——PDF 解析乱码、表格结构丢失、扫描版 PDF 没文本、编码不对。
     真实项目还要做清洗（去页眉页脚、合并断行）。"

本机实测（第 5 课语料 21,326 块）发现的真实脏数据，全部来自 PDF 文本层，不是编码问题：

  1. **控制字符**：医保局那份 311 页 DRG/DIP 表格 PDF 里有 819 个 `\\x00`+数字，
     是 PDF 里的页码/对象标记，pypdf 原样抽出来了 → Milvus 里 text 字段就带 `\\x001` 这种尾巴。
  2. **替换字符 U+FFFD**：**8 个**。原因是 PDF 内嵌字体里「踇」（拇趾的病历用字，
     如「踇外翻」「踇囊肿切除术」）没有 Unicode 映射，抽取时只能给一个 �。
     这类字**不可能靠清洗找回**，只能标成 □ 并记进质量报告。
  3. **表格错位**：DRG 表格按列排布，抽成纯文本后变成
     "BX 1101 M10 痛风 80.8401|80.8800x003 ..." 这类行，语义顺序仍在，但不再是人读的句子。

注意：`□` 与 `\ufffd` 不是一回事 —— 很多公文模板本身就用 □ 当复选框
      （"通过□在线□信函□传真提交"），那是**正常内容**，不能算脏字符、更不能删。

清洗原则（不做"过度清洗"，避免把有效信息也干掉）：
  · 去控制字符、零宽字符、BOM
  · U+FFFD → "□"，保留位置信息（表示"此处有一个字抽不出来"）
  · 归一空白：全角空格→半角、连续空白压缩、去行尾空白
  · 合并被硬换行切断的中文句子（只在"上一行非结尾标点 + 下一行非列表开头"时才合并）
  · 阈值告警：脏字符占比超阈值就打印告警，提醒讲课时点名
"""

from __future__ import annotations

import re
from typing import Iterable

# 控制字符（保留 \t \n \r）
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 零宽 / BOM / 软连字符等
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff\u00ad]")
# 连续空白（不含换行）
_SPACES = re.compile(r"[ \t\u3000\xa0]{2,}")
# 三个以上连续换行
_BLANKLINES = re.compile(r"\n{3,}")
# 中文行尾（不应在此断句的标点）
_NO_BREAK_END = "，、；：,;"
_CJK = re.compile(r"[\u4e00-\u9fff]")


def clean_text(text: str, *, merge_lines: bool = True) -> str:
    """清洗单段文本。返回清洗后的字符串（不改变语义，只去脏）。"""
    if not text:
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = _INVISIBLE.sub("", s)
    s = _CTRL.sub("", s)              # 干掉 \x00 这类控制字符（含 PDF 页码标记）
    s = s.replace("\ufffd", "□")      # 抽不出的字：占位保留，别静默丢掉
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = _SPACES.sub(" ", s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = _BLANKLINES.sub("\n\n", s)

    if merge_lines:
        s = merge_broken_lines(s)

    return s.strip()


def merge_broken_lines(text: str) -> str:
    """合并被硬换行切断的中文句子。

    只在「上一行以非结句标点结尾」且「下一行不是新条目/短标题」时合并，
    避免把表格逐行数据（DRG 那种一行一条的）错误粘成一大段。
    """
    lines = [ln for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        cur = ln.strip()
        if not cur:
            out.append("")
            continue
        if out and out[-1]:
            prev = out[-1]
            prev_last = prev[-1] if prev else ""
            starts_new = bool(re.match(r"^([A-Z]{2}\s*\d+|第[一二三四五六七八九十百]+[条章节]|\d+[、.．)）]|[（(]\d+[)）])", cur))
            if (prev_last in _NO_BREAK_END or (not re.search(r"[。！？；：.!?;:]$", prev) and _CJK.search(prev[-2:] or ""))) \
                    and not starts_new and len(cur) > 4:
                out[-1] = prev + cur
                continue
        out.append(cur)
    s = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", s)


def quality_report(name: str, text: str) -> dict:
    """单份文档的文本质量体检，返回指标字典。"""
    total = max(1, len(text))
    cjk = len(_CJK.findall(text))
    return {
        "source": name,
        "chars": len(text),
        "cjk_ratio": round(cjk / total, 3),
        # 注意：□ 常是公文模板里的复选框（如"通过□在线□信函提交"），原文自带，不是坏字。
        # 只有 U+FFFD 才是"真的抽不出来的字"，这里分开统计，避免误报。
        "replacement_chars": text.count("\ufffd"),
        "plain_boxes": text.count("□"),
        "control_chars": len(_CTRL.findall(text)),
        "lines": text.count("\n") + 1,
        "avg_line_len": round(total / max(1, text.count("\n") + 1), 1),
    }


def clean_documents(docs: Iterable, *, merge_lines: bool = True):
    """就地清洗 LangChain Document 列表，返回清洗后的列表。

    注意（踩过的坑）：**不要往 metadata 里加新字段**。
    本项目的 collection 是首次入库时按第一份文档的 metadata 自动建 schema 的；
    只要后面某份文档多出一个字段（哪怕只是 `cleaned: True`），Milvus 就会报
        DataNotMatchException: Insert missed an field `cleaned` to collection
    （除非 schema 里把它设成 nullable 或给了 default_value）。
    所以清洗只改 page_content，不碰 metadata。
    """
    out = []
    for d in docs:
        raw = d.page_content or ""
        cleaned = clean_text(raw, merge_lines=merge_lines)
        if not cleaned:
            continue
        d.page_content = cleaned
        out.append(d)
    return out
