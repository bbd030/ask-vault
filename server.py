import json
import re
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_file, abort
from werkzeug.exceptions import HTTPException

from document_loader import document_suffix, extract_document_text
from knowledge_base import KnowledgeBase


def create_app(knowledge_base=None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    # 上传请求为 multipart 边界和请求头额外预留 64 KiB。
    app.config["MAX_UPLOAD_BYTES"] = 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 + 64 * 1024
    app.config["UPLOAD_FOLDER"] = Path(__file__).resolve().parent / "data" / "uploads"
    service = knowledge_base if knowledge_base is not None else KnowledgeBase()

    @app.before_request
    def limit_request_size():
        # 问答等接口仍保持原来的 64 KiB 请求上限。
        if request.endpoint != "upload":
            request.max_content_length = 64 * 1024

    @app.post("/api/upload")
    def upload():
        if request.mimetype != "multipart/form-data":
            return jsonify({"success": False, "error": "请使用 multipart/form-data 上传文件"}), 415
        files = list(request.files.items(multi=True))
        if len(files) != 1 or files[0][0] != "file" or not files[0][1].filename:
            return jsonify({"success": False, "error": "请通过 file 字段上传一个 TXT、PDF 或 Word (.docx) 文件"}), 400
        uploaded = files[0][1]
        # 原始名称仅用于显示，绝不直接参与保存路径。
        filename = uploaded.filename.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            suffix = document_suffix(filename)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        content = uploaded.stream.read(app.config["MAX_UPLOAD_BYTES"] + 1)
        if len(content) > app.config["MAX_UPLOAD_BYTES"]:
            return jsonify({"success": False, "error": "文件不能超过 1 MiB"}), 413
        if suffix != ".txt":
            try:
                extract_document_text(content, suffix)
            except ValueError as exc:
                return jsonify({"success": False, "error": str(exc)}), 400
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                return jsonify({"success": False, "error": "TXT 文件必须使用 UTF-8 编码"}), 400
            if not text.strip() or "\0" in text:
                return jsonify({"success": False, "error": "TXT 文件须包含非空文本，且不能含空字符"}), 400

        document_id = uuid4().hex
        stored_name = f"{document_id}{suffix}"
        destination = Path(app.config["UPLOAD_FOLDER"]) / stored_name
        created = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                created = True
                output.write(content)
            # 保存原始名称，后续问答来源不使用随机磁盘文件名。
            destination.with_suffix(".json").write_text(
                json.dumps({"filename": filename}, ensure_ascii=False), encoding="utf-8",
            )
        except OSError as exc:
            if created:
                destination.unlink(missing_ok=True)
                destination.with_suffix(".json").unlink(missing_ok=True)
            app.logger.error("保存上传文件失败，异常类型：%s", type(exc).__name__)
            return jsonify({"success": False, "error": "文件保存失败，请稍后重试"}), 503
        return jsonify({
            "success": True,
            "document": {"id": document_id, "filename": filename,
                         "stored_name": stored_name, "size_bytes": len(content),
                         "indexed": False},
            "message": "文件已保存，可在提问时指定文档编号；首次提问时建立索引",
        }), 201

    @app.get("/api/documents")
    def list_documents():
        # 列表只读取原文件和名称信息，不调用模型、不连接 Milvus。
        folder = Path(app.config["UPLOAD_FOLDER"])
        documents = []
        try:
            try:
                entries = list(folder.iterdir())
            except FileNotFoundError:
                entries = []
            for info_path in entries:
                if info_path.suffix != ".json":
                    continue
                document_id = info_path.stem
                if not re.fullmatch(r"[0-9a-f]{32}", document_id):
                    continue  # 跳过 .index.json 等非文档名称文件。
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    filename = info.get("filename") if isinstance(info, dict) else None
                    if not isinstance(filename, str) or not filename.strip():
                        continue
                    file_path = folder / f"{document_id}{document_suffix(filename)}"
                    stat = file_path.stat()
                    if not file_path.is_file():
                        continue
                except (FileNotFoundError, ValueError):
                    continue  # 不完整或损坏的文档不影响其他资料显示。
                documents.append({
                    "id": document_id, "filename": filename,
                    "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
                })
        except OSError as exc:
            app.logger.error("读取文档列表失败，异常类型：%s", type(exc).__name__)
            return jsonify({"success": False, "error": "暂时无法读取文档列表，请稍后重试"}), 503
        documents.sort(key=lambda item: (item["modified_ns"], item["id"]), reverse=True)
        for document in documents:
            document.pop("modified_ns")
        return jsonify({"success": True, "documents": documents})

    def uploaded_document(document_id):
        if not re.fullmatch(r"[0-9a-f]{32}", document_id):
            abort(404)
        folder = Path(app.config["UPLOAD_FOLDER"])
        info_path = folder / f"{document_id}.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            filename = info.get("filename") if isinstance(info, dict) else None
            if not isinstance(filename, str):
                abort(404)
            file_path = folder / f"{document_id}{document_suffix(filename)}"
        except (FileNotFoundError, ValueError):
            abort(404)
        if not file_path.is_file():
            abort(404)
        return info_path, file_path, filename

    @app.get("/api/documents/<document_id>/download")
    def download_document(document_id):
        _, file_path, filename = uploaded_document(document_id)
        return send_file(file_path.resolve(), as_attachment=True, download_name=filename,
                         mimetype="application/octet-stream", max_age=0)

    @app.get("/api/health")
    def health():
        # 这里只检查 HTTP 服务存活，不触发模型调用或数据库连接。
        return jsonify({"status": "ok", "service": "AskVault"})

    @app.post("/api/query")
    def query():
        if not request.is_json:
            return jsonify({"success": False, "error": "请使用 application/json 请求体"}), 415
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "请求体必须是有效的 JSON 对象"}), 400
        question = data.get("question")
        top_k = data.get("top_k", 2)
        if not isinstance(question, str) or not question.strip():
            return jsonify({"success": False, "error": "question 必须是非空字符串"}), 400
        if len(question.strip()) > 2000:
            return jsonify({"success": False, "error": "问题不能超过 2000 个字符"}), 400
        if type(top_k) is not int or not 1 <= top_k <= 10:
            return jsonify({"success": False, "error": "top_k 必须是 1 到 10 之间的整数"}), 400
        document_id = data.get("document_id")
        if "document_id" in data and (not isinstance(document_id, str) or not re.fullmatch(r"[0-9a-f]{32}", document_id)):
            return jsonify({"success": False, "error": "document_id 必须是 32 位小写十六进制编号"}), 400
        document_ids = data.get("document_ids")
        if "document_ids" in data:
            if "document_id" in data or not isinstance(document_ids, list) or not document_ids:
                return jsonify({"success": False, "error": "请仅指定一份资料或一组资料编号"}), 400
            if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{32}", item) for item in document_ids):
                return jsonify({"success": False, "error": "资料编号格式不正确"}), 400
            if len(document_ids) != len(set(document_ids)):
                return jsonify({"success": False, "error": "资料编号不能重复"}), 400
        try:
            if document_ids is not None:
                result = service.ask(question.strip(), top_k=top_k, document_ids=document_ids)
            elif document_id is None:
                result = service.ask(question.strip(), top_k=top_k)
            else:
                result = service.ask(question.strip(), top_k=top_k, document_id=document_id)
        except FileNotFoundError:
            return jsonify({"success": False, "error": "问答资料不存在，请重新上传或选择资料"}), 404
        except Exception as exc:
            app.logger.error("知识库问答失败，异常类型：%s", type(exc).__name__)
            return jsonify({"success": False, "error": "问答服务暂不可用，请检查数据库和模型配置"}), 503
        return jsonify({"success": True, **result})

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {404: "接口不存在", 405: "请求方法不支持", 413: "请求内容过大"}
        return jsonify({"success": False, "error": messages.get(error.code, error.name)}), error.code

    return app


if __name__ == "__main__":
    create_app().run(
        host="127.0.0.1", port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False, use_reloader=False, threaded=False,
    )
