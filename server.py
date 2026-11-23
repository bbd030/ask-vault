import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from knowledge_base import KnowledgeBase


def create_app(knowledge_base=None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    service = knowledge_base if knowledge_base is not None else KnowledgeBase()

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
        try:
            result = service.ask(question.strip(), top_k=top_k)
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
