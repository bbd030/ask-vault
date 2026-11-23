import unittest
from unittest.mock import Mock

from server import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.ask.return_value = {
            "question": "报销需要哪些材料？", "answer": "需要发票。[1]",
            "sources": [{"id": 1, "source": "sample.txt", "chunk_id": 2}],
            "matches": [{"score": 0.9, "source": "sample.txt", "chunk_id": 2, "text": "需要发票。"}],
            "backend": "milvus",
        }
        self.app = create_app(self.service)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_health_does_not_call_question_service(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok", "service": "AskVault"})
        self.service.ask.assert_not_called()

    def test_query_returns_answer_and_sources(self):
        response = self.client.post("/api/query", json={"question": "  报销需要哪些材料？  ", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["success"])
        self.assertEqual(response.json["sources"][0]["chunk_id"], 2)
        self.assertEqual(response.json["matches"][0]["score"], 0.9)
        self.service.ask.assert_called_once_with("报销需要哪些材料？", top_k=2)

    def test_invalid_questions_and_top_k_do_not_call_model(self):
        invalid = [{}, {"question": " "}, {"question": 12}, {"question": "问" * 2001}]
        invalid += [{"question": "材料？", "top_k": value} for value in [0, 11, True, 1.5, "2", None]]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/api/query", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json["success"])
        self.service.ask.assert_not_called()

    def test_wrong_content_type_and_malformed_json(self):
        self.assertEqual(self.client.post("/api/query", data="hello").status_code, 415)
        for data in ["{broken", "[]", "null"]:
            response = self.client.post("/api/query", data=data, content_type="application/json")
            self.assertEqual(response.status_code, 400)
        self.service.ask.assert_not_called()

    def test_upstream_failure_returns_no_internal_details(self):
        self.service.ask.side_effect = RuntimeError("private-provider-error sk-test-secret")
        response = self.client.post("/api/query", json={"question": "材料？"})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json["success"])
        self.assertNotIn("sk-test-secret", response.text)
        self.assertNotIn("private-provider-error", response.text)

    def test_http_errors_are_json(self):
        self.assertEqual(self.client.get("/missing").status_code, 404)
        self.assertIsNotNone(self.client.get("/missing").json)
        self.assertEqual(self.client.get("/api/query").status_code, 405)
        response = self.client.post("/api/query", data='{"question":"' + 'x' * 70000 + '"}', content_type="application/json")
        self.assertEqual(response.status_code, 413)
        self.service.ask.assert_not_called()
