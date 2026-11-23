import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from knowledge_base import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "sample.txt").write_text("公司报销材料包括发票、审批单和付款凭证。", encoding="utf-8")
        self.environment = patch.dict(os.environ, {"EMBEDDING_MODEL": "test-model", "VECTOR_BACKEND": "json"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.service = KnowledgeBase(self.root)

    def test_cache_is_reused_and_vectors_are_not_returned(self):
        calls = []
        def fake_embed(text, text_type="document"):
            calls.append(text_type)
            return [1.0, 0.0]
        with patch("knowledge_base.embed_text", side_effect=fake_embed), patch("knowledge_base.answer_question", return_value={"answer": "需要发票。[1]", "sources": []}):
            first = self.service.ask("需要什么材料？")
            document_calls = calls.count("document")
            second = self.service.ask("需要什么材料？")
        self.assertGreater(document_calls, 0)
        self.assertEqual(calls.count("document"), document_calls)
        self.assertEqual(calls.count("query"), 2)
        self.assertEqual(first["matches"], second["matches"])
        self.assertNotIn("embedding", first["matches"][0])

    def test_invalid_input_rejected_before_processing(self):
        with patch.object(self.service, "prepare") as prepare:
            for question in [None, 12, "", "  ", "问" * 2001]:
                with self.subTest(question=question), self.assertRaises(ValueError):
                    self.service.ask(question)
            for top_k in [True, 0, 11, 1.5]:
                with self.assertRaises(ValueError):
                    self.service.ask("问题", top_k=top_k)
            prepare.assert_not_called()

    def test_database_connection_closed_on_search_failure(self):
        store = Mock()
        store.search.side_effect = RuntimeError("database unavailable")
        with patch.dict(os.environ, {"VECTOR_BACKEND": "milvus"}), patch("knowledge_base.embed_text", return_value=[1.0, 0.0]), patch("knowledge_base.MilvusVectorStore", return_value=store), patch("knowledge_base.answer_question") as answer:
            with self.assertRaises(RuntimeError):
                self.service.ask("需要什么材料？")
            store.close.assert_called_once()
            answer.assert_not_called()
