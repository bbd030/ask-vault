import hashlib
import os
import unittest

from milvus_store import MilvusVectorStore


@unittest.skipUnless(
    os.getenv("ASKVAULT_MILVUS_TEST") == "1",
    "需要启动 Milvus 并设置 ASKVAULT_MILVUS_TEST=1；测试不调用模型服务",
)
class MilvusIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"chunk_id": 1, "source": "test_policy.txt", "text": "报销需要发票和审批单。", "embedding": [1.0, 0.0]},
            {"chunk_id": 2, "source": "test_policy.txt", "text": "会议在周一举行。", "embedding": [0.0, 1.0]},
        ]
        self.metadata = {
            "embedding_model": "integration-test-only",
            "source": "test_policy.txt",
            "chunk_count": 2,
            "text_sha256": hashlib.sha256("测试版本一".encode()).hexdigest(),
        }
        self.store = self.make_store(self.metadata)
        self.store.sync_records(self.records)
        self.addCleanup(self.store.close)

    def make_store(self, metadata):
        return MilvusVectorStore(
            os.getenv("MILVUS_URI", "http://127.0.0.1:19630"),
            metadata, dimension=2, collection_prefix="ask_vault_test",
        )

    def test_repeated_import_does_not_duplicate_records(self):
        self.assertFalse(self.store.sync_records(self.records))
        rows = self.store.client.query(
            collection_name=self.store.collection_name,
            filter=f'index_key == "{self.store.index_key}"',
            output_fields=["count(*)"], consistency_level="Strong",
        )
        self.assertEqual(rows[0]["count(*)"], 2)

    def test_search_preserves_score_text_and_source(self):
        results = self.store.search([1.0, 0.0])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["record"]["chunk_id"], 1)
        self.assertEqual(results[0]["record"]["source"], "test_policy.txt")
        self.assertEqual(results[0]["record"]["text"], "报销需要发票和审批单。")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)

    def test_updated_document_does_not_mix_old_version(self):
        metadata = {**self.metadata, "text_sha256": hashlib.sha256("测试版本二".encode()).hexdigest()}
        new_records = [
            {"chunk_id": 1, "source": "test_policy.txt", "text": "会议改为周二。", "embedding": [0.0, 1.0]},
            {"chunk_id": 2, "source": "test_policy.txt", "text": "新的报销材料为收据。", "embedding": [1.0, 0.0]},
        ]
        updated_store = self.make_store(metadata)
        self.addCleanup(updated_store.close)
        updated_store.sync_records(new_records)
        new_result = updated_store.search([1.0, 0.0])[0]["record"]
        old_result = self.store.search([1.0, 0.0])[0]["record"]
        self.assertEqual(new_result["text"], "新的报销材料为收据。")
        self.assertEqual(old_result["text"], "报销需要发票和审批单。")


if __name__ == "__main__":
    unittest.main()
