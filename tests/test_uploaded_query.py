import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge_base import KnowledgeBase
from server import create_app


class UploadedQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        (self.root / 'data/sample.txt').write_text('默认资料。', encoding='utf-8')
        self.app = create_app(KnowledgeBase(self.root))
        self.app.config.update(TESTING=True, UPLOAD_FOLDER=self.root / 'data/uploads')
        self.client = self.app.test_client()
        env = patch.dict(os.environ, EMBEDDING_MODEL='test-model', VECTOR_BACKEND='json')
        env.start()
        self.addCleanup(env.stop)

    def upload(self, text):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO(text.encode('utf-8-sig')), '同名资料.txt')})
        self.assertEqual(response.status_code, 201)
        return response.json['document']['id']

    def test_same_name_documents_are_isolated_and_cache_survives_switch(self):
        first = self.upload('甲库口令为蓝鲸七号。')
        second = self.upload('乙库口令为白鹭九号。')
        with patch('knowledge_base.embed_text', return_value=[1., 0.]) as embed, patch('knowledge_base.answer_question', return_value={'answer': '测试回答', 'sources': []}):
            for document_id, expected in [(first, '蓝鲸七号'), (second, '白鹭九号'), (first, '蓝鲸七号')]:
                response = self.client.post('/api/query', json={'question': '口令？', 'document_id': document_id})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json['matches'][0]['source'], '同名资料.txt')
                self.assertIn(expected, response.json['matches'][0]['text'])
                self.assertFalse(response.json['matches'][0]['text'].startswith('\ufeff'))
            self.assertEqual(len([c for c in embed.call_args_list if c.kwargs.get('text_type') != 'query']), 2)
            default = self.client.post('/api/query', json={'question': '默认资料？'})
            self.assertEqual(default.json['matches'][0]['source'], 'sample.txt')

    def test_invalid_and_missing_document_never_calls_model(self):
        with patch('knowledge_base.embed_text') as embed:
            for value in [None, '', '../sample.txt', 'A' * 32, 3, True]:
                response = self.client.post('/api/query', json={'question': '内容？', 'document_id': value})
                self.assertEqual(response.status_code, 400)
            response = self.client.post('/api/query', json={'question': '内容？', 'document_id': 'a' * 32})
            self.assertEqual(response.status_code, 404)
            embed.assert_not_called()

    def test_uploaded_document_reaches_milvus_with_document_filter_metadata(self):
        document_id = self.upload('上传测试内容。')
        with patch.dict(os.environ, VECTOR_BACKEND='milvus'), patch('knowledge_base.embed_text', return_value=[1., 0.]), patch('knowledge_base.MilvusVectorStore') as store_class, patch('knowledge_base.answer_question', return_value={'answer': '测试', 'sources': []}):
            store = store_class.return_value
            store.search.return_value = []
            response = self.client.post('/api/query', json={'question': '内容？', 'document_id': document_id})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(store_class.call_args.kwargs['metadata']['document_id'], document_id)
            self.assertIn('上传测试内容', store.sync_records.call_args.args[0][0]['text'])
            store.search.assert_called_once()
            store.close.assert_called_once()
