import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from server import create_app


class DocumentListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'uploads'
        self.service = Mock()
        self.app = create_app(self.service)
        self.app.config.update(TESTING=True, UPLOAD_FOLDER=self.folder)
        self.client = self.app.test_client()

    def upload(self):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO('报销测试资料。'.encode()), '同名.txt')})
        self.assertEqual(response.status_code, 201)
        return response.json['document']

    def test_missing_folder_is_empty_and_read_does_not_create_it(self):
        response = self.client.get('/api/documents')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'success': True, 'documents': []})
        self.assertFalse(self.folder.exists())
        self.service.ask.assert_not_called()

    def test_same_name_uploads_survive_app_restart_and_hide_contents(self):
        first = self.upload()
        second = self.upload()
        other_app = create_app(self.service)
        other_app.config['UPLOAD_FOLDER'] = self.folder
        response = other_app.test_client().get('/api/documents')
        self.assertEqual(response.status_code, 200)
        docs = response.json['documents']
        self.assertEqual({doc['id'] for doc in docs}, {first['id'], second['id']})
        self.assertEqual(len(docs), 2)
        for doc in docs:
            self.assertEqual(set(doc), {'id', 'filename', 'size_bytes'})
            self.assertEqual(doc['filename'], '同名.txt')
            self.assertEqual(doc['size_bytes'], len('报销测试资料。'.encode()))
        self.assertNotIn('报销测试资料', response.text)
        self.assertNotIn(str(self.folder), response.text)
        self.service.ask.assert_not_called()

    def test_invalid_entries_and_cache_do_not_pollute_list(self):
        valid = self.upload()
        (self.folder / f"{valid['id']}.index.json").write_text('{"metadata":{}}')
        (self.folder / ('b' * 32 + '.json')).write_text('{broken')
        (self.folder / ('c' * 32 + '.json')).write_text(json.dumps({'filename': 'missing.txt'}))
        (self.folder / ('d' * 32 + '.json')).write_text('[]')
        (self.folder / ('e' * 32 + '.json')).write_bytes(b'\xff')
        (self.folder / 'legacy.txt').write_text('legacy')
        response = self.client.get('/api/documents')
        self.assertEqual([doc['id'] for doc in response.json['documents']], [valid['id']])

    def test_documents_sort_newest_first(self):
        import os
        first = self.upload()
        second = self.upload()
        os.utime(self.folder / f"{first['id']}.txt", ns=(1000000000, 1000000000))
        os.utime(self.folder / f"{second['id']}.txt", ns=(2000000000, 2000000000))
        docs = self.client.get('/api/documents').json['documents']
        self.assertEqual([doc['id'] for doc in docs], [second['id'], first['id']])

    def test_unreadable_folder_returns_safe_json(self):
        with patch('pathlib.Path.iterdir', side_effect=PermissionError('private-path')):
            response = self.client.get('/api/documents')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json['success'])
        self.assertNotIn('private-path', response.text)
