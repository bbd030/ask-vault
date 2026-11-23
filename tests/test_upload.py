import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from server import create_app


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "uploads"
        self.service = Mock()
        self.app = create_app(self.service)
        self.app.config.update(TESTING=True, UPLOAD_FOLDER=self.folder)
        self.client = self.app.test_client()

    def post_file(self, content, filename="制度.txt"):
        return self.client.post("/api/upload", data={"file": (io.BytesIO(content), filename)})

    def assert_no_files(self):
        self.assertFalse(self.folder.exists() and list(self.folder.iterdir()))
        self.service.ask.assert_not_called()

    def test_chinese_bom_and_duplicate_names_preserve_bytes(self):
        content = b"\xef\xbb\xbf" + "公司报销需要发票。".encode("utf-8")
        first = self.post_file(content)
        second = self.post_file(content)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertNotEqual(first.json["document"]["id"], second.json["document"]["id"])
        for response in (first, second):
            document = response.json["document"]
            self.assertTrue(response.json["success"])
            self.assertFalse(document["indexed"])
            self.assertEqual(document["filename"], "制度.txt")
            self.assertEqual(document["size_bytes"], len(content))
            self.assertEqual((self.folder / document["stored_name"]).read_bytes(), content)
        self.service.ask.assert_not_called()

    def test_untrusted_filename_never_controls_save_path(self):
        response = self.post_file(b"hello", "../../outside.TXT")
        self.assertEqual(response.status_code, 201)
        doc = response.json["document"]
        self.assertEqual(doc["filename"], "outside.TXT")
        self.assertRegex(doc["stored_name"], r"^[0-9a-f]{32}\.txt$")
        self.assertEqual(len(list(self.folder.glob("*.txt"))), 1)

    def test_invalid_content_is_rejected_without_saving(self):
        for content, filename in [(b"", "a.txt"), (b" \n\t", "a.txt"),
                                  (b"\xef\xbb\xbf", "a.txt"), (b"\xff", "a.txt"),
                                  (b"a\0b", "a.txt"), (b"hello", "a.pdf"),
                                  (b"hello", "")]:
            with self.subTest(content=content, filename=filename):
                response = self.post_file(content, filename)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json["success"])
                self.assert_no_files()

    def test_missing_wrong_field_and_multiple_files(self):
        for data in [{}, {"other": (io.BytesIO(b"ok"), "a.txt")},
                     {"file": [(io.BytesIO(b"one"), "a.txt"), (io.BytesIO(b"two"), "b.txt")]}]:
            response = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 400)
            self.assert_no_files()
        self.assertEqual(self.client.post("/api/upload", json={}).status_code, 415)

    def test_size_boundary_and_request_limit(self):
        limit = self.app.config["MAX_UPLOAD_BYTES"]
        self.assertEqual(self.post_file(b"a" * limit).status_code, 201)
        count = len(list(self.folder.iterdir()))
        for size in [limit + 1, self.app.config["MAX_CONTENT_LENGTH"] + 1]:
            response = self.post_file(b"a" * size)
            self.assertEqual(response.status_code, 413)
            self.assertFalse(response.json["success"])
            self.assertEqual(len(list(self.folder.iterdir())), count)
        self.service.ask.assert_not_called()

    def test_disk_failure_does_not_expose_details(self):
        with patch("pathlib.Path.mkdir", side_effect=OSError("private-disk-path")):
            response = self.post_file(b"hello")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-disk-path", response.text)
        self.assert_no_files()

    def test_partial_file_is_removed_on_write_failure(self):
        real_open = Path.open

        class FailedWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def write(self, content):
                self.stream.write(content[:1])
                raise OSError("disk full")

            def __exit__(self, *args):
                self.stream.close()

        def failed_open(path, *args, **kwargs):
            return FailedWriter(real_open(path, *args, **kwargs))

        with patch("pathlib.Path.open", failed_open):
            response = self.post_file(b"hello")
        self.assertEqual(response.status_code, 503)
        self.assert_no_files()
