import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fetch_data import download


class FetchDataTests(unittest.TestCase):
    def test_prefix_does_not_read_rest_when_range_is_ignored(self):
        class CountingResponse(io.BytesIO):
            total_read = 0

            def read(self, n=-1):
                data = super().read(n)
                self.total_read += len(data)
                return data

        response = CountingResponse(b"abcd" + b"large archive remainder" * 1000)
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=response):
            destination = Path(temp) / "prefix.bin"
            download("https://example.invalid/file", destination, 4,
                     hashlib.sha256(b"abcd").hexdigest(), prefix=True)
            self.assertEqual(response.total_read, 4)
            self.assertEqual(destination.read_bytes(), b"abcd")

    def test_oversized_download_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=io.BytesIO(b"abcde")):
            destination = Path(temp) / "file.bin"
            with self.assertRaises(ValueError):
                download("https://example.invalid/file", destination, 4, hashlib.sha256(b"abcd").hexdigest())
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".bin.part").exists())

    def test_wrong_hash_does_not_replace_destination(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen", return_value=io.BytesIO(b"abce")):
            destination = Path(temp) / "file.bin"
            with self.assertRaises(ValueError):
                download("https://example.invalid/file", destination, 4, hashlib.sha256(b"abcd").hexdigest())
            self.assertFalse(destination.exists())

    def test_existing_modified_cache_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch("urllib.request.urlopen") as request:
            destination = Path(temp) / "file.bin"
            destination.write_bytes(b"user")
            with self.assertRaises(ValueError):
                download("https://example.invalid/file", destination, 4, hashlib.sha256(b"abcd").hexdigest())
            self.assertEqual(destination.read_bytes(), b"user")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
