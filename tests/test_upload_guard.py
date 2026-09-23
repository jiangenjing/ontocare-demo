"""The public upload route must not store arbitrary or traversal-named files."""
import io
import os
import unittest
from unittest.mock import patch
import app


class UploadGuard(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def upload(self, content, name):
        return self.client.post("/api/upload", data={"image": (io.BytesIO(content), name)},
                                content_type="multipart/form-data")

    def test_fake_png_is_rejected(self):
        response = self.upload(b"not an image", "fake.png")
        self.assertEqual(response.status_code, 400)

    def test_traversal_name_is_replaced_and_temporary_file_removed(self):
        seen = []

        def fake_vlm(path):
            seen.append(path)
            self.assertTrue(os.path.exists(path))
            return None

        with patch.object(app, "call_vlm", side_effect=fake_vlm):
            response = self.upload(b"\xff\xd8\xffdemo", "../../outside.jpg")
        self.assertEqual(response.status_code, 200)
        filename = response.get_json()["filename"]
        self.assertNotIn("..", filename)
        self.assertEqual(os.path.dirname(seen[0]), app.UP_DIR)
        self.assertFalse(os.path.exists(seen[0]))


if __name__ == "__main__":
    unittest.main()
