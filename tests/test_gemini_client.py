import unittest
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np

from spatial_guidance.gemini_client import GeminiClientConfig, GeminiClient
from spatial_guidance.data_contracts.dataset import DatasetOut


class GeminiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("spatial_guidance.gemini_client.gemini_client.genai.Client")
        self.addCleanup(patcher.stop)
        self.mock_client_cls = patcher.start()
        self.mock_client = Mock()
        self.mock_client.models.generate_content.return_value = Mock(text="ok")
        self.mock_client_cls.return_value = self.mock_client

        self.config = GeminiClientConfig()
        self.client = self.config.setup_target()

        self.dataset = DatasetOut(
            rgb_image=Image.new("RGB", (2, 2)),
            depth_image=Image.new("L", (2, 2)),
            camera_intrinsics=np.eye(3),
            camera_pose=np.eye(4),
        )

    def test_add_and_filter_by_tag(self) -> None:
        self.client.add_message("user", "hi", tags=["greet"])
        self.client.add_message("assistant", "hello", tags=["greet"])
        ctx = self.client.get_context(tags=["greet"])
        self.assertEqual(len(ctx), 2)
        self.assertTrue(all("greet" in m.tags for m in ctx))

    def test_generate_response_appends_history(self) -> None:
        text, mode = self.client.generate_response("describe", self.dataset, tags=["scene"])
        self.assertEqual(text, "ok")
        self.assertEqual(len(self.client.chat_history), 2)
        self.mock_client.models.generate_content.assert_called()


if __name__ == "__main__":
    unittest.main()
