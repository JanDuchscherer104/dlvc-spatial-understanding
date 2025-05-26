"""
Unit tests for testing the AABBDetection and AABBDetections classes,
specifically their ability to process segmentation masks from VLM outputs.
"""

import base64
import io
import json
import unittest
from typing import Any, Dict, List

import numpy as np

# For manual parsing
from langchain_core.output_parsers import PydanticOutputParser
from PIL import Image

from spatial_guidance.pipeline.data_contracts import AABBDetection, AABBDetections


class TestAABBProcessing(unittest.TestCase):
    """Test suite for AABBDetection and AABBDetections processing capabilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small test image to use as a mask
        self.test_img_width = 100
        self.test_img_height = 80
        self.test_mask_img = Image.new("L", (20, 20), color=255)

        # Convert the image to base64 string
        img_byte_arr = io.BytesIO()
        self.test_mask_img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        self.base64_mask = (
            f"data:image/png;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
        )

        # Create a sample VLM output string with normalized coordinates
        # Format: [y1, x1, y2, x2] in range 0-1000
        self.sample_data = {
            "objects": [
                {
                    "label": "chair",
                    "approx_distance": 2.5,
                    "height": "waist-height",
                    "is_hazard": False,
                    "is_dynamic": False,
                    "hazard_type": None,
                    "is_crowd": False,
                    "is_moving": False,
                    "aabb_2d": [200, 300, 700, 600],  # normalized [y1, x1, y2, x2]
                    "segmentation_mask": self.base64_mask,
                },
                {
                    "label": "table",
                    "approx_distance": 3.0,
                    "height": "waist-height",
                    "is_hazard": False,
                    "is_dynamic": False,
                    "hazard_type": None,
                    "is_crowd": False,
                    "is_moving": False,
                    "aabb_2d": [100, 100, 500, 400],  # normalized [y1, x1, y2, x2]
                    "segmentation_mask": self.base64_mask,
                },
            ]
        }
        self.sample_json = json.dumps(self.sample_data)

    def test_output_schema(self):
        parser = PydanticOutputParser(pydantic_object=AABBDetections)
        assert isinstance(
            parser.get_format_instructions(), str
        ), "Output schema should be a string"

    def test_parser_with_raw_masks(self):
        """Test that our custom parser correctly parses the sample data."""
        # Parse the sample JSON using our custom parser
        detections = PydanticOutputParser(pydantic_object=AABBDetections).parse(
            self.sample_json
        )

        # Verify the parsed output
        self.assertIsInstance(detections, AABBDetections)
        self.assertEqual(len(detections.objects), 2)

        # Check first object
        chair = detections.objects[0]
        self.assertEqual(chair.label, "chair")
        self.assertEqual(chair.approx_distance, 2.5)
        self.assertEqual(chair.height, "waist-height")

        # Verify aabb_2d is a numpy array with the correct values
        self.assertIsInstance(chair.box_2d, np.ndarray)
        np.testing.assert_array_equal(chair.box_2d, np.array([200, 300, 700, 600]))

        # Verify that mask is a PIL Image (thanks to the validator)
        self.assertIsInstance(chair.mask, Image.Image)

        # Check that the mask hasn't been processed yet
        self.assertEqual(chair.processed_, False)

    def test_process_masks(self):
        """Test the processing of masks and bounding boxes."""
        # Parse the sample JSON using our custom parser
        detections = PydanticOutputParser(pydantic_object=AABBDetections).parse(
            self.sample_json
        )

        # Process all masks
        detections.process_all(self.test_img_height, self.test_img_width)

        # Check first object
        chair = detections.objects[0]

        # Verify that the mask has been processed
        self.assertEqual(chair.processed_, True)
        self.assertIsInstance(chair.mask, Image.Image)

        # Verify that mask dimensions match the image dimensions
        self.assertEqual(chair.mask.width, self.test_img_width)
        self.assertEqual(chair.mask.height, self.test_img_height)

        # Convert to numpy array to check where mask values are non-zero
        mask_array = np.array(chair.mask)

        # Calculate expected bounding box in absolute pixel values
        y0, x0, y1, x1 = 200, 300, 700, 600  # Normalized [0-1000]
        abs_y0 = int(y0 / 1000 * self.test_img_height)
        abs_x0 = int(x0 / 1000 * self.test_img_width)
        abs_y1 = int(y1 / 1000 * self.test_img_height)
        abs_x1 = int(x1 / 1000 * self.test_img_width)

        # Check that pixels outside the bounding box are all zero
        self.assertTrue((mask_array[:abs_y0, :] == 0).all())  # Above the box
        self.assertTrue((mask_array[abs_y1:, :] == 0).all())  # Below the box
        self.assertTrue((mask_array[:, :abs_x0] == 0).all())  # Left of the box
        self.assertTrue((mask_array[:, abs_x1:] == 0).all())  # Right of the box

        # Check that there are some non-zero pixels inside the box
        # (at least one pixel should be non-zero)
        self.assertTrue((mask_array[abs_y0:abs_y1, abs_x0:abs_x1] > 0).any())


if __name__ == "__main__":
    unittest.main()
