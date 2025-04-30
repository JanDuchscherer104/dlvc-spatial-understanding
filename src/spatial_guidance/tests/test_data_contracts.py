import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from spatial_guidance.pipeline.data_contracts import (
    DataModel,
    DatasetOut,
    DepthEstimationInput,
    DepthEstimationOutput,
    DetectedObject,
    DetectionStageOut,
    EnrichedObject,
    PipelineIn,
    RefinedObject,
    RefinementStageInput,
    RefinementStageOutput,
    SegmentationMask,
    SegmentationStageInput,
    SegmentationStageOutput,
    VisualizationIn,
    VisualizationOut,
)


class TestDataModel(unittest.TestCase):
    def test_empty_from(self):
        """Test that from_ returns an empty instance when no args provided."""
        model = DataModel.from_()
        self.assertIsInstance(model, DataModel)

    def test_from_single_source(self):
        """Test from_ with a single source model."""

        class TestModelA(DataModel):
            field1: str
            field2: int

        class TestModelB(DataModel):
            field1: str
            field3: bool

        source = TestModelA(field1="test", field2=42)
        result = TestModelB.from_(source)

        self.assertEqual(result.field1, "test")
        self.assertFalse(hasattr(result, "field2"))
        self.assertIsNone(getattr(result, "field3", None))

    def test_from_multiple_sources(self):
        """Test from_ with multiple source models."""

        class TestModelC(DataModel):
            field1: Optional[str] = None
            field2: Optional[int] = None
            field3: Optional[bool] = None

        # Create source instances with overlapping fields
        source1 = TestModelC(field1="original", field2=10)
        source2 = TestModelC(field1="updated", field3=True)

        result = TestModelC.from_(source1, source2)

        self.assertEqual(result.field1, "updated")  # Should take the latest value
        self.assertEqual(result.field2, 10)
        self.assertEqual(result.field3, True)

    def test_load_from_artifact_dir(self):
        """Test loading a model from an artifact directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test metadata
            meta = {
                "field1": "string_value",
                "field2": 42,
                "image_field": {"file": "image.png", "type": "pil"},
                "array_field": {"file": "array.npy"},
            }

            # Save metadata
            meta_path = Path(temp_dir) / "data.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            # Create and save a test image
            img = Image.new("RGB", (10, 10), color="red")
            img.save(Path(temp_dir) / "image.png")

            # Create and save a test numpy array
            arr = np.array([1, 2, 3, 4, 5])
            np.save(Path(temp_dir) / "array.npy", arr)

            # Define a test model
            class TestArtifactModel(DataModel):
                field1: str
                field2: int
                image_field: Image
                array_field: np.ndarray

            # Load model from artifact directory
            model = TestArtifactModel.load_from_artifact_dir(temp_dir)

            # Verify loaded data
            self.assertEqual(model.field1, "string_value")
            # Check image and array fields
            self.assertIsInstance(model.image_field, Image.Image)
            self.assertTrue(np.array_equal(model.array_field, arr))


if __name__ == "__main__":
    unittest.main()
