"""Data models for pipeline data exchange between stages."""

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="DataModel")


class DataModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PipelineIn(DataModel):
    """Input data for the input stage."""

    idx: int
    user_prompt: Optional[str] = None


class DatasetOut(DataModel):
    """Input data for detection stage."""

    rgb_image: Image
    depth_image: Image
    user_prompt: Optional[str] = None


class DetectedObject(DataModel):
    """Detected object with bounding box and further information by a VLM."""

    aabb_2d: List[int] = Field(
        ..., description="Normalized coordinates [y1, x1, y2, x2] from 0-1000"
    )
    points_2d: List[Tuple[int, int]] = Field(
        ...,
        description="List of at least 3 points on the object surface. Format [(y1, x1), (y1, x2), (y3, x3)] normalized to 0-1000. (not the AABB!)",
    )
    label: str = Field(
        ...,
        description="Concise and unique instance label with distinguishing characteristics (snake_case)",
    )
    approx_distance: float = Field(
        ...,
        description="Approximate distance to the object in meters (rounded to 1 decimal place)",
    )
    height: Literal[
        "floor-level",
        "knee-height",
        "waist-height",
        "chest-height",
        "eye-level",
        "overhead",
    ] = Field(..., description="Height relative to user")
    description: str = Field(..., description="Position relative to user")
    is_hazard: bool = Field(..., description="Whether object poses collision/trip risk")
    is_dynamic: bool = Field(
        ...,
        description="Whether object might be moving (some vehicle, escalator, rotating door) or stationary",
    )
    hazard_type: Optional[Union[Literal["trip", "collision"], str]] = Field(
        None, description="Type of hazard"
    )


class DetectionStageOut(DataModel):
    """Complete analysis of a scene for navigation assistance."""

    objects: List[DetectedObject] = Field(
        ..., description="List of detected objects in the scene"
    )


# ===== SEGMENTATION STAGE I/O MODELS =====


class SegmentationStageInput(DataModel):
    """Input data for segmentation stage."""

    rgb_frame: Any = Field(..., description="RGB frame data")
    depth_frame: Optional[Any] = Field(None, description="Depth frame data")
    user_query: Optional[str] = None


class SegmentationMask(DataModel):
    """Segmentation mask with label and confidence.

    Note: The actual mask data is no longer stored in the Pydantic model
    and should be passed separately.
    """

    label: str = Field(..., description="Label for the segmentation mask")
    confidence: float = Field(1.0, description="Confidence score")
    mask_id: str = Field(
        ..., description="ID to associate with the binary mask stored separately"
    )


class SegmentationStageOutput(DataModel):
    """Output data from segmentation stage."""

    masks: List[SegmentationMask] = Field(
        ..., description="List of segmentation masks identified in the scene"
    )


# ===== DEPTH ESTIMATION STAGE I/O MODELS =====


class DepthEstimationInput(DataModel):
    """Input data for depth estimation stage."""

    depth_frame: Optional[Any] = Field(None, description="Depth frame data")
    rgb_frame: Optional[Any] = Field(None, description="RGB frame data")
    detected_objects: List[DetectedObject]
    segmentation_masks: Optional[List[SegmentationMask]] = None


class EnrichedObject(DetectedObject):
    """Detected object enriched with precise depth information."""

    precise_distance: float = Field(
        ..., description="Precise distance to the object in meters"
    )
    confidence: float = Field(
        1.0, description="Confidence level of the depth estimation (0-1)"
    )


class DepthEstimationOutput(DataModel):
    """Output data from depth estimation stage."""

    objects_with_depth: List[EnrichedObject] = Field(
        ..., description="List of detected objects with depth information"
    )


class VisualizationIn(DataModel):
    """Input for the visualization stage."""

    rgb_image: Image
    depth_image: Image
    detection_output: DetectionStageOut


class VisualizationOut(DataModel):
    """Output from the visualization stage."""

    visualization: Image
    object_count: int


# ===== REFINEMENT STAGE I/O MODELS =====


class RefinementStageInput(DataModel):
    """Input data for refinement stage."""

    rgb_frame: Any = Field(..., description="RGB frame data")
    objects_with_depth: List[EnrichedObject]
    user_query: Optional[str] = None


class RefinedObject(EnrichedObject):
    """Object with refined metadata and priority information."""

    priority_score: float = Field(
        ...,
        description="Priority score for navigation importance (higher is more important)",
    )
    navigation_advice: str = Field(
        ..., description="Specific navigation advice related to this object"
    )


class RefinementStageOutput(DataModel):
    """Final output from the refinement stage."""

    refined_objects: List[RefinedObject] = Field(
        ..., description="List of refined and prioritized objects"
    )
    scene_description: str = Field(
        ..., description="Natural language description of the scene for navigation"
    )
    path_recommendation: Optional[str] = Field(
        None, description="Recommended path through the scene if applicable"
    )


# ===== UNIT TESTS FOR DataModel CLASSES INCLUDING ParFlip USAGE =====

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


class TestDataModel(unittest.TestCase):
    def test_from_no_args_creates_empty_instance(self):
        class TestModel(DataModel):
            a: int = 1
            b: str = "default"

        instance = TestModel.from_()
        self.assertEqual(instance.a, 1)
        self.assertEqual(instance.b, "default")

    def test_from_merges_fields_from_multiple_models(self):
        class TestModel(DataModel):
            a: int
            b: str

        m1 = TestModel(a=1, b="one")
        m2 = TestModel(a=2, b="two")
        merged = TestModel.from_(m1, m2)
        self.assertEqual(merged.a, 2)  # last takes precedence
        self.assertEqual(merged.b, "two")

    def test_from_ignores_fields_not_in_target(self):
        class SourceModel(DataModel):
            a: int
            c: float

        class TargetModel(DataModel):
            a: int
            b: str = "default"

        source = SourceModel(a=5, c=3.14)
        target = TargetModel.from_(source)
        self.assertEqual(target.a, 5)
        self.assertEqual(target.b, "default")

    def test_load_from_artifact_dir_loads_json_and_files(self):
        # Setup temporary directory with data.json and dummy files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Prepare dummy numpy array and save
            np_array = np.array([1, 2, 3])
            np_file = tmp_path / "array.npy"
            np.save(np_file, np_array)

            # Prepare dummy image and save
            img = Image.new("RGB", (10, 10), color="red")
            img_file = tmp_path / "image.png"
            img.save(img_file)

            # Create data.json metadata
            meta = {
                "field1": 123,
                "field2": {"file": "array.npy", "type": "numpy"},
                "field3": {"file": "image.png", "type": "pil"},
            }
            meta_file = tmp_path / "data.json"
            with open(meta_file, "w") as f:
                json.dump(meta, f)

            class TestModel(DataModel):
                field1: int
                field2: np.ndarray
                field3: Image.Image

            loaded = TestModel.load_from_artifact_dir(tmp_path)
            self.assertEqual(loaded.field1, 123)
            np.testing.assert_array_equal(loaded.field2, np_array)
            self.assertIsInstance(loaded.field3, Image.Image)
            self.assertEqual(loaded.field3.size, (10, 10))

    def test_load_from_artifact_dir_handles_missing_file_type_assumed_pil(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img = Image.new("RGB", (5, 5), color="blue")
            img_file = tmp_path / "pic.png"
            img.save(img_file)

            meta = {"image_field": {"file": "pic.png"}}
            meta_file = tmp_path / "data.json"
            with open(meta_file, "w") as f:
                json.dump(meta, f)

            class TestModel(DataModel):
                image_field: Image.Image

            loaded = TestModel.load_from_artifact_dir(tmp_path)
            self.assertIsInstance(loaded.image_field, Image.Image)
            self.assertEqual(loaded.image_field.size, (5, 5))


class TestParFlipUsage(unittest.TestCase):
    """Example tests demonstrating ParFlip usage for isolated tests."""

    def test_parflip_isolation(self):
        # ParFlip is a tool to run tests in isolated processes.
        # Here we demonstrate its usage by running a simple test function.

        import parflip

        def test_func():
            # This function runs in isolated process
            self.assertEqual(1 + 1, 2)

        result = parflip.run(test_func)
        self.assertTrue(result.success)

    def test_parflip_with_data_model(self):
        import parflip

        def test_data_model_creation():
            obj = PipelineIn(idx=42, user_prompt="test")
            assert obj.idx == 42
            assert obj.user_prompt == "test"

        result = parflip.run(test_data_model_creation)
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
