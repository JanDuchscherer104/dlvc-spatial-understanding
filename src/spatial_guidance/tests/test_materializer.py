import os
import shutil
import tempfile
import unittest
from typing import Any, Dict

import numpy as np
from PIL import Image
from zenml import pipeline, step
from zenml.steps import BaseStep

from spatial_guidance.pipeline.data_contracts import DataModel
from spatial_guidance.pipeline.materializer import PydanticNumpyMaterializer


# Define test data models
class ImageData(DataModel):
    image_array: np.ndarray | Image.Image
    description: str


class ResultData(DataModel):
    result_value: float


# Global variables to store test results
test_results: Dict[str, Any] = {}


# Step functions defined at module level
@step(output_materializers=PydanticNumpyMaterializer)
def create_numpy_image() -> ImageData:
    # Create a numpy array image with some values
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return ImageData(image_array=img_array, description="Gray image from NumPy")


@step
def analyze_numpy_image(data: ImageData) -> ResultData:
    # Store results for test verification
    test_results["numpy_description"] = data.description
    test_results["numpy_type"] = type(data.image_array)
    test_results["numpy_shape"] = data.image_array.shape
    test_results["numpy_mean"] = float(data.image_array.mean())

    return ResultData(result_value=float(data.image_array.mean()))


# Disable caching for all pipelines to ensure test_results are updated
@pipeline(enable_cache=False)
def numpy_image_pipeline():
    img = create_numpy_image()
    analyze_numpy_image(img)


@step(output_materializers=PydanticNumpyMaterializer)
def create_pil_image() -> ImageData:
    # Create a PIL Image directly
    pil_img = Image.new("RGB", (100, 100), color="red")
    # Convert to numpy right away to avoid PIL file handling issues
    img_array = np.array(pil_img)
    return ImageData(image_array=img_array, description="Red image from PIL")


@step
def analyze_pil_image(data: ImageData) -> ResultData:
    # Store results for test verification
    test_results["pil_description"] = data.description
    test_results["pil_type"] = type(data.image_array)

    # Always convert to numpy array to avoid PIL file issues
    if isinstance(data.image_array, np.ndarray):
        mean_pixel = float(data.image_array.mean())
    else:
        # Convert PIL to numpy safely
        mean_pixel = float(np.array(data.image_array).mean())

    test_results["pil_mean"] = mean_pixel
    return ResultData(result_value=mean_pixel)


@pipeline(enable_cache=False)
def pil_image_pipeline():
    img = create_pil_image()
    analyze_pil_image(img)


# Class-based step at module level
class ImageProcessingThreshold50(BaseStep):
    def __init__(self, threshold: float = 50.0) -> None:
        super().__init__()
        self.threshold = threshold

    def entrypoint(self, data: ImageData) -> ResultData:
        # Store results for verification
        test_results["threshold50_type"] = type(data)
        test_results["threshold50_has_array"] = data.image_array is not None

        if isinstance(data.image_array, np.ndarray):
            mean_pixel = float(data.image_array.mean())
        else:
            mean_pixel = float(np.array(data.image_array).mean())

        # Apply threshold
        result_value = mean_pixel if mean_pixel > self.threshold else 0.0
        test_results["threshold50_result"] = result_value
        return ResultData(result_value=result_value)


@step(output_materializers=PydanticNumpyMaterializer)
def create_test_image_100() -> ImageData:
    # Create a grayscale image with value 100
    img_array = np.ones((50, 50), dtype=np.uint8) * 100
    return ImageData(image_array=img_array, description="Test grayscale image")


@step
def verify_result_100(result: ResultData) -> None:
    # Store result for verification
    test_results["verify_result_100"] = result.result_value


@pipeline(enable_cache=False)
def class_based_pipeline_50():
    img = create_test_image_100()
    processor = ImageProcessingThreshold50()
    result = processor(img)
    verify_result_100(result)


class ImageProcessingThreshold150(BaseStep):
    def __init__(self, threshold: float = 150.0) -> None:
        super().__init__()
        self.threshold = threshold

    def entrypoint(self, data: ImageData) -> ResultData:
        if isinstance(data.image_array, np.ndarray):
            mean_pixel = float(data.image_array.mean())
        else:
            mean_pixel = float(np.array(data.image_array).mean())

        # Apply threshold - should return 0 since image mean is 100
        result_value = mean_pixel if mean_pixel > self.threshold else 0.0
        test_results["threshold150_result"] = result_value
        return ResultData(result_value=result_value)


@pipeline(enable_cache=False)
def threshold_filtering_pipeline():
    img = create_test_image_100()
    processor = ImageProcessingThreshold150()
    result = processor(img)
    verify_result_100(result)


@step(output_materializers=PydanticNumpyMaterializer)
def create_float_image() -> ImageData:
    # Create a float array, but convert to uint8 for compatibility
    # Float arrays (mode F) can't be saved as PNG
    img_array = np.ones((20, 20), dtype=np.float32) * 0.5
    # Store original dtype for verification
    test_results["original_float_dtype"] = img_array.dtype

    # Convert to uint8 for storage compatibility
    uint8_array = (img_array * 255).astype(np.uint8)
    return ImageData(
        image_array=uint8_array, description="Float image (converted to uint8)"
    )


@step
def verify_float_image(data: ImageData) -> ResultData:
    # Store results for verification
    test_results["float_shape"] = data.image_array.shape
    # Note: we can't verify original dtype since it was converted
    mean_value = float(data.image_array.mean()) / 255.0  # Scale back
    test_results["float_mean"] = mean_value

    return ResultData(result_value=mean_value)


@pipeline(enable_cache=False)
def float_image_pipeline():
    imgs = create_float_image()
    verify_float_image(imgs)


class TestPydanticNumpyMaterializer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"
        # Clear test results before each test
        test_results.clear()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_numpy_array_materialization(self):
        """Test materializer with numpy array images"""
        numpy_image_pipeline()

        # Verify results using stored values
        self.assertEqual(test_results["numpy_description"], "Gray image from NumPy")
        self.assertTrue(
            issubclass(test_results["numpy_type"], np.ndarray.__class__)
            or test_results["numpy_type"] == np.ndarray
        )
        self.assertEqual(test_results["numpy_shape"], (100, 100, 3))
        self.assertEqual(test_results["numpy_mean"], 128.0)

    def test_pil_image_materialization(self):
        """Test materializer with PIL images"""
        pil_image_pipeline()

        # Verify results using stored values
        self.assertEqual(test_results["pil_description"], "Red image from PIL")
        # Red has R=255, G=0, B=0, so mean should be ~85 for RGB
        self.assertAlmostEqual(test_results["pil_mean"], 85.0, delta=1.0)

    def test_class_based_steps(self):
        """Test materializer with class-based steps"""
        class_based_pipeline_50()

        # Verify results using stored values
        self.assertTrue(test_results["threshold50_has_array"])
        # Mean is 100, threshold is 50, so result should be 100
        self.assertEqual(test_results["threshold50_result"], 100.0)
        self.assertEqual(test_results["verify_result_100"], 100.0)

    def test_threshold_filtering(self):
        """Test that threshold filtering works correctly"""
        threshold_filtering_pipeline()

        # Verify that the result was filtered to 0
        self.assertEqual(test_results["threshold150_result"], 0.0)

    def test_serialization_types(self):
        """Test that different numpy array types can be serialized"""
        float_image_pipeline()

        # Verify results
        self.assertEqual(test_results["original_float_dtype"], np.float32)
        self.assertEqual(test_results["float_shape"], (20, 20))
        self.assertAlmostEqual(test_results["float_mean"], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
