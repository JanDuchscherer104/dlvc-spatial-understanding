# tests for DataModel.load_from_artifact_dir
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_guidance.pipeline.data_contracts import DataModel


def setup_dummy_artifact(base_dir, dummy_class, arr, img, value):
    artifact = Path(base_dir) / "artifact"
    artifact.mkdir()
    # Write data.json
    meta = {
        "__class__": dummy_class.__name__,
        "__module__": dummy_class.__module__,
        "arr": {
            "file": "arr.npy",
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "type": "numpy",
        },
        "img": {"file": "img.png", "mode": img.mode, "size": img.size, "type": "pil"},
        "value": value,
    }
    with open(artifact / "data.json", "w") as f:
        json.dump(meta, f)
    # Save array and image files
    np.save(artifact / "arr.npy", arr)
    img.save(artifact / "img.png")
    return artifact


class TestDataModelLoad(unittest.TestCase):
    def test_load_from_artifact_dir(self):
        # Create temporary directory for artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            # Define a dummy DataModel subclass
            class DummyModel(DataModel):
                arr: np.ndarray
                img: Image.Image
                value: int

            arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
            img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            value = 42

            artifact = setup_dummy_artifact(tmpdir, DummyModel, arr, img, value)
            loaded = DummyModel.load_from_artifact_dir(str(artifact))
            self.assertIsInstance(loaded, DummyModel)
            # Assert numpy array loaded correctly
            self.assertTrue(np.array_equal(loaded.arr, arr))
            # Assert PIL Image loaded and has correct size
            self.assertIsInstance(loaded.img, Image.Image)
            self.assertEqual(loaded.img.size, img.size)
            # Assert scalar field
            self.assertEqual(loaded.value, value)


if __name__ == "__main__":
    unittest.main()
