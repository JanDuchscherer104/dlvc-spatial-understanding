import json
from pathlib import Path
from typing import Type

import numpy as np
from PIL import Image
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from .data_contracts import DataModel


class PydanticNumpyMaterializer(BaseMaterializer):
    """Custom materializer for DataModel objects (Pydantic models with images)."""

    ASSOCIATED_TYPES = (DataModel,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def _is_image_array(self, array: np.ndarray) -> bool:
        """
        Determine whether a numpy array should be treated as an image.
        For example, a 2D array (grayscale) or a 3D array with last dim 1, 3, or 4.
        """
        if not isinstance(array, np.ndarray):
            return False
        if array.ndim == 2 or array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return True
        return False

    def save(self, data: DataModel) -> None:
        # Create dict of serializable metadata at once
        meta = {}

        # Store the actual class name for proper reconstruction
        meta["__class__"] = data.__class__.__name__
        meta["__module__"] = data.__class__.__module__

        # Process all fields without loops
        for field, value in data.model_dump().items():
            if isinstance(value, np.ndarray):
                # Handle numpy array
                is_image = self._is_image_array(value)
                file_name = f"{field}.png" if is_image else f"{field}.npy"
                file_path = Path(self.uri) / file_name

                with self.artifact_store.open(str(file_path), "wb") as f:
                    if is_image:
                        img = Image.fromarray(value)
                        img.save(f, format="PNG")
                    else:
                        np.save(f, value)

                meta[field] = {
                    "file": file_name,
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "type": "numpy",
                }
            elif isinstance(value, Image.Image):
                # Handle PIL Image directly
                file_name = f"{field}.png"
                file_path = Path(self.uri) / file_name

                with self.artifact_store.open(str(file_path), "wb") as f:
                    value.save(f, format="PNG")

                meta[field] = {
                    "file": file_name,
                    "mode": value.mode,
                    "size": value.size,
                    "type": "pil",
                }
            else:
                # Simple metadata field
                meta[field] = value

        # Write metadata file
        meta_path = Path(self.uri) / "data.json"
        with self.artifact_store.open(str(meta_path), "w") as f:
            json.dump(meta, f)

    def load(self, data_type: Type[DataModel]) -> DataModel:
        # Read the metadata JSON
        meta_path = Path(self.uri) / "data.json"
        with self.artifact_store.open(str(meta_path), "r") as f:
            meta = json.load(f)

        # Process all fields without building dict in a loop
        data_fields = {}

        # Get the original class if stored
        original_class = meta.pop("__class__", None)
        original_module = meta.pop("__module__", None)

        # Try to get the actual class if it exists
        target_class = data_type
        if original_class and original_module:
            try:
                import importlib

                module = importlib.import_module(original_module)
                if hasattr(module, original_class):
                    target_class = getattr(module, original_class)
            except (ImportError, AttributeError):
                # If we can't find the class, fallback to the provided data_type
                pass

        for field, value in meta.items():
            if isinstance(value, dict) and "file" in value:
                file_path = Path(self.uri) / value["file"]

                with self.artifact_store.open(str(file_path), "rb") as f:
                    if value.get("type") == "pil" or value["file"].endswith(".png"):
                        img = Image.open(f)
                        # Keep as PIL or convert based on original type
                        data_fields[field] = (
                            np.array(img) if value.get("type") == "numpy" else img
                        )
                    else:
                        # Load .npy file
                        data_fields[field] = np.load(f, allow_pickle=False)
            else:
                # Simple field
                data_fields[field] = value

        # Use the target class (original or fallback)
        return target_class(**data_fields)
