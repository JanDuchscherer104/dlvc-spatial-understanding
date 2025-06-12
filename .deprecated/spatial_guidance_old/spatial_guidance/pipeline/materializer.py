import json
from pathlib import Path
from typing import Any, List, Type

import numpy as np
from PIL import Image
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from ..data_contracts import DataModel


class PydanticNumpyMaterializer(BaseMaterializer):
    """Custom materializer for DataModel objects (Pydantic models with images)."""

    ASSOCIATED_TYPES = (DataModel,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def _serialize_for_json(self, obj: Any, prefix: str = "") -> Any:
        """
        Recursively serialize complex objects to JSON-compatible format.

        Args:
            obj: Object to serialize
            prefix: Prefix for file names to avoid conflicts

        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.ndarray):
            # Handle numpy array - always preserve original dtype
            # Don't convert to images to avoid data type loss
            file_name = f"{prefix}_array.npy"
            file_path = Path(self.uri) / file_name

            with self.artifact_store.open(str(file_path), "wb") as f:
                np.save(f, obj)

            return {
                "__type__": "numpy",
                "file": file_name,
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }

        elif isinstance(obj, Image.Image):
            # Handle PIL Image - preserve original mode and data
            if obj.mode == "F":
                # For float32 images, save as TIFF to preserve precision
                file_name = f"{prefix}_image.tiff"
                file_path = Path(self.uri) / file_name
                original_mode = obj.mode

                with self.artifact_store.open(str(file_path), "wb") as f:
                    obj.save(f, format="TIFF")
            else:
                # For other modes, use PNG
                file_name = f"{prefix}_image.png"
                file_path = Path(self.uri) / file_name
                original_mode = obj.mode

                with self.artifact_store.open(str(file_path), "wb") as f:
                    obj.save(f, format="PNG")

            return {
                "__type__": "pil",
                "file": file_name,
                "mode": original_mode,
                "size": obj.size,
            }

        elif isinstance(obj, DataModel):
            # Handle nested DataModel objects
            nested_data = {}
            nested_data["__class__"] = obj.__class__.__name__
            nested_data["__module__"] = obj.__class__.__module__
            nested_data["__type__"] = "datamodel"

            for field_name, field_value in obj.model_dump().items():
                nested_prefix = f"{prefix}_{field_name}" if prefix else field_name
                nested_data[field_name] = self._serialize_for_json(
                    field_value, nested_prefix
                )

            return nested_data

        elif isinstance(obj, list):
            # Handle lists (like List[AABBDetection])
            serialized_list = []
            for i, item in enumerate(obj):
                item_prefix = f"{prefix}_item{i}" if prefix else f"item{i}"
                serialized_list.append(self._serialize_for_json(item, item_prefix))
            return serialized_list

        elif isinstance(obj, dict):
            # Handle dictionaries
            serialized_dict = {}
            for key, value in obj.items():
                value_prefix = f"{prefix}_{key}" if prefix else str(key)
                serialized_dict[key] = self._serialize_for_json(value, value_prefix)
            return serialized_dict

        else:
            # Simple types (str, int, float, bool, None) - return as is
            return obj

    def _deserialize_from_json(self, obj: Any) -> Any:
        """
        Recursively deserialize objects from JSON-compatible format.

        Args:
            obj: Object to deserialize

        Returns:
            Deserialized object
        """
        if isinstance(obj, dict) and ("__type__" in obj or "type" in obj):
            # Handle backward compatibility: both "__type__" and "type" keys
            type_key = "__type__" if "__type__" in obj else "type"

            if obj[type_key] == "numpy":
                # Load numpy array - always load from .npy files to preserve dtype
                file_path = Path(self.uri) / obj["file"]
                with self.artifact_store.open(str(file_path), "rb") as f:
                    return np.load(f, allow_pickle=False)

            elif obj[type_key] == "pil":
                # Load PIL Image - handle both PNG and TIFF formats
                file_path = Path(self.uri) / obj["file"]
                with self.artifact_store.open(str(file_path), "rb") as f:
                    img = Image.open(f)
                    img.load()
                    # Ensure the image has the correct mode as stored in metadata
                    if "mode" in obj and img.mode != obj["mode"]:
                        # This shouldn't happen with our current implementation,
                        # but keep as safety check for backward compatibility
                        img = img.convert(obj["mode"])
                    return img

            elif obj[type_key] == "datamodel":
                # Reconstruct nested DataModel object
                class_name = obj.get("__class__")
                module_name = obj.get("__module__")

                # Try to get the actual class
                target_class = None
                if class_name and module_name:
                    try:
                        import importlib

                        module = importlib.import_module(module_name)
                        if hasattr(module, class_name):
                            target_class = getattr(module, class_name)
                    except (ImportError, AttributeError):
                        pass

                if target_class is None:
                    raise ValueError(
                        f"Could not find class {class_name} in module {module_name}"
                    )

                # Recursively deserialize all fields
                fields = {}
                for field_name, field_value in obj.items():
                    if field_name not in [
                        "__class__",
                        "__module__",
                        "__type__",
                        "type",
                    ]:
                        fields[field_name] = self._deserialize_from_json(field_value)

                return target_class(**fields)

        elif isinstance(obj, list):
            # Handle lists
            return [self._deserialize_from_json(item) for item in obj]

        elif isinstance(obj, dict):
            # Handle regular dictionaries (not special objects)
            return {
                key: self._deserialize_from_json(value) for key, value in obj.items()
            }

        else:
            # Simple types - return as is
            return obj

    def save(self, data: DataModel) -> None:
        # Create dict of serializable metadata at once
        meta = {}

        # Store the actual class name for proper reconstruction
        meta["__class__"] = data.__class__.__name__
        meta["__module__"] = data.__class__.__module__

        # Process all fields using recursive serialization
        for field, value in data.model_dump().items():
            meta[field] = self._serialize_for_json(value, field)

        # Write metadata file
        meta_path = Path(self.uri) / "data.json"
        with self.artifact_store.open(str(meta_path), "w") as f:
            json.dump(meta, f)

    def load(self, data_type: Type[DataModel]) -> DataModel:
        # Read the metadata JSON
        meta_path = Path(self.uri) / "data.json"
        with self.artifact_store.open(str(meta_path), "r") as f:
            meta = json.load(f)

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

        # Deserialize all fields using recursive deserialization
        data_fields = {}
        for field, value in meta.items():
            data_fields[field] = self._deserialize_from_json(value)

        # Use the target class (original or fallback)
        return target_class(**data_fields)
