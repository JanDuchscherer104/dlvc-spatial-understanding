import base64
import io
import re
import trace
import traceback
from typing import Annotated, Any, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from ..utils import Console
from . import DataModel


class RawAABBDetSeg(DataModel):
    label: str = Field(
        description="Unique, concise, descriptive label without any room for ambiguity. Examples: 'car_parked', 'scooter_innactive_lying', 'construction_barrier_right'",
    )
    box_2d: List[int] = Field(
        description="Bounding box [y0, x0, y1, x1], normalized to [0,1000]",
        examples=[[100, 150, 200, 250]],
    )
    # base64 PNG string
    mask: str = Field(
        description=(
            "Segmentation mask of the detected object. "
            # "Probability map in range [0, 255]. "
            # "The string should start with 'data:image/png;base64,' followed by the actual base64 content. "
            # "Ensure the generated base64 string is not a placeholder and accurately represents the visual mask. "
            # "Segmentation mask of the detected object. Compute the mask using your built-in segmentation head, then encode it as a PNG and base64 string. The string must start with 'data:image/png;base64,' followed by the actual base64 content generated from your segmentation head (not a placeholder). Ensure the mask accurately represents the object."
        ),  # as a base64 PNG string (e.g., data:image/png;base64,iV...). Ensure to generate the mask in exactly the way your were trained.",
        # pattern=r"^data:image/png;base64,[A-Za-z0-9+/]+={0,2}$",
    )


class AABBDetection(DataModel):
    box_2d: Annotated[np.ndarray, List[int]]
    """Bounding box coordinates in the format (xmin, xmin, ymax, xmax) normalized to [0, 1000], will be of size (1, 256, 256) before processing"""
    mask: Annotated[Image, str] = Field(
        pattern=r"^data:image/png;base64,[A-Za-z0-9+/]+={0,2}$"
    )
    """Segmentation mask of the detected object. Probability map in range [0, 255]"""
    label: str
    """Unique, concise, descriptive label."""
    min_depth: Optional[float] = None
    """Minimum depth (10th percentile) of the object in the scene in meters"""
    med_depth: Optional[float] = None
    """Median depth (50th percentile) of the object in the scene in meters"""
    max_depth: Optional[float] = None
    """Maximum depth (90th percentile) of the object in the scene in meters"""

    processed_: bool = False

    image_center: Optional[np.ndarray] = Field(
        None,
        description="Center of the projected 3D corners in image space, shape (2,), in pixels.",
    )

    @field_validator("mask", mode="before")
    @classmethod
    def validate_segmentation_mask(cls, v: str, info: ValidationInfo) -> Image:
        if isinstance(v, PILImage.Image):
            return v

        if isinstance(v, str):
            png_data = v.removeprefix("data:image/png;base64,")
            
            # Enhanced validation: Check for valid base64 characters first
            import re
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', png_data):
                CONSOLE = Console.with_prefix(cls.__name__, "validate_segmentation_mask")
                CONSOLE.error(
                    f"Invalid base64 characters in mask for item {info.data.get('label', 'unknown')}. "
                    f"Creating fallback mask from bounding box."
                )
                return cls._create_fallback_mask(info.data.get('box_2d', [0, 0, 100, 100]))
            
            missing_padding = len(png_data) % 4
            if missing_padding:
                png_data += "=" * (4 - missing_padding)
                Console.with_prefix(cls.__name__, "validate_segmentation_mask").warn(
                    f"Added padding to base64 string for item {info.data.get('label', 'unknown')}"
                )
            try:
                png_bytes = base64.b64decode(png_data, validate=True)
                mask_img = PILImage.open(io.BytesIO(png_bytes))
                
                # Validate image properties
                if mask_img.size[0] < 1 or mask_img.size[1] < 1:
                    raise ValueError("Invalid mask dimensions")
                    
                return mask_img
                
            except Exception as e:
                CONSOLE = Console.with_prefix(
                    cls.__name__, "validate_segmentation_mask"
                )
                CONSOLE.error(
                    f"Failed to decode mask for item {info.data.get('label', 'unknown')}: {e}. "
                    f"Creating fallback mask from bounding box."
                )
                return cls._create_fallback_mask(info.data.get('box_2d', [0, 0, 100, 100]))

        # Invalid type - create fallback
        CONSOLE = Console.with_prefix(cls.__name__, "validate_segmentation_mask")
        CONSOLE.error(
            f"Invalid mask type for item {info.data.get('label', 'unknown')}: {type(v)}. "
            f"Creating fallback mask from bounding box."
        )
        return cls._create_fallback_mask(info.data.get('box_2d', [0, 0, 100, 100]))

    @classmethod
    def _create_fallback_mask(cls, bbox: List[int]) -> PILImage.Image:
        """Create a simple rectangular mask from bounding box coordinates."""
        from PIL import ImageDraw
        
        # Create 256x256 mask (standard size)
        mask = PILImage.new('L', (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        
        # Scale bbox to 256x256 if needed (assuming bbox is in [0,1000] range)
        if len(bbox) >= 4:
            scaled_bbox = [
                max(0, min(255, int(bbox[1] * 256 / 1000))),  # x1 (from y0)
                max(0, min(255, int(bbox[0] * 256 / 1000))),  # y1 (from x0) 
                max(0, min(255, int(bbox[3] * 256 / 1000))),  # x2 (from y1)
                max(0, min(255, int(bbox[2] * 256 / 1000)))   # y2 (from x1)
            ]
            draw.rectangle(scaled_bbox, fill=255)
        else:
            # If bbox is invalid, create a small center mask
            draw.rectangle([100, 100, 156, 156], fill=255)
        
        return mask

    @field_validator("box_2d", mode="before")
    @classmethod
    def validate_box_2d(cls, v: List[int], info: ValidationInfo) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        assert isinstance(
            v, list
        ), f"box_2d must be a list, but got {type(v)} for {info.data.get('label', 'unknown')}"
        assert len(v) == 4, (
            "box_2d must have 4 elements [y1, x1, y2, x2] in range 0-1000\n"
            f" but got {len(v)} elements: {v} for {info.data.get('label', 'unknown')}"
        )
        return np.array(v, dtype=np.float32)  # type: ignore

    def process(
        self,
        img_size: Tuple[int, int],  # img_size is (width, height)
        confidence_tresh: Optional[float] = None,
        depth_image: Optional[np.ndarray] = None,
    ) -> None:
        """
        Process the segmentation mask and bounding box for proper visualization.

        This method scales the normalized bounding box coordinates to actual pixel values
        and resizes/positions the segmentation mask within the image frame.

        Args:
            img_size: Tuple containing the width and height of the original image in pixels
            confidence_tresh: Optional confidence threshold for filtering masks
            depth_image: Optional depth image for depth statistics calculation
        """
        if self.processed_:
            return
        CONSOLE = Console.with_prefix(self.__class__.__name__, "process")

        try:
            y0_norm, x0_norm, y1_norm, x1_norm = self.box_2d

            img_width, img_height = img_size
            abs_y0 = int(y0_norm / 1000 * img_height)
            abs_x0 = int(x0_norm / 1000 * img_width)
            abs_y1 = int(y1_norm / 1000 * img_height)
            abs_x1 = int(x1_norm / 1000 * img_width)

            # Validate bounding box dimensions
            if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
                CONSOLE.warn(
                    f"Invalid bounding box dimensions after normalization: y0={abs_y0},x0={abs_x0},y1={abs_y1},x1={abs_x1} for label {self.label}. Original normalized: {self.box_2d}"
                )
                self.mask = PILImage.new("L", (img_width, img_height))
                self.processed_ = True
                return

            bbox_height = abs_y1 - abs_y0
            bbox_width = abs_x1 - abs_x0

            if bbox_height < 1 or bbox_width < 1:
                CONSOLE.warn(
                    f"Bounding box too small after normalization: {bbox_width}x{bbox_height} for label {self.label}. Original normalized: {self.box_2d}"
                )
                self.mask = PILImage.new("L", (img_width, img_height))
                self.processed_ = True
                return

            mask_resized = self.mask.resize(
                (bbox_width, bbox_height), resample=PILImage.Resampling.LANCZOS
            ).convert("L")

            np_mask_full = np.zeros((img_height, img_width), dtype=np.uint8)
            np_mask_full[abs_y0:abs_y1, abs_x0:abs_x1] = np.array(mask_resized)

            if confidence_tresh is not None:
                threshold_value = round(confidence_tresh * 255)
                np_mask_full[np_mask_full <= threshold_value] = 0

            if depth_image is not None:
                try:
                    binary_mask = np_mask_full > 0
                    if binary_mask.sum() > 0:
                        depth_values = depth_image[binary_mask]

                        if len(depth_values) > 0:
                            self.min_depth = float(np.percentile(depth_values, 10))
                            self.med_depth = float(np.percentile(depth_values, 50))
                            self.max_depth = float(np.percentile(depth_values, 90))
                except Exception as e:
                    CONSOLE.error(
                        f"Error calculating depth for {self.label}:\n"
                        f"{e}\n{traceback.format_exc()}"
                    )

            self.mask = PILImage.fromarray(np_mask_full, mode="L")
            self.box_2d = np.array([abs_y0, abs_x0, abs_y1, abs_x1], dtype=np.float32)

            # Compute image-space center of the projected corners
            if isinstance(self.bev_bbox, np.ndarray):
                self.image_center = np.nanmean(self.bev_bbox, axis=0)

            self.processed_ = True

        except Exception as e:
            CONSOLE.error(
                f"Error processing object {self.label}:\n"
                f"{e}\n{traceback.format_exc()}"
            )


class AABBDetections(DataModel):
    """Complete analysis of a scene for navigation assistance."""

    objects: List[AABBDetection]
    """"List of detected objects in the scene"""
    visualization_rgb: Optional[Image] = Field(
        default=None, description="PIL Image of RGB detections overlay"
    )
    visualization_depth: Optional[Image] = Field(
        default=None, description="PIL Image of depth detections overlay"
    )

    def process_all(
        self,
        img_size: Tuple[int, int],  # img_size is (width, height)
        confidence_thresh: Optional[float] = None,
        depth_image: Optional[
            PILImage.Image
        ] = None,  # Changed to PILImage.Image for consistency
    ) -> None:
        """
        Process all detection objects with the given image dimensions.

        Args:
            img_size: Tuple containing the width and height of the original image in pixels
            confidence_thresh: Optional confidence threshold for filtering detections in the range [0, 1]
            depth_image: Optional PIL depth image for depth statistics calculation
        """
        depth_array = (
            np.array(depth_image, dtype=np.float32) if depth_image is not None else None
        )

        for obj in self.objects:
            obj.process(img_size, confidence_thresh, depth_array)

    def __getitem__(self, index: int) -> AABBDetection:
        return self.objects[index]

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def to_json_list(self) -> str:
        """Return AABB detections as a JSON list with label, bbox, and median depth."""
        import json

        detections_list = []
        for obj in self.objects:
            detection_dict = {
                "label": obj.label,
                "bbox": (
                    obj.box_2d.tolist()
                    if hasattr(obj.box_2d, "tolist")
                    else list(obj.box_2d)
                ),
                "depth": obj.med_depth,
            }
            detections_list.append(detection_dict)

        return json.dumps(detections_list)
