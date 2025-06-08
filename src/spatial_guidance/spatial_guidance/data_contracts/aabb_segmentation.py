import base64
import io
import math
import trace
import traceback
from typing import Annotated, Any, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..utils import Console
from . import DataModel


def pixel_to_camera_coordinates(
    pixel_coords: np.ndarray, depth: float, camera_intrinsics: np.ndarray
) -> np.ndarray:
    """
    Convert pixel coordinates to camera coordinates using depth and camera intrinsics.

    Args:
        pixel_coords: Array of shape (2,) with [x, y] pixel coordinates
        depth: Depth value in meters
        camera_intrinsics: 3x3 camera intrinsics matrix

    Returns:
        Camera coordinates as np.ndarray of shape (3,) in meters
    """
    x_pixel, y_pixel = pixel_coords
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Convert to camera coordinates
    x_cam = (x_pixel - cx) * depth / fx
    y_cam = (y_pixel - cy) * depth / fy
    z_cam = depth

    return np.array([x_cam, y_cam, z_cam])


def camera_to_world_coordinates(
    camera_coords: np.ndarray, camera_pose: np.ndarray
) -> np.ndarray:
    """
    Transform camera coordinates to world coordinates using camera pose.

    Args:
        camera_coords: Array of shape (3,) with camera coordinates in meters
        camera_pose: 4x4 world-to-camera transformation matrix

    Returns:
        World coordinates as np.ndarray of shape (3,) in meters
    """
    # Convert to homogeneous coordinates
    camera_coords_homo = np.append(camera_coords, 1.0)

    # Invert the camera pose to get camera-to-world transformation
    world_to_camera = camera_pose
    camera_to_world = np.linalg.inv(world_to_camera)

    # Transform to world coordinates
    world_coords_homo = camera_to_world @ camera_coords_homo

    return world_coords_homo[:3]


def compute_3d_center_from_bbox(
    bbox: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the 3D center of an object from its bounding box using depth statistics.

    Args:
        bbox: Bounding box coordinates [y0, x0, y1, x1] in pixels
        depth_image: Depth image as numpy array (height, width) in meters
        camera_intrinsics: 3x3 camera intrinsics matrix
        camera_pose: 4x4 world-to-camera transformation matrix

    Returns:
        3D center in world coordinates as np.ndarray of shape (3,) or None if computation fails
    """
    try:
        y0, x0, y1, x1 = bbox.astype(int)

        # Extract depth values from bounding box region
        bbox_depth = depth_image[y0:y1, x0:x1]
        valid_depths = bbox_depth[bbox_depth > 0]  # Filter out invalid depths

        if len(valid_depths) == 0:
            return None

        # Use median depth for robust estimation
        median_depth = np.median(valid_depths)

        # Center of bounding box in pixel coordinates
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        pixel_coords = np.array([center_x, center_y])

        # Convert to camera coordinates then to world coordinates
        camera_coords = pixel_to_camera_coordinates(
            pixel_coords, median_depth, camera_intrinsics
        )
        # world_coords = camera_to_world_coordinates(camera_coords, camera_pose)

        return camera_coords

    except Exception as e:
        Console.with_prefix("compute_3d_center_from_bbox").error(
            f"Error computing 3D center: {e}"
        )
        return None


def compute_3d_center_from_mask(
    mask: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the 3D center of an object from its segmentation mask using depth statistics.

    Args:
        mask: Binary mask as numpy array (height, width) with values > 0 for object pixels
        depth_image: Depth image as numpy array (height, width) in meters
        camera_intrinsics: 3x3 camera intrinsics matrix
        camera_pose: 4x4 world-to-camera transformation matrix

    Returns:
        3D center in world coordinates as np.ndarray of shape (3,) or None if computation fails
    """
    try:
        # Find pixels belonging to the object
        object_pixels = mask > 0

        if not np.any(object_pixels):
            return None

        # Extract depth values for object pixels
        object_depths = depth_image[object_pixels]
        valid_depths = object_depths[object_depths > 0]  # Filter out invalid depths

        if len(valid_depths) == 0:
            return None

        # Use median depth for robust estimation
        median_depth = np.median(valid_depths)

        # Center of mass of the mask in pixel coordinates
        y_coords, x_coords = np.where(object_pixels)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        pixel_coords = np.array([center_x, center_y])

        # Convert to camera coordinates then to world coordinates
        camera_coords = pixel_to_camera_coordinates(
            pixel_coords, median_depth, camera_intrinsics
        )
        # world_coords = camera_to_world_coordinates(camera_coords, camera_pose)

        return camera_coords

    except Exception as e:
        Console.with_prefix("compute_3d_center_from_mask").error(
            f"Error computing 3D center: {e}"
        )
        return None


def compute_rotation_from_3d_position(
    center_3d: Optional[List[float]],
) -> Tuple[Optional[float], Optional[int]]:
    """
    Compute rotation information from 3D position in camera coordinates for BEV representation.

    The rotation represents the angle between the forward axis (z-axis) in a Bird's Eye View (BEV)
    perspective, computed from metric camera coordinates in the x-z plane.

    Args:
        center_3d: 3D position [x, y, z] in camera frame where:
                  - x: right/left (positive right, negative left) → BEV x-axis
                  - y: up/down (positive up, negative down) → BEV height (not used in rotation)
                  - z: forward/backward (positive forward) → BEV y-axis (forward direction)

    Returns:
        Tuple of (rotation_deg, rotation_clock) where:
        - rotation_deg: BEV rotation angle in degrees from z-axis (forward) in x-z plane
                       (0° = straight ahead/12 o'clock, measured clockwise from z-axis)
        - rotation_clock: Single 12-hour clock position (1-12) for BEV orientation
    """
    try:
        if center_3d is None or len(center_3d) != 3:
            return None, None

        x_cam, y_cam, z_cam = center_3d

        # Check if object is behind camera (negative z)
        if z_cam <= 0:
            return None, None

        # Calculate BEV bearing angle in x-z plane using atan2(x, z)
        # This computes the angle from the forward z-axis to the object position
        # In BEV: z-axis → forward (y in BEV), x-axis → lateral (x in BEV)
        # atan2(x, z) returns angle in range [-π, π] from z-axis (forward direction)
        bearing_rad = math.atan2(x_cam, z_cam)
        bearing_deg = math.degrees(bearing_rad)

        # Convert to 0-360 degree range for BEV with 0° = forward (12 o'clock)
        # Positive x (right) gives positive angles (1-6 o'clock in BEV)
        # Negative x (left) gives negative angles, wrapped to (7-11 o'clock in BEV)
        if bearing_deg < 0:
            rotation_deg = 360 + bearing_deg
        else:
            rotation_deg = bearing_deg

        # Convert to 12-hour clock position for BEV orientation
        # Each hour represents 30 degrees (360° / 12 hours)
        # 12 o'clock = 0° = forward (z-axis), measured clockwise in BEV
        hour_step = 30.0
        clock_hour = round(rotation_deg / hour_step) % 12
        if clock_hour == 0:
            clock_hour = 12

        return rotation_deg, clock_hour

    except Exception as e:
        Console.with_prefix("compute_rotation_from_3d_position").error(
            f"Error computing rotation from 3D position: {e}"
        )
        return None, None


def compute_rotation_from_bbox(
    bbox: np.ndarray,
    img_size: Optional[Tuple[int, int]] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[int]]:
    """
    DEPRECATED: Compute rotation information from bounding box geometry.

    This function is kept for backward compatibility but should not be used
    for new code. Use compute_rotation_from_3d_position instead.

    Args:
        bbox: Bounding box coordinates [y0, x0, y1, x1] in pixels
        img_size: Image size (width, height) in pixels for FOV calculation
        camera_intrinsics: 3x3 camera intrinsics matrix for FOV calculation

    Returns:
        Tuple of (rotation_deg, rotation_clock) where:
        - rotation_deg: Rotation in degrees (0 = straight ahead/12 o'clock)
        - rotation_clock: Single 12-hour clock position (1-12), None if outside visible FOV
    """
    try:
        y0, x0, y1, x1 = bbox

        # Calculate bounding box dimensions and center
        width = x1 - x0
        height = y1 - y0
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0

        if width == 0 and height == 0:
            return None, None

        # Calculate rotation based on position in image if camera intrinsics are available
        if img_size is not None and camera_intrinsics is not None:
            img_width, img_height = img_size
            fx = camera_intrinsics[0, 0]
            cx = camera_intrinsics[0, 2]
            cy = camera_intrinsics[1, 2]

            # Calculate horizontal FOV for boundary checking
            horizontal_fov_rad = 2 * math.atan(img_width / (2 * fx))
            horizontal_fov_deg = math.degrees(horizontal_fov_rad)

            # Calculate angular position using proper trigonometry
            # Angular offset from optical center (positive = right, negative = left)
            pixel_offset_x = center_x - cx
            angular_offset = math.degrees(math.atan(pixel_offset_x / fx))

            # Check if object is within visible FOV range
            if abs(angular_offset) > horizontal_fov_deg / 2:
                # Object is outside visible FOV
                return None, None

            # Convert to compass bearing with 12 o'clock = 0° (straight ahead)
            # Positive angular_offset (right side) maps to positive degrees (1-6 o'clock)
            # Negative angular_offset (left side) maps to negative degrees, wrapped to (7-11 o'clock)
            if angular_offset >= 0:
                # Right side: 0° to +FOV/2 maps to 0° to +FOV/2
                rotation_deg = angular_offset
            else:
                # Left side: -FOV/2 to 0° maps to 360°-FOV/2 to 360°
                rotation_deg = 360 + angular_offset
        else:
            # Fallback to simple aspect ratio-based rotation
            if width > height:
                # Horizontally oriented - assume pointing East (3 o'clock)
                rotation_deg = 90.0
            else:
                # Vertically oriented - assume pointing North (12 o'clock)
                rotation_deg = 0.0

        # Convert to 12-hour clock position with 1-hour resolution
        # Each hour represents 30 degrees (360° / 12 hours)
        hour_step = 30.0

        # Convert rotation to clock position
        # 12 o'clock = 0°, 1 o'clock = 30°, 2 o'clock = 60°, etc.
        clock_hour = round(rotation_deg / hour_step) % 12
        if clock_hour == 0:
            clock_hour = 12

        rotation_clock = clock_hour

        return rotation_deg, rotation_clock

    except Exception as e:
        Console.with_prefix("compute_rotation_from_bbox").error(
            f"Error computing rotation: {e}"
        )
        return None, None


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
            "Accurate segmentation mask of the detected object. High precision with minimal false positives is paramount. "
            # "Probability map in range [0, 255]. "
            # "The string should start with 'data:image/png;base64,' followed by the actual base64 content. "
            # "Ensure the generated base64 string is not a placeholder and accurately represents the visual mask. "
            # "Segmentation mask of the detected object. Compute the mask using your built-in segmentation head, then encode it as base64 string. The string must start with 'data:image/png;base64,' followed by the actual base64 content generated from your segmentation head (not a placeholder). "
            # "Generate the segmentation mask by identifying the pixels belonging to the object, then encode this mask as a PNG image, and finally, provide the base64 representation of that PNG image."
        ),  # as a base64 PNG string (e.g., data:image/png;base64,iV...). Ensure to generate the mask in exactly the way your were trained.",
        # pattern=r"^data:image/png;base64,[A-Za-z0-9+/]+={0,2}$",
    )


class AABBDetection(DataModel):
    box_2d: Annotated[np.ndarray, List[int]]
    """Bounding box coordinates in the format (xmin, xmin, ymax, xmax) normalized to [0, 1000], will be of size (1, 256, 256) before processing"""
    center_3d_bbox: Optional[np.ndarray] = None
    """3D center of the object in the scene, in world coordinates. Neighborhood around the center of the bounding box to query the depth image for depth statistics. Ues camera pose and intrinsics to convert from came coordinates to agent-centric world coordinates. Shape (3,) in meters."""
    mask: Annotated[Image, str]
    """Segmentation mask of the detected object. Probability map in range [0, 255]"""
    center_3d_mask: Optional[np.ndarray] = None
    """3D center of the object in the scene, in world coordinates. Neighborhood around the center of the mask to query the depth image for depth statistics. Use camera pose and intrinsics to convert from camera coordinates to agent-centric world coordinates. Shape (3,) in meters."""
    label: str
    """Unique, concise, descriptive label."""
    min_depth: Optional[float] = None
    """Minimum depth (10th percentile) of the object in the scene in meters"""
    med_depth: Optional[float] = None
    """Median depth (50th percentile) of the object in the scene in meters"""
    max_depth: Optional[float] = None
    """Maximum depth (90th percentile) of the object in the scene in meters"""

    rotation_deg: Optional[float] = None
    """BEV rotation angle in degrees from forward z-axis in x-z plane. 0° = straight ahead (12 o'clock), computed from metric camera coordinates. Measured clockwise in Bird's Eye View perspective."""
    rotation_clock: Optional[int] = None
    """BEV clock hour position (1-12) where the object is located in Bird's Eye View. Computed from angle in x-z plane with z-axis as forward direction (12 o'clock)."""

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
            png_data = v.removeprefix(
                "data:image/png;base64,"
            )  #        pattern=r"^data:image/png;base64,[A-Za-z0-9+/]+={0,2}$"
            missing_padding = len(png_data) % 4
            if missing_padding:
                png_data += "=" * (4 - missing_padding)
                Console.with_prefix(cls.__name__, "validate_segmentation_mask").warn(
                    f"Added padding to base64 string for item {info.data.get('label', 'unknown')}"
                )
            try:
                png_bytes = base64.b64decode(png_data)
                mask_img = PILImage.open(io.BytesIO(png_bytes))
            except Exception as e:
                CONSOLE = Console.with_prefix(
                    cls.__name__, "validate_segmentation_mask"
                )
                CONSOLE.error(
                    f"Invalid mask format for item {info.data.get('label', 'unknown')}\n"
                    f"{e}\n{traceback.format_exc()}"
                )
                return PILImage.new("L", (1, 1))

            return mask_img

        CONSOLE = Console.with_prefix(cls.__name__, "validate_segmentation_mask")
        CONSOLE.error(
            f"Invalid mask format for item {info.data['label']}\n"
            f"Expected base64 PNG string, but got {type(v)}: {v}"
        )
        return PILImage.new("L", (1, 1))

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
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
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

            # Compute 3D center and rotation if camera parameters are available
            if (
                camera_intrinsics is not None
                and camera_pose is not None
                and depth_image is not None
            ):
                try:
                    # Compute 3D center from bounding box
                    bbox_array = np.array([abs_y0, abs_x0, abs_y1, abs_x1])
                    self.center_3d_bbox = compute_3d_center_from_bbox(
                        bbox_array, depth_image, camera_intrinsics, camera_pose
                    )

                    # Compute 3D center from mask
                    self.center_3d_mask = compute_3d_center_from_mask(
                        np_mask_full, depth_image, camera_intrinsics, camera_pose
                    )

                    # Compute rotation from 3D position (prefer mask-based center, fallback to bbox-based)
                    # Select the 3D center: use mask center if available, otherwise bbox center
                    center_for_rotation = (
                        self.center_3d_mask
                        if self.center_3d_mask is not None
                        else self.center_3d_bbox
                    )
                    # Compute rotation_deg and rotation_clock from 3D
                    self.rotation_deg, self.rotation_clock = (
                        compute_rotation_from_3d_position(center_for_rotation)
                        if center_for_rotation is not None
                        else (None, None)
                    )
                except Exception as e:
                    CONSOLE.error(
                        f"Error computing 3D parameters for {self.label}:\n"
                        f"{e}\n{traceback.format_exc()}"
                    )

            self.mask = PILImage.fromarray(np_mask_full, mode="L")
            self.box_2d = np.array([abs_y0, abs_x0, abs_y1, abs_x1], dtype=np.float32)

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
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
    ) -> None:
        """
        Process all detection objects with the given image dimensions.

        Args:
            img_size: Tuple containing the width and height of the original image in pixels
            confidence_thresh: Optional confidence threshold for filtering detections in the range [0, 1]
            depth_image: Optional PIL depth image for depth statistics calculation
            camera_intrinsics: Optional 3x3 camera intrinsics matrix
            camera_pose: Optional 4x4 world-to-camera transformation matrix
        """
        depth_array = (
            np.array(depth_image, dtype=np.float32) if depth_image is not None else None
        )

        for obj in self.objects:
            obj.process(
                img_size, confidence_thresh, depth_array, camera_intrinsics, camera_pose
            )

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
                "rotation_clock": obj.rotation_clock,
            }
            detections_list.append(detection_dict)

        return json.dumps(detections_list)
