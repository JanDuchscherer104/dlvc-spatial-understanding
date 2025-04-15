"""Dataset class for StrayScanner data focusing on essential functionality."""

from pathlib import Path
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np
import open3d as o3d
import skvideo.io
from click import prompt
from PIL import Image
from pydantic import Field
from sympy import use

from utils import CONSOLE, BaseConfig

from ...pipeline.data_contracts import DataSetOut, PipelineIn
from ...pipeline.pipeline_stage import PipelineStage, PipelineStageConfig
from .data_parser import StrayScannerDataParserConfig


class StrayDatasetConfig(PipelineStageConfig["StrayDataset"]):
    """Configuration for StrayScanner dataset."""

    target: Type["StrayDataset"] = Field(default_factory=lambda: StrayDataset)
    """Target class to instantiate."""

    data_parser_config: StrayScannerDataParserConfig = Field(
        default_factory=StrayScannerDataParserConfig
    )
    """Configuration for the data parser."""

    scale_factor: float = 1.0
    """Scale factor for camera poses."""

    is_rotated: bool = True
    """Whether the frames are rotated."""

    auto_scale_poses: bool = True
    """Whether to auto-scale poses to fit in a unit cube."""

    confidence_threshold: int = 1
    """Minimum confidence level for depth values (0-2)."""

    depth_unit_scale_factor: float = 1e-3
    """Factor to convert depth values to meters (1e-3 converts mm to m)."""

    depth_hw: Tuple[int, int] = (192, 256)

    def setup_target(self) -> "StrayDataset":
        return self.target(self)


class StrayDataset(PipelineStage[PipelineIn, DataSetOut]):
    """Dataset class for StrayScanner data with minimal but essential functionality."""

    def __init__(self, config: StrayDatasetConfig):
        """Initialize the dataset.

        Args:
            config: Configuration for the dataset.
        """
        super().__init__(config=config)
        self.config = config

        self.parser = self.config.data_parser_config.target(
            self.config.data_parser_config
        )

        # Cache containers
        self._poses_cache: Optional[List[np.ndarray]] = None
        self._rgb_frames: Optional[List[Path]] = None
        self._depth_frames: Optional[List[Path]] = None

    def entrypoint(self, input_data: PipelineIn) -> DataSetOut:
        return DataSetOut(
            rgb_image=Image.fromarray(self.get_rgb(input_data.idx)),
            depth_image=Image.fromarray(self.get_depth(input_data.idx)),
            user_prompt=input_data.user_prompt,
        )

    def get_rgb_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of RGB images (height, width).

        Returns:
            Tuple of (height, width) for RGB images
        """
        # Get RGB dimensions from any RGB frame
        if self._rgb_frames is None:
            self._rgb_frames = self.parser.get_available_rgb_frames()
            if not self._rgb_frames:
                raise ValueError("No RGB frames found in the dataset")

        # Load the first RGB frame to get dimensions
        rgb_path = self._rgb_frames[0]
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        rgb = np.array(Image.open(rgb_path))
        rgb_height, rgb_width = rgb.shape[:2]

        # Check if the image is from rotated directory
        is_from_rotated = str(self.parser.config.paths.get_rgb_rotated_dir()) in str(
            rgb_path
        )
        if self.config.is_rotated and not is_from_rotated:
            # Swap height and width if rotated
            rgb_height, rgb_width = rgb_width, rgb_height
        return rgb_height, rgb_width

    def get_scaled_intrinsics(
        self, target_width: int, target_height: int
    ) -> np.ndarray:
        """Get camera intrinsics matrix scaled to target resolution.

        Args:
            target_width: Target width in pixels
            target_height: Target height in pixels

        Returns:
            Scaled intrinsics matrix as numpy array
        """
        intrinsics = self.parser.get_intrinsics()
        rgb_height, rgb_width = self.get_rgb_dimensions()

        scale_x = target_width / rgb_width
        scale_y = target_height / rgb_height
        return self.parser._resize_camera_matrix(intrinsics, scale_x, scale_y)  # type: ignore

    def get_imu_data(self) -> Optional[np.ndarray]:
        """Get IMU sensor data if available."""
        return self.parser.get_imu_data()  # type: ignore

    def get_poses(self) -> List[np.ndarray]:
        """Get all camera poses, potentially scaled."""
        if self._poses_cache is not None:
            return self._poses_cache

        poses = self.parser.get_poses()

        # Scale poses if requested
        if self.config.auto_scale_poses or self.config.scale_factor != 1.0:
            poses = self._scale_poses(poses)

        if self._poses_cache is None:
            self._poses_cache = poses

        return poses

    def _scale_poses(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """Scale camera poses.

        Args:
            poses: List of camera pose matrices.

        Returns:
            Scaled camera pose matrices.
        """
        # Create a copy to avoid modifying the original poses
        poses = [pose.copy() for pose in poses]

        # Calculate scale factor based on maximum extent
        scale_factor = self.config.scale_factor
        if self.config.auto_scale_poses:
            positions = np.array([pose[:3, 3] for pose in poses])
            max_extent = np.max(np.abs(positions))  # type: ignore
            if max_extent > 0:
                scale_factor /= max_extent

        # Apply scaling
        if scale_factor != 1.0:
            for pose in poses:
                pose[:3, 3] *= scale_factor

        return poses

    def get_rgb(self, idx: int) -> Image.Image:
        """Get RGB image for a specific frame.

        Args:
            idx: Frame index.

        Returns:
            RGB image as a numpy array.
        """
        # Get available rgb frames (will check both regular and rotated)
        rgb_frames = self.parser.get_available_rgb_frames()

        # Try individual RGB frames first if available
        if rgb_frames and idx < len(rgb_frames):
            rgb_path = rgb_frames[idx]
            if rgb_path.exists():
                rgb = np.array(Image.open(rgb_path))
                # Check if the image is from rotated directory
                is_from_rotated = str(
                    self.parser.config.paths.get_rgb_rotated_dir()
                ) in str(rgb_path)
            else:
                raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        else:
            # Try loading from video
            rgb_video_path = self.parser.config.paths.get_rgb_video_path()
            if not rgb_video_path.exists():
                raise FileNotFoundError(f"No RGB data found for frame {idx}")

            # Find the frame in the video
            frame = None
            video = skvideo.io.vreader(str(rgb_video_path))
            for i, current_frame in enumerate(video):
                if i == idx:
                    frame = current_frame
                    break

            if frame is None:
                raise IndexError(f"Frame {idx} not found in video")
            rgb = frame
            is_from_rotated = False

        # Rotate if necessary - only rotate if it's not from the rotated directory and rotation is enabled
        if self.config.is_rotated and not is_from_rotated:
            # Rotate 90 degrees clockwise
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

        return rgb

    def get_depth(self, idx: int) -> np.ndarray:
        """Get depth map for a specific frame.

        Args:
            idx: Frame index.

        Returns:
            Depth map as numpy array in meters.
        """
        # Use the parser's specialized method for depth frame loading if available
        if hasattr(self.parser, "get_depth_frame"):
            depth_m = self.parser.get_depth_frame(idx)
        else:
            # Get depth frames
            depth_frames = self.parser.get_depth_frames()

            if depth_frames and idx < len(depth_frames):
                depth_path = depth_frames[idx]
                if not depth_path.exists():
                    raise FileNotFoundError(f"Depth file not found: {depth_path}")

                # Load depth map using cv2 for proper 16-bit depth handling
                depth_mm = cv2.imread(str(depth_path), -1)

                if depth_mm is None:
                    raise IOError(f"Failed to load depth image: {depth_path}")

                # Convert to meters
                depth_m = (
                    depth_mm.astype(np.float32) * self.config.depth_unit_scale_factor
                )
            else:
                raise FileNotFoundError(f"No depth data found for frame {idx}")

        # Rotate if necessary
        if self.config.is_rotated:
            # Rotate 90 degrees clockwise
            depth_m = cv2.rotate(depth_m, cv2.ROTATE_90_CLOCKWISE)

        return depth_m

    def get_rgbd(self, idx: int) -> o3d.geometry.RGBDImage:
        """Get RGBD image for a specific frame.

        Args:
            idx: Frame index.

        Returns:
            Open3D RGBD image.
        """
        rgb = self.get_rgb(idx)
        depth = self.get_depth(idx)

        # Get depth dimensions
        depth_height, depth_width = depth.shape[:2]

        # Resize RGB to match depth dimensions
        rgb_pil = Image.fromarray(rgb)
        rgb_pil = rgb_pil.resize((depth_width, depth_height))
        rgb = np.array(rgb_pil)

        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            convert_rgb_to_intensity=False,
        )

    # def get_point_cloud(self, idx: int) -> o3d.geometry.PointCloud:
    #     """Get colored point cloud for a specific frame.

    #     Args:
    #         idx: Frame index.

    #     Returns:
    #         Open3D point cloud with colors.
    #     """
    #     rgbd = self.get_rgbd(idx)
    #     intrinsics = self.get_o3d_intrinsics()
    #     T_WC = self.parser.get_pose(idx)
    #     T_CW = np.linalg.inv(T_WC)

    #     return o3d.geometry.PointCloud.create_from_rgbd_image(
    #         rgbd, intrinsics, extrinsic=T_CW
    #     )

    def __len__(self) -> int:
        """Get number of frames in dataset."""
        return self.parser.get_frame_count()  # type: ignore
