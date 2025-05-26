"""Dataset class for StrayScanner data focusing on essential functionality."""

from pathlib import Path
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np

# import open3d as o3d
from PIL import Image
from pydantic import Field
from zenml.steps import BaseStep

from ...data_contracts.dataset import DatasetOut, PipelineIn
from ...utils import BaseConfig, Console
from .data_parser import StrayScannerDataParserConfig


class StrayDatasetConfig(BaseConfig["StrayDataset"]):
    """Configuration for StrayScanner dataset."""

    target: Type["StrayDataset"] = Field(default_factory=lambda: StrayDataset)
    """Target class to instantiate."""

    data_parser_config: StrayScannerDataParserConfig = Field(
        default_factory=StrayScannerDataParserConfig
    )
    """Configuration for the data parser."""

    scale_factor: float = 1.0
    """Scale factor for camera poses. No scaling if 1.0."""

    is_rotated: bool = True
    """Whether the frames are rotated."""

    auto_scale_poses: bool = True
    """Whether to auto-scale poses to fit in a unit cube."""

    confidence_threshold: int = 1  # TODO: add filtering
    """Minimum confidence level for depth values (0-2)."""

    resize_depth_to_rgb: bool = True
    """Whether to resize depth images to match RGB dimensions."""

    depth_unit_scale_factor: float = 1e-3
    """Factor to convert depth values to meters (1e-3 converts mm to m)."""

    def setup_target(self) -> "StrayDataset":
        return self.target(self)


class StrayDataset(BaseStep):
    """Dataset class for StrayScanner data with minimal but essential functionality."""

    def __init__(self, config: Optional[StrayDatasetConfig] = None, **step_kwargs):
        """Initialize the dataset.

        Args:
            config: Configuration for the dataset.
        """
        super().__init__(**step_kwargs)
        self.config = config or StrayDatasetConfig()

        self.parser = self.config.data_parser_config.setup_target()

        # Cache containers
        self._poses_cache: Optional[List[np.ndarray]] = None
        self._rgb_frames: Optional[List[Path]] = None
        self._depth_frames: Optional[List[Path]] = None

    def entrypoint(self, input_data: PipelineIn) -> DatasetOut:
        rgb_image = Image.fromarray(self.get_rgb(input_data.idx))
        # Use mode="F" for 32-bit float grayscale to preserve depth information
        depth_image = Image.fromarray(self.get_depth(input_data.idx), mode="F")

        # Resize depth image to match RGB dimensions if required
        if self.config.resize_depth_to_rgb:
            Console.with_prefix(self.__class__.__name__, "entrypoint").log(
                f"Resizing depth image to match RGB dimensions {depth_image.size} -> {rgb_image.size}"
            )
            depth_image = depth_image.resize(rgb_image.size, Image.LANCZOS)

        # Get camera intrinsics - scale if resizing was applied
        camera_intrinsics = self.parser.get_intrinsics()
        if self.config.resize_depth_to_rgb and rgb_image.size != depth_image.size:
            # Scale intrinsics to match the image dimensions
            width, height = rgb_image.size
            camera_intrinsics = self.get_scaled_intrinsics(width, height)

        # Get the camera pose for the current frame
        camera_pose = self.get_poses(idx=input_data.idx)[0]

        return DatasetOut(
            rgb_image=rgb_image,
            depth_image=depth_image,
            user_prompt=input_data.user_prompt,
            camera_intrinsics=camera_intrinsics.copy(),
            camera_pose=camera_pose.copy(),
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
        """
        Returns a resized intrinsics matrix K' for a new image size (W',H').

        Given original K and original image dims (W,H), compute:

            sx = W'/W,   sy = H'/H

        then

                 [ fx·sx,    0,  cx·sx]
            K' = [  0,    fy·sy, cy·sy]
                 [  0,        0,     1]
        """
        intrinsics = self.parser.get_intrinsics()
        rgb_height, rgb_width = self.get_rgb_dimensions()

        scale_x = target_width / rgb_width
        scale_y = target_height / rgb_height
        return self.parser._resize_camera_matrix(intrinsics, scale_x, scale_y)  # type: ignore

    def get_imu_data(self) -> Optional[np.ndarray]:
        """Get IMU sensor data if available."""
        return self.parser.get_imu_data()  # type: ignore

    def get_poses(self, idx: Optional[int] = None) -> List[np.ndarray]:
        """
        Returns a list of 4x4 world-to-camera transforms T_WC:

            T_WC = [ R | t ]
                   [ 0 | 1 ]

        where R is the rotation matrix from the unit quaternion (qx,qy,qz,qw)
        and t = [x, y, z]^T is the camera center in world coordinates.

        Args:
            idx: Optional index to get a specific pose. If None, returns all poses.
        """
        if self._poses_cache is not None:
            return self._poses_cache

        poses = self.parser.get_poses()

        # Scale poses if requested
        if self.config.auto_scale_poses or self.config.scale_factor != 1.0:
            poses = self._scale_poses(poses)

        if self._poses_cache is None:
            self._poses_cache = poses

        return poses if idx is None else [poses[idx]]

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
            # # Try loading from video
            # rgb_video_path = self.parser.config.paths.get_rgb_video_path()
            # if not rgb_video_path.exists():
            #     raise FileNotFoundError(f"No RGB data found for frame {idx}")

            # # Find the frame in the video
            # frame = None
            # video = skvideo.io.vreader(str(rgb_video_path))
            # for i, current_frame in enumerate(video):
            #     if i == idx:
            #         frame = current_frame
            #         break

            # if frame is None:
            #     raise IndexError(f"Frame {idx} not found in video")
            # rgb = frame
            # is_from_rotated = False
            raise ValueError("Loading RGB frames from video is not supported. ")

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

    # def get_rgbd(self, idx: int) -> o3d.geometry.RGBDImage:
    #     """Get RGBD image for a specific frame.

    #     Args:
    #         idx: Frame index.

    #     Returns:
    #         Open3D RGBD image.
    #     """
    #     rgb = self.get_rgb(idx)
    #     depth = self.get_depth(idx)

    #     # Get depth dimensions
    #     depth_height, depth_width = depth.shape[:2]

    #     # Resize RGB to match depth dimensions
    #     rgb_pil = Image.fromarray(rgb)
    #     rgb_pil = rgb_pil.resize((depth_width, depth_height))
    #     rgb = np.array(rgb_pil)

    #     return o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         o3d.geometry.Image(rgb),
    #         o3d.geometry.Image(depth),
    #         depth_scale=1.0,
    #         convert_rgb_to_intensity=False,
    #     )

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
