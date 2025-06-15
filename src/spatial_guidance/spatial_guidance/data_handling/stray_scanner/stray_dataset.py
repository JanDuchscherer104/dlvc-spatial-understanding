from pathlib import Path
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np
import open3d as o3d

# import open3d as o3d
from PIL import Image
from pydantic import Field

from ...data_contracts.dataset import DatasetOut
from ...utils.base_config import BaseConfig
from ...utils.console import Console
from .data_parser import StrayScannerDataParserConfig


def _rotate_intrinsics_90_cw(K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Rotate camera intrinsics matrix 90 degrees clockwise.

    Args:
        K: 3x3 camera intrinsics matrix
        w: width of the original (unrotated) image
        h: height of the original (unrotated) image

    Returns:
        Rotated 3x3 intrinsics matrix
    """
    K_rot = K.copy()
    K_rot[0, 0] = K[1, 1]  # fx' = fy
    K_rot[1, 1] = K[0, 0]  # fy' = fx
    K_rot[0, 2] = h - 1 - K[1, 2]  # cx' = h-1 - cy
    K_rot[1, 2] = K[0, 2]  # cy' = cx
    return K_rot


class StrayDatasetConfig(BaseConfig["StrayDataset"]):
    """Configuration for StrayScanner dataset."""

    target: Type["StrayDataset"] = Field(default_factory=lambda: StrayDataset)
    """Target class to instantiate."""

    is_debug: bool = True
    """Enable verbose debug logging."""

    data_parser: StrayScannerDataParserConfig = Field(
        default_factory=StrayScannerDataParserConfig
    )
    """Configuration for the data parser."""

    scale_factor: float = 1.0
    """Scale factor for camera poses. No scaling if 1.0."""

    is_rotated: bool = False
    """Whether the frames are rotated."""

    auto_scale_poses: bool = False
    """Whether to auto-scale poses to fit in a unit cube."""

    confidence_threshold: int = 1  # TODO: add filtering
    """Minimum confidence level for depth values (0-2)."""

    resize_depth_to_rgb: bool = True
    """Whether to resize depth images to match RGB dimensions."""

    depth_unit_scale_factor: float = 1e-3
    """Factor to convert depth values to meters (1e-3 converts mm to m)."""

    detect_ground_plane: bool = True
    """Whether to detect and include ground plane."""

    ground_plane_voxel_size: float = 0.05
    """Voxel size for ground plane detection downsampling."""

    ground_plane_distance_threshold: float = 0.02
    """Distance threshold for ground plane RANSAC detection."""

    ground_plane_max_iterations: int = 1000
    """Maximum iterations for ground plane RANSAC detection."""

    ground_plane_use_world_coords: bool = False
    """Whether to detect ground plane in world coordinates (True) or camera coordinates (False)."""

    def setup_target(self) -> "StrayDataset":

        if self.auto_scale_poses:
            Console.with_prefix(self.__class__.__name__, "_setup_target").warn(
                "Auto-scaling poses is enabled! This does not make sense for GeminiLiveAgent!"
            )
        return self.target(self)


class StrayDataset:
    """Dataset class for StrayScanner data with minimal but essential functionality."""

    def __init__(self, config: Optional[StrayDatasetConfig] = None, **step_kwargs):
        """Initialize the dataset.

        Args:
            config: Configuration for the dataset.
        """
        super().__init__(**step_kwargs)
        self.config = config or StrayDatasetConfig()

        self.parser = self.config.data_parser.setup_target()

        # Cache containers
        self._poses_cache: Optional[List[np.ndarray]] = None
        self._rgb_frames: Optional[List[Path]] = None
        self._depth_frames: Optional[List[Path]] = None

        self._ds_out_cache: dict[int, DatasetOut] = {}

    def __getitem__(self, idx: int) -> DatasetOut:
        if idx in self._ds_out_cache:
            return self._ds_out_cache[idx]

        rgb_image = Image.fromarray(self.get_rgb(idx))
        # Use mode="F" for 32-bit float grayscale to preserve depth information
        depth_array = self.get_depth(idx)
        depth_image = Image.fromarray(depth_array, mode="F")

        if self.config.resize_depth_to_rgb:
            Console.with_prefix(self.__class__.__name__, "__getitem__").set_debug(
                self.config.is_debug
            ).dbg(
                f"Resizing depth image to match RGB dimensions {depth_image.size} -> {rgb_image.size}"
            )
            depth_image = depth_image.resize(
                rgb_image.size, Image.LANCZOS
            )  # Get camera intrinsics and handle resolution/rotation properly
        camera_intrinsics = self.parser.get_intrinsics()

        # First, scale intrinsics from RGB resolution to depth resolution
        rgb_image_array = self.get_rgb(idx)
        rgb_h, rgb_w = rgb_image_array.shape[:2]
        depth_h, depth_w = depth_array.shape

        # Scale intrinsics to match depth resolution
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 0] *= scale_x  # fx
        camera_intrinsics[1, 1] *= scale_y  # fy
        camera_intrinsics[0, 2] *= scale_x  # cx
        camera_intrinsics[1, 2] *= scale_y  # cy

        # Then, if depth is rotated, rotate the intrinsics to match
        if self.config.is_rotated:
            # Since depth_array is already rotated, we need to determine original dimensions
            # After a 90° CW rotation: new_height = old_width, new_width = old_height
            # So: old_width = new_height, old_height = new_width
            h_rotated, w_rotated = depth_array.shape
            w_original = h_rotated
            h_original = w_rotated
            camera_intrinsics = _rotate_intrinsics_90_cw(
                camera_intrinsics, w_original, h_original
            )

        # Finally, if we're resizing depth to RGB, scale intrinsics again
        if self.config.resize_depth_to_rgb and rgb_image.size != (
            depth_array.shape[1],
            depth_array.shape[0],
        ):
            # Scale intrinsics to match the final RGB dimensions
            width, height = rgb_image.size
            final_scale_x = width / depth_array.shape[1]
            final_scale_y = height / depth_array.shape[0]
            camera_intrinsics[0, 0] *= final_scale_x  # fx
            camera_intrinsics[1, 1] *= final_scale_y  # fy
            camera_intrinsics[0, 2] *= final_scale_x  # cx
            camera_intrinsics[1, 2] *= final_scale_y  # cy

        # Get the camera pose for the current frame
        camera_pose = self.get_poses(idx=idx)[0]

        # Optionally detect ground plane
        ground_plane = None
        if self.config.detect_ground_plane:
            try:
                # Choose coordinate system for ground plane detection
                pose_for_detection = (
                    camera_pose if self.config.ground_plane_use_world_coords else None
                )

                # Use the depth data that matches the final intrinsics
                if self.config.resize_depth_to_rgb and rgb_image.size != (
                    depth_array.shape[1],
                    depth_array.shape[0],
                ):
                    # Convert resized PIL depth image back to numpy array for ground plane detection
                    depth_for_plane_detection = np.array(depth_image)
                else:
                    depth_for_plane_detection = depth_array

                # Use the corrected intrinsics for ground plane detection
                ground_plane = detect_ground_plane(
                    depth_m=depth_for_plane_detection,
                    K=camera_intrinsics,
                    camera_pose=pose_for_detection,
                    voxel_size=self.config.ground_plane_voxel_size,
                    dist_thresh=self.config.ground_plane_distance_threshold,
                    max_iter=self.config.ground_plane_max_iterations,
                )
            except Exception as e:
                Console.with_prefix(self.__class__.__name__, "ground_plane").error(
                    e, f"Failed to detect ground plane for frame {idx}"
                )
                ground_plane = None

        ds_out = DatasetOut(
            idx=idx,
            rgb_image=rgb_image,
            depth_image=depth_image,
            camera_intrinsics=camera_intrinsics.copy(),
            camera_pose=camera_pose.copy(),
            ground_plane=ground_plane,
        )
        self._ds_out_cache.update({idx: ds_out})
        return ds_out

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
            return [self._poses_cache[idx]] if idx is not None else self._poses_cache

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
        Console.with_prefix(self.__class__.__name__, "_scale_poses").warn(
            "Scaling poses does *not* make sense for this application! Only use it if normalization is necessary, e.g. for NN inputs or plotting!"
        )
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

        return depth_m.astype(np.float32)

    def __len__(self) -> int:
        """Get number of frames in dataset."""
        return self.parser.get_frame_count()  # type: ignore

    def detect_ground_plane_for_frame(
        self, idx: int
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Detect ground plane for a specific frame.

        Args:
            idx: Frame index.

        Returns:
            Tuple of (normal, d) where the plane equation is: normal^T * x + d = 0
            Returns None if detection fails.
        """
        if not self.config.detect_ground_plane:
            return None

        try:
            # Get depth array and camera data
            depth_array = self.get_depth(idx)
            camera_intrinsics = self.parser.get_intrinsics()

            # If depth is rotated, rotate the intrinsics to match
            if self.config.is_rotated:
                # Since depth_array is already rotated, we need to determine original dimensions
                # After a 90° CW rotation: new_height = old_width, new_width = old_height
                # So: old_width = new_height, old_height = new_width
                h_rotated, w_rotated = depth_array.shape
                w_original = h_rotated
                h_original = w_rotated
                camera_intrinsics = _rotate_intrinsics_90_cw(
                    camera_intrinsics, w_original, h_original
                )

            # Scale intrinsics if depth was resized
            if self.config.resize_depth_to_rgb:
                rgb_image = Image.fromarray(self.get_rgb(idx))
                width, height = rgb_image.size
                camera_intrinsics = self.get_scaled_intrinsics(width, height)

            # Choose coordinate system for ground plane detection
            camera_pose = (
                self.get_poses(idx=idx)[0]
                if self.config.ground_plane_use_world_coords
                else None
            )

            return detect_ground_plane(
                depth_m=depth_array,
                K=camera_intrinsics,
                camera_pose=camera_pose,
                voxel_size=self.config.ground_plane_voxel_size,
                dist_thresh=self.config.ground_plane_distance_threshold,
                max_iter=self.config.ground_plane_max_iterations,
            )
        except Exception as e:
            Console.with_prefix(self.__class__.__name__, "ground_plane").error(
                e, f"Failed to detect ground plane for frame {idx}"
            )
            return None


def detect_ground_plane(
    depth_m: np.ndarray,
    K: np.ndarray,
    camera_pose: np.ndarray | None = None,
    voxel_size: float = 0.05,
    dist_thresh: float = 0.02,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, float]:
    """
    Returns (normal, d) of plane n^T x + d = 0 in *camera* coordinates.

    Args:
        depth_m: Depth map in meters (should already be rotated if needed)
        K: 3x3 camera intrinsics matrix (should match the depth map orientation)
        camera_pose: Optional 4x4 world-to-camera transformation matrix
        voxel_size: Voxel size for downsampling
        dist_thresh: Distance threshold for RANSAC plane detection
        max_iter: Maximum iterations for RANSAC

    Returns:
        Tuple of (normal, d) where the plane equation is: normal^T * x + d = 0
    """
    h, w = depth_m.shape

    # Check if we have enough valid depth values
    valid_depth_mask = (depth_m > 0) & (
        depth_m < 100
    )  # reasonable depth range in meters
    valid_count: int = int(np.sum(valid_depth_mask))

    if valid_count < 10:  # Need at least 10 valid points for meaningful plane fitting
        raise ValueError(f"Insufficient valid depth points: {valid_count} < 10")

    # For very sparse data, use larger voxel size to avoid over-downsampling
    if valid_count < 1000:
        voxel_size = max(voxel_size, 0.1)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )

    # Open3D wants depth in millimetres → scale is 1 / depth_scale
    rgb_dummy = o3d.geometry.Image(np.zeros((h, w, 3), np.uint8))

    # Ensure depth values are in reasonable range and convert to uint16
    depth_mm = np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)
    depth_o3d = o3d.geometry.Image(depth_mm)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_dummy,
        depth_o3d,
        depth_scale=1000.0,  # 1 mm
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Check if point cloud has enough points
    if len(pcd.points) < 3:
        raise ValueError(f"Insufficient points in point cloud: {len(pcd.points)} < 3")

    if camera_pose is not None:  # optional world transform
        pcd.transform(np.linalg.inv(camera_pose))

    # Downsample and check again
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    if len(pcd_downsampled.points) < 3:
        # Try with original point cloud if downsampling removes too many points
        pcd_downsampled = pcd

    if len(pcd_downsampled.points) < 3:
        raise ValueError(
            f"Insufficient points after downsampling: {len(pcd_downsampled.points)} < 3"
        )

    plane_model, inliers = pcd_downsampled.segment_plane(
        distance_threshold=dist_thresh, ransac_n=3, num_iterations=max_iter
    )  # Open3D RANSAC

    n = plane_model[:3].astype(float)
    n /= np.linalg.norm(n)
    d = float(plane_model[3])

    # Robustness improvement: Ensure normal points sky-ward and towards camera
    up = np.array([0, -1, 0])  # Y-axis points down in camera coords, so up is -Y
    fwd = np.array([0, 0, 1])  # Z-axis points forward

    # Make normal point up (negative Y direction) and towards camera (positive Z direction)
    if np.dot(n, up) < 0 or np.dot(n, fwd) < 0:
        n, d = -n, -d

    return n, d
