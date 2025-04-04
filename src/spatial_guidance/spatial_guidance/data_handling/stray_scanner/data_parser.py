from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Type, Union, cast

import cv2
import numpy as np
import open3d as o3d
import skvideo.io
from PIL import Image
from pydantic import Field, model_validator
from regex import T
from scipy.spatial.transform import Rotation

from utils import CONSOLE, BaseConfig

from .stray_scanner_paths import StrayScannerPaths


class StrayScannerDataParserConfig(BaseConfig["StrayScannerDataParser"]):
    """Configuration for Stray Scanner data parser.
    https://github.com/strayrobots/scanner/blob/main/docs/format.md
    """

    target: Type["StrayScannerDataParser"] = Field(
        default_factory=lambda: StrayScannerDataParser
    )
    """Target class to instantiate."""

    paths: StrayScannerPaths = Field(default_factory=StrayScannerPaths)
    """Paths configuration for the dataset."""

    depth_unit_scale_factor: float = 1e-3
    """Factor to convert depth values to meters (1e-3 converts mm to m)."""


class StrayScannerDataParser:
    """Parser for Stray Scanner datasets focusing on essential functionality."""

    def __init__(self, config: StrayScannerDataParserConfig):
        """Initialize the data parser with validated configuration.

        Args:
            config: Configuration for the parser.
        """
        self.config = config

        # Lazy loaded properties - initialize with proper types
        self._intrinsics: Optional[np.ndarray] = None
        self._poses: Optional[List[np.ndarray]] = None
        self._imu_data: Optional[np.ndarray] = None
        self._frame_count: Optional[int] = None
        self._rgb_frames: Optional[List[Path]] = None
        self._rgb_frames_rotated: Optional[List[Path]] = None
        self._depth_frames: Optional[List[Path]] = None

        # Image dimensions - cached once detected
        self._rgb_dimensions: Optional[Tuple[int, int]] = None
        self._depth_dimensions: Optional[Tuple[int, int]] = None

    def _resize_camera_matrix(
        self, camera_matrix: np.ndarray, scale_x: float, scale_y: float
    ) -> np.ndarray:
        """Resize camera intrinsics matrix based on scaling factors."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        return np.array(  # type: ignore
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y],
                [0.0, 0.0, 1.0],
            ]
        )

    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix."""
        if self._intrinsics is None:
            self._intrinsics = np.loadtxt(
                self.config.paths.get_camera_matrix_path(), delimiter=","
            )
            assert isinstance(self._intrinsics, np.ndarray)
            assert self._intrinsics.shape == (3, 3), "Camera matrix must be 3x3"

        return self._intrinsics.copy()

    def get_poses(self) -> List[np.ndarray]:
        """Get camera poses from odometry data."""
        if self._poses is None:
            odometry = np.loadtxt(
                self.config.paths.get_odometry_path(), delimiter=",", skiprows=1
            )
            poses: List[np.ndarray] = []

            for line in odometry:
                # Format: timestamp, frame_number, x, y, z, qx, qy, qz, qw
                if len(line) > 7:  # Format with timestamp, frame number
                    position = line[2:5]
                    quaternion = line[5:9]
                else:  # Simple format: x, y, z, qx, qy, qz, qw
                    position = line[:3]
                    quaternion = line[3:7]

                T_WC = np.eye(4)
                T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
                T_WC[:3, 3] = position
                poses.append(T_WC)

            self._poses = poses

        return self._poses.copy()

    def get_frame_count(self) -> int:
        """Get the number of frames in the dataset."""
        if self._frame_count is None:
            # Use poses as primary reference for frame count
            poses = self.get_poses()
            self._frame_count = len(poses)

        return self._frame_count

    def get_imu_data(self) -> Optional[np.ndarray]:
        """Get IMU sensor data if available."""
        if self._imu_data is None:
            imu_path = self.config.paths.get_imu_path()
            if imu_path.exists():
                self._imu_data = np.loadtxt(imu_path, delimiter=",", skiprows=1)
            else:
                return None

        return self._imu_data.copy() if self._imu_data is not None else None

    def get_rgb_frames(self) -> List[Path]:
        """Get all RGB frame paths sorted."""
        if self._rgb_frames is None:
            rgb_dir = self.config.paths.get_rgb_dir()
            if not rgb_dir.exists():
                CONSOLE.warn(
                    f"[red]RGB directory {rgb_dir} does not exist. "
                    "Falling back to RGB video."
                )
                return []
            self._rgb_frames = sorted(
                rgb_dir.glob(f"*{self.config.paths.rgb_extension}")
            )

        return self._rgb_frames

    def get_rgb_rotated_frames(self) -> List[Path]:
        """Get all rotated RGB frame paths sorted."""
        if self._rgb_frames_rotated is None:
            rgb_rotated_dir = self.config.paths.get_rgb_rotated_dir()
            if not rgb_rotated_dir.exists():
                CONSOLE.warn(
                    f"[red]Rotated RGB directory {rgb_rotated_dir} does not exist. "
                    "Falling back to RGB directory."
                )
                return []
            self._rgb_frames_rotated = sorted(
                rgb_rotated_dir.glob(f"*{self.config.paths.rgb_extension}")
            )
        return self._rgb_frames_rotated

    def get_depth_frames(self) -> List[Path]:
        """Get all depth frame paths sorted."""
        if self._depth_frames is None:
            depth_dir = self.config.paths.get_depth_dir()
            if not depth_dir.exists():
                CONSOLE.warn(
                    f"[red]Depth directory {depth_dir} does not exist. "
                    "Falling back to RGB directory."
                )
                return []
            self._depth_frames = sorted(
                depth_dir.glob(f"*{self.config.paths.depth_extension}")
            )
        return self._depth_frames

    def get_depth_frame(self, idx: int) -> np.ndarray:
        """Load depth map for a specific frame.

        Args:
            idx: Frame index.

        Returns:
            Depth map as numpy array in meters.
        """
        depth_frames = self.get_depth_frames()

        if not depth_frames or idx >= len(depth_frames):
            raise IndexError(
                f"Depth frame index {idx} out of range (found {len(depth_frames)} frames)"
            )

        depth_path = depth_frames[idx]

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # Use cv2 for depth image loading which handles 16-bit PNGs properly
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if depth_mm is None:
            raise IOError(f"Failed to load depth image: {depth_path}")

        # Convert to float32 and apply scale factor to get meters
        depth_m = depth_mm.astype(np.float32) * self.config.depth_unit_scale_factor

        return depth_m
