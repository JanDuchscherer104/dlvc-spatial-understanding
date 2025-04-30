import os
from pathlib import Path
from typing import Annotated, List, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from ...utils import CONSOLE, BaseConfig, PathConfig


class StrayScannerPaths(BaseConfig):
    """Configuration for Stray Scanner dataset paths."""

    dataset_dir: Annotated[Path, Field(default="baustelle")]
    """Root directory of the Stray Scanner dataset."""

    rgb_video_filename: str = "rgb.mp4"
    """Name of the RGB video file."""

    rgb_dir: str = "rgb"
    """Directory containing individual RGB frames."""

    rgb_rotated_dir: str = "rgb_rotated"
    """Directory containing rotated RGB frames."""

    depth_dir: str = "depth"
    """Directory containing depth maps."""

    confidence_dir: str = "confidence"
    """Directory containing confidence maps."""

    camera_matrix_filename: str = "camera_matrix.csv"
    """Name of the camera intrinsics matrix file."""

    odometry_filename: str = "odometry.csv"
    """Name of the odometry file."""

    imu_filename: str = "imu.csv"
    """Name of the IMU data file."""

    rgb_extension: str = ".jpg"
    """File extension for RGB images."""

    depth_extension: str = ".png"
    """File extension for depth maps."""

    confidence_extension: str = ".png"
    """File extension for confidence maps."""

    def get_rgb_video_path(self) -> Path:
        """Get path to the RGB video file."""
        return self.dataset_dir / self.rgb_video_filename

    def get_camera_matrix_path(self) -> Path:
        """Get path to the camera matrix file."""
        return self.dataset_dir / self.camera_matrix_filename

    def get_odometry_path(self) -> Path:
        """Get path to the odometry file."""
        return self.dataset_dir / self.odometry_filename

    def get_imu_path(self) -> Path:
        """Get path to the IMU data file."""
        return self.dataset_dir / self.imu_filename

    def get_rgb_dir(self) -> Path:
        """Get path to the RGB frames directory."""
        return self.dataset_dir / self.rgb_dir

    def get_rgb_rotated_dir(self) -> Path:
        """Get path to the rotated RGB frames directory."""
        return self.dataset_dir / self.rgb_rotated_dir

    def get_depth_dir(self) -> Path:
        """Get path to the depth maps directory."""
        return self.dataset_dir / self.depth_dir

    def get_confidence_dir(self) -> Path:
        """Get path to the confidence maps directory."""
        return self.dataset_dir / self.confidence_dir

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def validate_dataset_dir(cls, v: str | Path, info: ValidationInfo) -> Path:
        if isinstance(v, str) or isinstance(v, Path) and not Path(v).is_absolute():
            pth = PathConfig().root / ".data" / v
        else:
            pth = Path(v)
        assert pth.exists(), f"Dataset root directory {pth} does not exist."

        return pth.resolve()  # type: ignore

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate that the dataset root contains required files and directories.

        Returns:
            True if dataset is valid, False otherwise.
        """
        # Check if root directory exists
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory {self.dataset_dir} does not exist."
            )

        # Check for required files
        required_files = [
            self.get_camera_matrix_path(),
            self.get_odometry_path(),
            self.get_rgb_video_path(),
            self.get_rgb_dir(),
            self.get_rgb_rotated_dir(),
            self.get_depth_dir(),
        ]

        # At least one of these video/image sources must be available
        missing_files = list(filter(lambda f: not f.exists(), required_files))
        if missing_files:
            CONSOLE.warn(
                f"[red]Required files missing: {', '.join(str(f) for f in missing_files)}"
            )

        return self
