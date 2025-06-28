from math import atan2, degrees
from typing import Optional, Self, Tuple

import numpy as np
from PIL.Image import Image

from .core import DataModel


class DatasetOut(DataModel):
    """Input data for detection stage."""

    idx: int
    rgb_image: Image
    depth_image: Image
    camera_intrinsics: np.ndarray
    """
    3x3 camera intrinsics matrix K:
                [ fx  0  cx ]
           K =  [  0 fy  cy ]
                [  0  0   1 ]
        where (fx, fy) are the focal lengths in pixels and (cx, cy) is the principal point.
    """
    camera_pose: np.ndarray
    """
    4x4 world-to-camera transformation matrix T_WC in SE(3):

            T_WC = [ R | t ]
                   [ 0 | 1 ]

        where R is the rotation matrix from the unit quaternion (qx,qy,qz,qw)
        and t = [x, y, z]^T is the camera center in world coordinates.
    """
    ground_plane: Optional[Tuple[np.ndarray, float]] = None
    """
    Optional ground plane parameters in camera coordinates.
    Tuple of (normal, d) where the plane equation is: normal^T * x + d = 0
    - normal: 3D unit vector representing the plane normal
    - d: scalar distance parameter
    """

    def rel_move_description(
        self,
        other: "DatasetOut",
        idx_self: int,
        idx_other: int,
    ) -> Optional[str]:
        """
        Describe the camera motion from *self* frame (idx_self) to *other*
        (idx_other). Output: "moved X m, rotated +/- θ ° (CCW positive)".
        """
        if self.camera_pose is None or other.camera_pose is None:
            return None

        T_rel = other.camera_pose @ np.linalg.inv(self.camera_pose)

        # Extract rotation and translation
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        distance = float(np.linalg.norm(t_rel))
        distance = round(distance, 2)

        yaw_rad = atan2(-R_rel[2, 0], R_rel[0, 0])
        yaw_deg = round(degrees(yaw_rad), 1)

        return (
            f"Updated frame {idx_self} to {idx_other}: "
            f"moved {distance:.2f} m, rotated {yaw_deg:+.1f}° "
            f"({'ccw / left' if yaw_deg > 0 else 'cw / right'})"
        )
