from typing import Optional

import numpy as np
from PIL.Image import Image

from . import DataModel


class PipelineIn(DataModel):
    """Input data for the Dataset stage."""

    idx: int
    user_prompt: Optional[str] = None


class DatasetOut(DataModel):
    """Input data for detection stage."""

    rgb_image: Image
    depth_image: Image
    user_prompt: Optional[str] = None
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
