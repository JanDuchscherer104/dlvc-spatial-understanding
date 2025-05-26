from typing import Annotated, Any, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cv2 import LINE_8
from matplotlib.lines import Line2D
from PIL import Image
from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..utils.console import Console
from . import DataModel

VERTEX_ORDER = [1, 3, 7, 5, 0, 2, 6, 4]


class RawOBBDetection(DataModel):
    label: str = Field(
        ..., description="Unique descriptive label of the detected object."
    )
    box_3d: Annotated[
        List[float],
        Field(
            ...,
            description="3D bounding box parameters: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].",
            # "Center and size are in metric units. Euler angles (roll, pitch, yaw) are in degrees.",
            min_length=9,
            max_length=9,
        ),
    ]


class OBBDetection(DataModel):
    """Processed OBB detection with derived attributes.
    For 3D OBB, 'vertices' would represent the 8 corners of the 3D box.
    """

    label: str
    """Label of the detected object."""

    box_3d: List[float]
    """3D bounding box parameters: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]."""

    height: Optional[float] = Field(None)
    """Height of the 3D bounding box (z_size), in meters (m)."""

    center_point: Optional[np.ndarray] = Field(None)
    """Center point of the bounding box in world coordinates [x, y, z], shape (3,), in meters."""

    center_distance: Optional[float] = Field(None)
    """Distance from the ego origin to the box center projected onto ground plane, in meters (m).
    This represents the 2D Euclidean distance on the XY plane."""

    euler_matrix: Optional[np.ndarray] = Field(None)
    """3x3 rotation matrix derived from the roll, pitch, yaw angles of the box."""

    obb_vertices_3d: Optional[np.ndarray] = Field(None)
    """8 corner vertices of the 3D bounding box in world coordinates, shape (8, 3)."""

    obb_vertices_2d: Optional[np.ndarray] = Field(None)
    """Projected 3D OBB vertices onto the 2D image plane, shape (8, 2). Equiv. to pts2d as returned from project_to_image in the notebook."""

    obb_vertices_depth: Optional[np.ndarray] = Field(None)
    """Depth values for each projected vertex in the 2D image, shape (8,). Equiv. to depths as returned from project_to_image in the notebook."""

    bev_footprint: Optional[np.ndarray] = Field(None)
    """BEV (Bird's-Eye View) 2D OBB as 4 corner vertices in (x, y) meters."""

    bev_xlim: List[float] = Field(default_factory=list)
    """X limits of the BEV bounding box in meters, shape (2,)."""

    bev_ylim: List[float] = Field(default_factory=list)
    """Y limits of the BEV bounding box in meters, shape (2,)."""

    bev_area: Optional[float] = Field(None)
    """Area of the bounding box footprint on the ground plane after yaw rotation, in square meters (m²).
    Calculated using the Shoelace formula on the XY projection of the box."""

    processed_: bool = Field(False)
    """Indicates whether image-dependent processing has been completed.
    This flag is set to True after the process() method has successfully run."""

    @model_validator(mode="after")
    def calculate_intrinsic_fields(self) -> "OBBDetection":
        if self.box_3d and len(self.box_3d) == 9:
            cx, cy, cz, sx, sy, sz, roll, pitch, yaw = self.box_3d

            self.height = float(sz)
            self.center_point = np.array([cx, cy, cz], dtype=float)
            self.center_distance = float(np.hypot(cx, cy))

            # 2D Rotation matrix for yaw
            # Rotated footprint vertices relative to box center (cx, cy)
            phi = np.deg2rad(yaw)
            # Calculate bev_area (Bird's Eye View footprint area)
            Rz_2D = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            self.bev_footprint = (
                np.array(
                    [
                        [-sx / 2, -sy / 2],
                        [sx / 2, -sy / 2],
                        [sx / 2, sy / 2],
                        [-sx / 2, sy / 2],
                    ]
                )
                @ Rz_2D.T
                + self.center_point[:2]
            )

            # # Using Shoelace formula for area of polygon given vertices
            # self.bev_area = 0.5 * float(
            #     np.abs(
            #         x_coords @ np.roll(y_coords, 1) - y_coords @ np.roll(x_coords, 1)
            #     )
            # )

            # Calculate rotation matrix from roll, pitch, yaw (using same method as reference code)
            self.euler_matrix = self._build_rotation_matrix(roll, pitch, yaw)

            # Calculate BEV limits
            self.bev_xlim = [cx - sx / 2, cx + sx / 2]
            self.bev_ylim = [cy - sy / 2, cy + sy / 2]

            self.obb_vertices_3d = self.vertices_from_box3d()
        return self

    def _build_rotation_matrix(
        self, roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """Build rotation matrix using same quaternion method as reference code.

        Args:
            roll, pitch, yaw: Euler angles in degrees

        Returns:
            3x3 rotation matrix
        """
        # Convert to radians and build quaternion (same as reference code)
        r, p, y = np.deg2rad([roll, pitch, yaw])
        sr, sp, sy_ = np.sin([r / 2, p / 2, y / 2])
        cr, cp, cy_ = np.cos([r / 2, p / 2, y / 2])
        q = np.array(
            [
                sr * cp * cy_ - cr * sp * sy_,
                cr * sp * cy_ + sr * cp * sy_,
                cr * cp * sy_ - sr * sp * cy_,
                cr * cp * cy_ + sr * sp * sy_,
            ]
        )

        # Build rotation matrix from quaternion (same as reference code)
        R = np.array(
            [
                [
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                    2 * (q[0] * q[1] - q[3] * q[2]),
                    2 * (q[0] * q[2] + q[3] * q[1]),
                ],
                [
                    2 * (q[0] * q[1] + q[3] * q[2]),
                    1 - 2 * (q[0] ** 2 + q[2] ** 2),
                    2 * (q[1] * q[2] - q[3] * q[0]),
                ],
                [
                    2 * (q[0] * q[2] - q[3] * q[1]),
                    2 * (q[1] * q[2] + q[3] * q[0]),
                    1 - 2 * (q[0] ** 2 + q[1] ** 2),
                ],
            ]
        )
        return R

    def vertices_from_box3d(
        self,
    ) -> np.ndarray:
        """Generate 8 corner vertices of a 3D bounding box from box parameters.

        Returns:
            Array of shape (8, 3) containing the 8 corner vertices in world coordinates
        """
        cx, cy, cz, sx, sy, sz, _, _, _ = self.box_3d
        local = np.array(
            [
                [x, y, z]
                for x in (-sx / 2, sx / 2)
                for y in (-sy / 2, sy / 2)
                for z in (-sz / 2, sz / 2)
            ]
        )
        return (self.euler_matrix @ local[VERTEX_ORDER].T).T + np.array([cx, cy, cz])

    def project_to_image(self, K: np.ndarray):
        """Project 3D camera coordinates to 2D image coordinates.

        Args:
            K: 3x3 camera intrinsics matrix

        Sets:
            - obb_vertices_2d: Array of shape (8, 2) containing 2D image coordinates (NaN for invalid points) - eqiv. to pts2d
            - obb_vertices_depth: Array of shape (8,) containing depth values for each point - eqiv. to depths
        """
        tilt = np.deg2rad(90)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        pts3d = (Rx @ self.obb_vertices_3d.T).T

        proj = (K @ pts3d.T).T
        self.obb_vertices_2d = proj[:, :2] / proj[:, 2, None]
        self.obb_vertices_depth = pts3d[:, 2]

    def process(
        self,
        img_size: Tuple[int, int],
        camera_intrinsics: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
    ) -> None:
        """Process OBB detection using actual camera intrinsics and robust pose handling."""
        if self.processed_:
            return

        self.project_to_image(camera_intrinsics)
        self.processed_ = True


class OBBDetections(DataModel):
    """Container for a list of processed 3D OBB detections."""

    objects: List[OBBDetection] = Field(default_factory=list)
    "List of processed 3D OBB detections."
    visualization: Optional[Any] = None
    "Visualization data for the 3D scene."

    bev_center_x: Optional[float] = Field(None)
    """Center X coordinate of the BEV bounding box in meters. Eqiv to cx_theta."""
    bev_center_y: Optional[float] = Field(None)
    """Center Y coordinate of the BEV bounding box in meters. Eqiv to cy_theta."""
    bev_width: Optional[float] = Field(None)
    """Width of the BEV view. Equiv to dx."""
    bev_height: Optional[float] = Field(None)
    """Height of the BEV view. Equiv to dy."""

    def process_all(
        self,
        img_size: Tuple[int, int],
        camera_intrinsics: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
    ):
        """Processes all raw detections in the list using camera intrinsics.

        Args:
            img_size: (width, height) of the image
            camera_intrinsics: 3x3 camera intrinsics matrix
            camera_pose: 4x4 camera pose transformation (optional)

        Raises:
            ValueError: If any input matrices have incorrect shapes or camera pose is not invertible
        """
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for obj in self.objects:
            obj.process(img_size, camera_intrinsics, camera_pose)
            x_min, x_max = min(x_min, obj.bev_xlim[0]), max(x_max, obj.bev_xlim[1])
            y_min, y_max = min(y_min, obj.bev_ylim[0]), max(y_max, obj.bev_ylim[1])

        # Include ego position (0,0) in bounds calculation like in reference code
        x_min, x_max = min(x_min, 0), max(x_max, 0)
        y_min, y_max = min(y_min, 0), max(y_max, 0)

        self.bev_center_x = (x_min + x_max) / 2.0
        self.bev_center_y = (y_min + y_max) / 2.0
        self.bev_width = max(x_max - x_min, 0.01)
        self.bev_height = max(y_max - y_min, 0.01)

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> OBBDetection:
        return self.objects[index]

    def __iter__(self) -> Iterator[OBBDetection]:
        return iter(self.objects)

    def visualize_3d_boxes(
        self,
        pil_image: Image.Image,
        fov: float = 60.0,
        zoom: float = 1.5,
        figsize: tuple = (12, 5),
        ego_marker_size: float = 0.5,
        show_grid: bool = True,
        use_distinct_bbox_colors: bool = False,
    ):
        """
        Draw a static two-panel visualization of 3D bounding boxes using pre-computed data,
        with a bow-arrow ego marker and BEV frustum lines.

        Parameters
        ----------
        pil_image : PIL.Image
            Background image for the perspective view.
        fov : float
            Horizontal field-of-view in degrees (used only for frustum visualization).
        zoom : float
            BEV zoom multiplier around your data centre.
        figsize : (w, h)
            Figure size for matplotlib.
        ego_marker_size : float
            Half-width of the little bot arrow in world units.
        show_grid : bool
            Whether to display the grid lines in the BEV view.
        use_distinct_bbox_colors : bool
            If True, use distinct colors for each bounding box instead of the default colors.
        """
        if not self.objects:
            print("No objects to visualize")
            return

        # Check that all objects are processed
        if not all(obj.processed_ for obj in self.objects):
            raise ValueError(
                "All objects must be processed before visualization. Call process_all() first."
            )

        img = np.array(pil_image)
        H, W = img.shape[:2]

        # Generate distinct colors for bounding boxes if requested
        if use_distinct_bbox_colors and len(self.objects) > 0:
            # Create visually distinct colors using HSV color space
            import matplotlib.colors as mcolors

            def generate_distinct_colors(n):
                # Generate evenly spaced hues
                HSVs = [(i / n, 0.9, 0.9) for i in range(n)]
                # Convert to RGB
                RGBs = [mcolors.hsv_to_rgb(hsv) for hsv in HSVs]
                return RGBs

            box_colors = generate_distinct_colors(len(self.objects))
        else:
            # Default colors
            box_colors = ["cyan"] * len(self.objects)  # For perspective view

        # Use pre-computed BEV bounds
        cx_theta, cy_theta = self.bev_center_x, self.bev_center_y
        dx, dy = self.bev_width, self.bev_height
        half_x, half_y = (dx / 2) * zoom, (dy / 2) * zoom

        # Create figure
        fig = plt.figure(figsize=figsize)

        # 4a) perspective view
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"Perspective")

        for i, obj in enumerate(self.objects):
            # Use pre-computed projection data
            pts2d = obj.obb_vertices_2d
            depths = obj.obb_vertices_depth
            dmin, dmax = depths.min(), depths.max()

            # Use the generated color if distinct colors are requested
            color = box_colors[i] if use_distinct_bbox_colors else "cyan"

            # Draw wireframe using same logic as reference
            for j in range(4):
                # top face edges
                x0, y0 = pts2d[j]
                x1, y1 = pts2d[(j + 1) % 4]
                w = (
                    4
                    - ((depths[j] + depths[(j + 1) % 4]) / 2 - dmin) / (dmax - dmin) * 3
                )
                ax1.add_line(Line2D([x0, x1], [y0, y1], lw=w, color=color))

                # bottom face edges
                bx0, by0 = pts2d[j + 4]
                bx1, by1 = pts2d[(j + 1) % 4 + 4]
                w2 = (
                    4
                    - ((depths[j + 4] + depths[(j + 1) % 4 + 4]) / 2 - dmin)
                    / (dmax - dmin)
                    * 3
                )
                ax1.add_line(Line2D([bx0, bx1], [by0, by1], lw=w2, color=color))

                # vertical edges
                ax1.add_line(Line2D([x0, bx0], [y0, by0], lw=w, color=color))

            # label
            cx2, cy2 = pts2d.mean(axis=0)
            bbox_color = color if use_distinct_bbox_colors else "blue"
            ax1.text(
                cx2,
                cy2,
                obj.label,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor=bbox_color, alpha=0.6, pad=2),
            )

        # 4b) BEV view
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("equal")
        ax2.set_title(f"Top View (zoom={zoom:.1f}×)")
        ax2.set_xlim(cx_theta - half_x, cx_theta + half_x)
        ax2.set_ylim(cy_theta - half_y, cy_theta + half_y)

        # Add grid only if show_grid is True
        if show_grid:
            # grid
            xt = np.arange(np.floor(cx_theta - half_x), np.ceil(cx_theta + half_x) + 1)
            yt = np.arange(np.floor(cy_theta - half_y), np.ceil(cy_theta + half_y) + 1)
            for x in xt:
                ax2.axvline(x, color="gray", lw=0.5)
            for y in yt:
                ax2.axhline(y, color="gray", lw=0.5)

        # 4c) frustum rays (using FOV parameter for visualization)
        theta = np.deg2rad(fov / 2)
        end_dist = cy_theta + half_y
        for sign in (+1, -1):
            dx_ray = np.sin(sign * theta) * end_dist
            dy_ray = np.cos(theta) * end_dist
            ax2.plot([0, dx_ray], [0, dy_ray], "--", color="blue", lw=1.5)

        # 4d) ego marker as up‐triangle
        ax2.scatter(
            0, 0, marker="^", s=(ego_marker_size * 200) ** 2, color="black", zorder=5
        )
        ax2.text(
            0,
            -ego_marker_size * 3,
            "Ego",
            color="black",
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
        )

        # 4e) draw box footprints using pre-computed BEV footprints
        for i, obj in enumerate(self.objects):
            # Use pre-computed BEV footprint
            rect = obj.bev_footprint

            # Use the generated color or default to magenta
            bev_color = box_colors[i] if use_distinct_bbox_colors else "magenta"

            ax2.add_patch(
                plt.Polygon(rect, closed=True, fill=False, edgecolor=bev_color, lw=2)
            )

            # Label at center of box
            cx, cy = obj.center_point[:2]
            ax2.text(
                cx,
                cy,
                obj.label,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor=bev_color, alpha=0.6, pad=2),
            )

        plt.tight_layout()
        plt.show()

    def to_boxes_json(self) -> List[dict]:
        """
        Convert OBBDetections to the JSON format expected by the original visualize_3d_boxes function.

        Returns:
            List of dictionaries with 'label' and 'box_3d' keys.
        """
        return [{"label": obj.label, "box_3d": obj.box_3d} for obj in self.objects]
