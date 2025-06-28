"""Scene visualization pipeline stage for displaying detection results.

This module provides a pipeline stage for visualizing detection results
alongside depth data for better spatial understanding. It supports different
visualization types including standard detection, 3D bounding boxes, segmentation
masks, and birds-eye-view (BEV).
"""

import base64
import io
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pydantic import Field
from zenml.steps import BaseStep

from ..pipeline.data_contracts import AABBDetection, VisualizationIn, VisualizationOut
from ..pipeline.data_contracts_3d import (
    Box3D,
    CombinedDetectionSegmentationOut,
    Detection3DStageOut,
    MultiviewPoint,
    MultiviewPointsStageOut,
    Point2D,
    PointDetectionStageOut,
    SegmentationMaskData,
    SegmentationStageOut,
)
from ..utils import BaseConfig, Console


class VisualizationType(str, Enum):
    """Defines different visualization types."""

    STANDARD = "standard"
    BOXES_3D = "3d_bounding_boxes"
    BIRDS_EYE_VIEW = "birds_eye_view"
    SEGMENTATION = "segmentation"
    COMBINED = "segmentation_and_3d"
    POINTS = "points"
    MULTIVIEW_POINTS = "multiview_points"


class SceneVisualizerConfig(BaseConfig["SceneVisualizer"]):
    """Configuration for the scene visualization stage."""

    # Visualization-specific configuration
    show_debug_info: bool = Field(
        True, description="Whether to show debug information like coordinates"
    )
    figure_size: Tuple[int, int] = Field(
        (16, 7), description="Size of the output figure (width, height)"
    )
    colorful_depth: bool = Field(
        True, description="Whether to use colorful depth visualization"
    )
    point_radius: int = Field(7, description="Radius of the points drawn on objects")
    target: Type["SceneVisualizer"] = Field(
        default_factory=lambda: SceneVisualizer,
        description="Target class to instantiate",
    )
    dpi: int = Field(100, description="DPI for rendering the output image")

    # 3D visualization options
    fov: float = Field(
        60.0, description="Field of view in degrees for 3D visualization"
    )
    zoom: float = Field(1.5, description="Zoom factor for birds-eye-view")
    ego_marker_size: float = Field(0.5, description="Size of ego marker in BEV")
    show_grid: bool = Field(True, description="Show grid in birds-eye-view")
    use_distinct_bbox_colors: bool = Field(
        True, description="Use distinct colors for bounding boxes"
    )

    # Segmentation options
    segmentation_alpha: float = Field(
        0.5, description="Alpha transparency for segmentation masks"
    )
    font_size: int = Field(16, description="Font size for labels")
    show_boxes: bool = Field(
        True, description="Show bounding boxes around segmentation masks"
    )

    # Visualization type
    visualization_types: List[VisualizationType] = Field(
        [VisualizationType.STANDARD], description="Types of visualization to perform"
    )

    def setup_target(self) -> "SceneVisualizer":
        return self.target(self)


class SceneVisualizer(BaseStep):
    """Scene visualization pipeline stage.

    This stage takes RGB and depth frames along with detection results,
    and produces a visualization of the scene with detected objects.
    """

    def __init__(
        self, config: Optional[SceneVisualizerConfig] = None, **step_kwargs: Any
    ) -> None:
        """Initialize the scene visualizer.

        Args:
            config: Configuration for the scene visualizer
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        super().__init__(**step_kwargs)
        # config = config or SceneVisualizerConfig()

        # Store visualization-specific configuration
        config = config or SceneVisualizerConfig()
        self.show_debug_info = config.show_debug_info
        self.figure_size = config.figure_size
        self.colorful_depth = config.colorful_depth
        self.point_radius = config.point_radius
        self.dpi = config.dpi

        # 3D visualization options
        self.fov = config.fov
        self.zoom = config.zoom
        self.ego_marker_size = config.ego_marker_size
        self.show_grid = config.show_grid
        self.use_distinct_bbox_colors = config.use_distinct_bbox_colors

        # Segmentation options
        self.segmentation_alpha = config.segmentation_alpha
        self.font_size = config.font_size
        self.show_boxes = config.show_boxes

        # Visualization types
        self.visualization_types = config.visualization_types

        CONSOLE.log(
            f"Initialized SceneVisualizer with debug={self.show_debug_info}, types={self.visualization_types}"
        )

    def entrypoint(self, input_data: Any) -> VisualizationOut:
        """Process the input data to produce a visualization.

        Args:
            input_data: Contains visualization data specific to the visualization type

        Returns:
            VisualizationOutput containing the visualization as a PIL Image
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "entrypoint")
        vis_image = None
        object_count = 0

        # Handle standard detection visualization
        if VisualizationType.STANDARD in self.visualization_types and hasattr(
            input_data, "detection_output"
        ):
            rgb_frame = np.array(input_data.rgb_image)
            depth_frame = np.array(input_data.depth_image)
            vis_image = self.visualize_scene(
                rgb_frame, depth_frame, input_data.detection_output.objects
            )
            object_count = len(input_data.detection_output.objects)

        # Handle 3D bounding box visualization
        elif VisualizationType.BOXES_3D in self.visualization_types and hasattr(
            input_data, "boxes_3d"
        ):
            vis_image = self.visualize_3d_boxes(
                input_data.rgb_image, input_data.boxes_3d
            )
            object_count = len(input_data.boxes_3d)

        # Handle birds-eye view (handled as part of BOXES_3D)
        elif VisualizationType.BIRDS_EYE_VIEW in self.visualization_types and hasattr(
            input_data, "boxes_3d"
        ):
            vis_image = self.visualize_3d_boxes(
                input_data.rgb_image, input_data.boxes_3d
            )
            object_count = len(input_data.boxes_3d)

        # Handle segmentation visualization
        elif VisualizationType.SEGMENTATION in self.visualization_types and hasattr(
            input_data, "segmentation"
        ):
            vis_image = self.visualize_segmentation(
                input_data.rgb_image, input_data.segmentation
            )
            object_count = len(input_data.segmentation)

        # Handle combined visualization
        elif (
            VisualizationType.COMBINED in self.visualization_types
            and hasattr(input_data, "segmentation")
            and hasattr(input_data, "boxes_3d")
        ):
            vis_image = self.visualize_combined(
                input_data.rgb_image, input_data.segmentation, input_data.boxes_3d
            )
            object_count = len(input_data.segmentation) + len(input_data.boxes_3d)

        # Handle point visualization
        elif VisualizationType.POINTS in self.visualization_types and hasattr(
            input_data, "points"
        ):
            vis_image = self.visualize_points(input_data.rgb_image, input_data.points)
            object_count = len(input_data.points)

        # Handle multiview point visualization
        elif VisualizationType.MULTIVIEW_POINTS in self.visualization_types and hasattr(
            input_data, "reference_points"
        ):
            vis_image = self.visualize_multiview_points(
                input_data.reference_image,
                input_data.reference_points,
                input_data.new_image,
                input_data.tracked_points,
            )
            object_count = len(input_data.tracked_points)

        # Default to standard visualization if no specialized visualization or type matches
        else:
            CONSOLE.warn(
                "No matching visualization type found, defaulting to standard visualization"
            )
            # Check if we have standard VisualizationIn data
            if (
                hasattr(input_data, "rgb_image")
                and hasattr(input_data, "depth_image")
                and hasattr(input_data, "detection_output")
            ):
                rgb_frame = np.array(input_data.rgb_image)
                depth_frame = np.array(input_data.depth_image)
                vis_image = self.visualize_scene(
                    rgb_frame, depth_frame, input_data.detection_output.objects
                )
                object_count = len(input_data.detection_output.objects)
            else:
                # Create an error message image if we can't visualize
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.text(
                    0.5,
                    0.5,
                    "Error: Unsupported visualization type or missing data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="red",
                )
                ax.axis("off")

                # Convert to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=self.dpi)
                buf.seek(0)
                vis_image = Image.open(buf)
                plt.close(fig)
                object_count = 0

        return VisualizationOut(
            visualization=vis_image,
            object_count=object_count,
        )

    def visualize_scene(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        objects: List[AABBDetection],
    ) -> Image.Image:
        """
        Visualize a scene with RGB and depth frames, showing detected objects.

        Args:
            rgb_frame: RGB image as numpy array
            depth_frame: Depth image as numpy array
            objects: List of DetectedObject instances

        Returns:
            PIL Image containing the visualization
        """
        # Convert RGB frame to PIL Image
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb_frame)
        rgb_width, rgb_height = rgb_img.size

        # Create depth visualization
        depth_img = self._create_depth_visualization(depth_frame, rgb_width, rgb_height)

        # Colors for objects
        colors = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta", "lime"]

        # Create drawing objects
        rgb_draw = ImageDraw.Draw(rgb_img)
        depth_draw = ImageDraw.Draw(depth_img)

        # Draw each object
        for i, obj in enumerate(objects):
            color = colors[i % len(colors)]
            self._draw_object(obj, rgb_draw, depth_draw, rgb_width, rgb_height, color)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        ax1.imshow(np.array(rgb_img))
        ax1.set_title(f"RGB View ({rgb_width}x{rgb_height})")
        ax1.axis("off")

        ax2.imshow(np.array(depth_img))
        ax2.set_title(f"Depth View ({rgb_width}x{rgb_height})")
        ax2.axis("off")

        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        return img

    def _create_depth_visualization(
        self, depth_frame: np.ndarray, target_width: int, target_height: int
    ) -> Image.Image:
        """Create a visualization of the depth frame.

        Args:
            depth_frame: Depth frame as a numpy array
            target_width: Target width for the output image
            target_height: Target height for the output image

        Returns:
            PIL Image with the depth visualization
        """
        if depth_frame is not None and np.any(depth_frame > 0):
            depth_vis = depth_frame.copy()

            # Filter out zeros (no depth data) for better normalization
            valid_depth = depth_vis[depth_vis > 0]

            # Find min and max of valid depth values for better contrast
            min_depth = np.min(valid_depth)
            max_depth = np.max(valid_depth)

            # Add small epsilon to avoid division by zero
            depth_range = max_depth - min_depth
            if depth_range < 0.001:  # Handle case where all depths are similar
                depth_range = 0.001

            # Apply contrast enhancement and invert (closer = brighter)
            depth_vis = np.where(
                depth_vis > 0,  # Only modify valid depth pixels
                255
                - (
                    (depth_vis - min_depth) / depth_range * 255
                ),  # Invert: closer is brighter
                0,  # Leave invalid depth as 0
            ).astype(np.uint8)

            if self.colorful_depth:
                # Create a more informative colormap (blue=far, red=close)
                depth_rgb = np.zeros(
                    (depth_vis.shape[0], depth_vis.shape[1], 3), dtype=np.uint8
                )
                depth_rgb[:, :, 0] = depth_vis  # Red channel (close objects)
                depth_rgb[:, :, 2] = 255 - depth_vis  # Blue channel (far objects)
                depth_img = Image.fromarray(depth_rgb)
            else:
                # Grayscale visualization
                depth_img = Image.fromarray(depth_vis, mode="L").convert("RGB")

            # Resize to match RGB dimensions
            depth_img = depth_img.resize((target_width, target_height), Image.BILINEAR)
        else:
            depth_img = Image.new("RGB", (target_width, target_height), color="black")

        return depth_img

    def _draw_object(
        self,
        obj: AABBDetection,
        rgb_draw: ImageDraw.Draw,
        depth_draw: ImageDraw.Draw,
        img_width: int,
        img_height: int,
        color: str,
    ) -> None:
        """Draw a single object on both RGB and depth images.

        Args:
            obj: The detected object to draw
            rgb_draw: ImageDraw object for the RGB image
            depth_draw: ImageDraw object for the depth image
            img_width: Width of the image
            img_height: Height of the image
            color: Color to use for drawing this object
        """
        # Get bounding box coordinates
        y1, x1, y2, x2 = obj.box_2d

        # Convert normalized coordinates (0-1000) to pixel coordinates
        abs_y1 = int(y1 / 1000 * img_height)
        abs_x1 = int(x1 / 1000 * img_width)
        abs_y2 = int(y2 / 1000 * img_height)
        abs_x2 = int(x2 / 1000 * img_width)

        # Draw bounding boxes
        rgb_draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)
        depth_draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
        )

        # Show coordinates for debugging
        if self.show_debug_info:
            coord_text = f"({x1},{y1})-({x2},{y2})"
            rgb_draw.text(
                (abs_x1, abs_y2 + 5),
                coord_text,
                fill="white",
                stroke_fill="black",
                stroke_width=1,
            )

        # Draw points
        for py, px in obj.points_2d:
            abs_px = int(px / 1000 * img_width)
            abs_py = int(py / 1000 * img_height)

            # Draw points on both images
            rgb_draw.ellipse(
                (
                    (abs_px - self.point_radius, abs_py - self.point_radius),
                    (abs_px + self.point_radius, abs_py + self.point_radius),
                ),
                fill=color,
            )
            depth_draw.ellipse(
                (
                    (abs_px - self.point_radius, abs_py - self.point_radius),
                    (abs_px + self.point_radius, abs_py + self.point_radius),
                ),
                fill=color,
            )

            # Show point coordinates for debugging
            if self.show_debug_info:
                rgb_draw.text(
                    (abs_px + self.point_radius, abs_py + self.point_radius),
                    f"({px},{py})",
                    fill="white",
                    stroke_fill="black",
                    stroke_width=1,
                )

        # Create label
        label = f"{obj.label}"
        if hasattr(obj, "approx_distance") and obj.approx_distance is not None:
            label += f" ({obj.approx_distance:.1f}m)"
        if hasattr(obj, "is_hazard") and obj.is_hazard:
            label += " HAZARD"
            if hasattr(obj, "hazard_type") and obj.hazard_type:
                label += f" ({obj.hazard_type})"

        # Draw labels
        rgb_draw.text((abs_x1 + 5, abs_y1 + 5), label, fill=color)
        depth_draw.text((abs_x1 + 5, abs_y1 + 5), label, fill=color)

    def visualize_3d_boxes(
        self, pil_image: Image.Image, boxes: List[Box3D]
    ) -> Image.Image:
        """
        Draw a two-panel visualization of 3D bounding boxes with a perspective view and BEV.

        Args:
            pil_image: Background image for the perspective view
            boxes: List of Box3D objects

        Returns:
            PIL Image containing the visualization
        """
        img = np.array(pil_image)
        H, W = img.shape[:2]

        # Generate distinct colors for bounding boxes if requested
        if self.use_distinct_bbox_colors and len(boxes) > 0:
            # Create visually distinct colors using HSV color space
            import matplotlib.colors as mcolors

            def generate_distinct_colors(n):
                # Generate evenly spaced hues
                HSVs = [(i / n, 0.9, 0.9) for i in range(n)]
                # Convert to RGB
                RGBs = [mcolors.hsv_to_rgb(hsv) for hsv in HSVs]
                return RGBs

            box_colors = generate_distinct_colors(len(boxes))
        else:
            # Default colors
            box_colors = ["cyan"] * len(boxes)  # For perspective view
            bev_colors = ["magenta"] * len(boxes)  # For BEV view

        # Helper: project one box into image plane
        def project_to_image(box3d):
            cx, cy, cz = box3d.center
            sx, sy, sz = box3d.size
            roll, pitch, yaw = box3d.rotation

            # Build quaternion from rpy
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
            corners = np.array(
                [
                    [x, y, z]
                    for x in (-sx / 2, sx / 2)
                    for y in (-sy / 2, sy / 2)
                    for z in (-sz / 2, sz / 2)
                ]
            )
            order = [1, 3, 7, 5, 0, 2, 6, 4]
            pts3d = (R @ corners[order].T).T + np.array([cx, cy, cz])

            # Tilt camera down 90° so world‐Y→cam‐Z
            tilt = np.deg2rad(90)
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            pts3d = (Rx @ pts3d.T).T

            # Intrinsics
            f = W / (2 * np.tan(np.deg2rad(self.fov) / 2))
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
            proj = (K @ pts3d.T).T
            pts2d = proj[:, :2] / proj[:, 2, None]
            depths = pts3d[:, 2]
            return pts2d, depths

        # Figure out BEV bounds
        xs, ys = [], []
        for box in boxes:
            cx, cy, cz = box.center
            sx, sy, sz = box.size
            xs += [cx - sx / 2, cx + sx / 2]
            ys += [cy - sy / 2, cy + sy / 2]
        xs.append(0)
        ys.append(0)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx, dy = max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3)
        cx_theta, cy_theta = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_x, half_y = (dx / 2) * self.zoom, (dy / 2) * self.zoom

        # Create figure
        fig = plt.figure(figsize=self.figure_size)

        # Perspective view
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"Perspective (FOV={self.fov:.0f}°)")

        for i, box in enumerate(boxes):
            pts2d, depths = project_to_image(box)
            dmin, dmax = depths.min(), depths.max()

            # Use the generated color if distinct colors are requested
            color = box_colors[i] if self.use_distinct_bbox_colors else "cyan"

            for j in range(4):
                # Top
                x0, y0 = pts2d[j]
                x1, y1 = pts2d[(j + 1) % 4]
                w = (
                    4
                    - ((depths[j] + depths[(j + 1) % 4]) / 2 - dmin) / (dmax - dmin) * 3
                    if dmax > dmin
                    else 4
                )
                ax1.add_line(Line2D([x0, x1], [y0, y1], lw=w, color=color))

                # Bottom
                bx0, by0 = pts2d[j + 4]
                bx1, by1 = pts2d[(j + 1) % 4 + 4]
                w2 = (
                    4
                    - ((depths[j + 4] + depths[(j + 1) % 4 + 4]) / 2 - dmin)
                    / (dmax - dmin)
                    * 3
                    if dmax > dmin
                    else 4
                )
                ax1.add_line(Line2D([bx0, bx1], [by0, by1], lw=w2, color=color))

                # Vertical
                ax1.add_line(Line2D([x0, bx0], [y0, by0], lw=w, color=color))

            # Label
            cx2, cy2 = pts2d.mean(axis=0)
            bbox_color = color if self.use_distinct_bbox_colors else "blue"
            ax1.text(
                cx2,
                cy2,
                box.label,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor=bbox_color, alpha=0.6, pad=2),
            )

        # BEV
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("equal")
        ax2.set_title(f"Bird's Eye View (zoom={self.zoom:.1f}×)")
        ax2.set_xlim(cx_theta - half_x, cx_theta + half_x)
        ax2.set_ylim(cy_theta - half_y, cy_theta + half_y)

        # Add grid
        if self.show_grid:
            xt = np.arange(np.floor(cx_theta - half_x), np.ceil(cx_theta + half_x) + 1)
            yt = np.arange(np.floor(cy_theta - half_y), np.ceil(cy_theta + half_y) + 1)
            for x in xt:
                ax2.axvline(x, color="gray", lw=0.5)
            for y in yt:
                ax2.axhline(y, color="gray", lw=0.5)

        # Frustum rays
        theta = np.deg2rad(self.fov / 2)
        end_dist = cy_theta + half_y
        for sign in (+1, -1):
            dx = np.sin(sign * theta) * end_dist
            dy = np.cos(theta) * end_dist
            ax2.plot([0, dx], [0, dy], "--", color="blue", lw=1.5)

        # Ego marker as up‐triangle
        ax2.scatter(
            0,
            0,
            marker="^",
            s=(self.ego_marker_size * 200) ** 2,
            color="black",
            zorder=5,
        )
        ax2.text(
            0,
            -self.ego_marker_size * 3,
            "Ego",
            color="black",
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
        )

        # Draw box footprints
        for i, box in enumerate(boxes):
            cx, cy, cz = box.center
            sx, sy, sz = box.size
            roll, pitch, yaw = box.rotation

            phi = np.deg2rad(yaw)
            Rz = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            rect = (
                np.array(
                    [
                        [-sx / 2, -sy / 2],
                        [sx / 2, -sy / 2],
                        [sx / 2, sy / 2],
                        [-sx / 2, sy / 2],
                    ]
                )
                @ Rz.T
            )
            rect += np.array([cx, cy])

            # Use the generated color or default to magenta
            bev_color = box_colors[i] if self.use_distinct_bbox_colors else "magenta"

            ax2.add_patch(
                plt.Polygon(rect, closed=True, fill=False, edgecolor=bev_color, lw=2)
            )
            ax2.text(
                cx,
                cy,
                box.label,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor=bev_color, alpha=0.6, pad=2),
            )

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        return img

    def visualize_segmentation(
        self, pil_image: Image.Image, segmentation_masks: List[SegmentationMaskData]
    ) -> Image.Image:
        """
        Visualize segmentation masks on an image.

        Args:
            pil_image: Input PIL Image
            segmentation_masks: List of SegmentationMaskData objects

        Returns:
            PIL Image with segmentation visualization
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "visualize_segmentation")

        img_width, img_height = pil_image.size
        result_img = pil_image.copy()

        # Define colors for different masks
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "brown",
            "gray",
            "beige",
            "turquoise",
            "cyan",
            "magenta",
            "lime",
            "navy",
            "maroon",
            "teal",
            "olive",
            "coral",
            "lavender",
            "violet",
            "gold",
            "silver",
        ]

        # Try to create a font with the specified size
        try:
            font = ImageFont.truetype("Arial", self.font_size)
        except IOError:
            # Fall back to default font if Arial is not available
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(result_img)

        # Process and draw each segmentation mask
        for i, mask_data in enumerate(segmentation_masks):
            color = colors[i % len(colors)]

            # Get normalized bounding box coordinates
            y_min, x_min, y_max, x_max = mask_data.box_2d

            # Convert to pixel coordinates
            abs_y_min = int(y_min / 1000 * img_height)
            abs_x_min = int(x_min / 1000 * img_width)
            abs_y_max = int(y_max / 1000 * img_height)
            abs_x_max = int(x_max / 1000 * img_width)

            # Process mask data
            if mask_data.mask.startswith("data:image/png;base64,"):
                try:
                    # Extract base64 content
                    png_str = mask_data.mask.removeprefix("data:image/png;base64,")
                    png_bytes = base64.b64decode(png_str)
                    mask_img = Image.open(io.BytesIO(png_bytes))

                    # Resize mask to fit bounding box
                    bbox_height = abs_y_max - abs_y_min
                    bbox_width = abs_x_max - abs_x_min

                    if bbox_height > 0 and bbox_width > 0:
                        # Resize with antialiasing
                        mask = mask_img.resize((bbox_width, bbox_height), Image.LANCZOS)

                        # Create full-size mask
                        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        np_mask[abs_y_min:abs_y_max, abs_x_min:abs_x_max] = np.array(
                            mask
                        )

                        # Apply mask overlay
                        color_rgb = ImageColor.getrgb(color)
                        alpha_int = int(self.segmentation_alpha * 255)
                        overlay_color_rgba = color_rgb + (alpha_int,)

                        # Create colored mask overlay
                        colored_mask_layer = np.zeros(
                            (img_height, img_width, 4), dtype=np.uint8
                        )
                        mask_binary = np_mask > 127
                        colored_mask_layer[mask_binary] = overlay_color_rgba

                        # Convert to PIL and composite
                        mask_layer_pil = Image.fromarray(colored_mask_layer, "RGBA")
                        result_img = Image.alpha_composite(
                            result_img.convert("RGBA"), mask_layer_pil
                        )

                except Exception as e:
                    CONSOLE.warn(f"Error processing mask: {e}")

            # Draw bounding box if requested
            if self.show_boxes:
                draw = ImageDraw.Draw(result_img)
                draw.rectangle(
                    ((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)),
                    outline=color,
                    width=3,
                )

            # Add label
            draw.text(
                (abs_x_min + 5, abs_y_min + 5), mask_data.label, fill=color, font=font
            )

        return result_img

    def visualize_combined(
        self,
        pil_image: Image.Image,
        segmentation_masks: List[SegmentationMaskData],
        boxes_3d: List[Box3D],
    ) -> Image.Image:
        """
        Combined visualization showing both segmentation masks and 3D bounding boxes.

        Args:
            pil_image: Input PIL Image
            segmentation_masks: List of segmentation masks
            boxes_3d: List of 3D bounding boxes

        Returns:
            PIL Image with combined visualization
        """
        img_np = np.array(pil_image)
        H, W = img_np.shape[:2]

        # Generate distinct colors
        import matplotlib.colors as mcolors

        def generate_distinct_colors(n):
            HSVs = [(i / n, 0.9, 0.9) for i in range(n)]
            RGBs = [mcolors.hsv_to_rgb(hsv) for hsv in HSVs]
            return RGBs

        # Use same colors for both segmentation and 3D boxes for consistency
        num_colors = max(len(segmentation_masks), len(boxes_3d))
        distinct_colors = generate_distinct_colors(num_colors) if num_colors > 0 else []

        # First apply segmentation masks
        result_img = self.visualize_segmentation(pil_image, segmentation_masks)

        # Create figure for two-panel visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # Left panel: Image with segmentation
        ax1.imshow(np.array(result_img))
        ax1.set_title("Segmentation & 3D Boxes")
        ax1.axis("off")

        # Helper function for projecting 3D box to 2D
        def project_box(box):
            cx, cy, cz = box.center
            sx, sy, sz = box.size
            roll, pitch, yaw = box.rotation

            # Build quaternion from rpy
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
            corners = np.array(
                [
                    [x, y, z]
                    for x in (-sx / 2, sx / 2)
                    for y in (-sy / 2, sy / 2)
                    for z in (-sz / 2, sz / 2)
                ]
            )
            order = [1, 3, 7, 5, 0, 2, 6, 4]
            pts3d = (R @ corners[order].T).T + np.array([cx, cy, cz])

            # Tilt camera down 90° so world‐Y→cam‐Z
            tilt = np.deg2rad(90)
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            pts3d = (Rx @ pts3d.T).T

            # Intrinsics
            f = W / (2 * np.tan(np.deg2rad(self.fov) / 2))
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
            proj = (K @ pts3d.T).T
            pts2d = proj[:, :2] / proj[:, 2, None]
            depths = pts3d[:, 2]
            return pts2d, depths

        # Draw 3D bounding boxes on the segmentation image
        for i, box in enumerate(boxes_3d):
            pts2d, depths = project_box(box)
            dmin, dmax = depths.min(), depths.max()

            # Get color for consistency with segmentation
            color = (
                distinct_colors[i % len(distinct_colors)]
                if distinct_colors
                else [0, 1, 1]
            )
            color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"

            # Draw box edges with depth-based thickness
            for j in range(4):
                # Top lines
                x0, y0 = pts2d[j]
                x1, y1 = pts2d[(j + 1) % 4]
                w = (
                    4
                    - ((depths[j] + depths[(j + 1) % 4]) / 2 - dmin) / (dmax - dmin) * 3
                    if dmax > dmin
                    else 4
                )
                ax1.add_line(Line2D([x0, x1], [y0, y1], lw=w, color=color_hex))

                # Bottom lines
                bx0, by0 = pts2d[j + 4]
                bx1, by1 = pts2d[(j + 1) % 4 + 4]
                w2 = (
                    4
                    - ((depths[j + 4] + depths[(j + 1) % 4 + 4]) / 2 - dmin)
                    / (dmax - dmin)
                    * 3
                    if dmax > dmin
                    else 4
                )
                ax1.add_line(Line2D([bx0, bx1], [by0, by1], lw=w2, color=color_hex))

                # Vertical lines
                ax1.add_line(Line2D([x0, bx0], [y0, by0], lw=w, color=color_hex))

        # Right panel: BEV visualization
        ax2.set_aspect("equal")
        ax2.set_title(f"Bird's Eye View (zoom={self.zoom:.1f}×)")

        # Calculate BEV bounds
        xs, ys = [], []
        for box in boxes_3d:
            cx, cy, cz = box.center
            sx, sy, sz = box.size
            xs += [cx - sx / 2, cx + sx / 2]
            ys += [cy - sy / 2, cy + sy / 2]

        xs.append(0)
        ys.append(0)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx, dy = max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3)
        cx_theta, cy_theta = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_x, half_y = (dx / 2) * self.zoom, (dy / 2) * self.zoom

        ax2.set_xlim(cx_theta - half_x, cx_theta + half_x)
        ax2.set_ylim(cy_theta - half_y, cy_theta + half_y)

        # Add grid if requested
        if self.show_grid:
            xt = np.arange(np.floor(cx_theta - half_x), np.ceil(cx_theta + half_x) + 1)
            yt = np.arange(np.floor(cy_theta - half_y), np.ceil(cy_theta + half_y) + 1)
            for x in xt:
                ax2.axvline(x, color="gray", lw=0.5)
            for y in yt:
                ax2.axhline(y, color="gray", lw=0.5)

        # Draw frustum rays
        theta = np.deg2rad(self.fov / 2)
        end_dist = cy_theta + half_y
        for sign in (+1, -1):
            dx = np.sin(sign * theta) * end_dist
            dy = np.cos(theta) * end_dist
            ax2.plot([0, dx], [0, dy], "--", color="blue", lw=1.5)

        # Draw ego marker
        ax2.scatter(
            0,
            0,
            marker="^",
            s=(self.ego_marker_size * 200) ** 2,
            color="black",
            zorder=5,
        )
        ax2.text(
            0,
            -self.ego_marker_size * 3,
            "Camera",
            color="black",
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
        )

        # Draw 3D box footprints on BEV
        for i, box in enumerate(boxes_3d):
            cx, cy, cz = box.center
            sx, sy, sz = box.size
            roll, pitch, yaw = box.rotation

            phi = np.deg2rad(yaw)
            Rz = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            rect = (
                np.array(
                    [
                        [-sx / 2, -sy / 2],
                        [sx / 2, -sy / 2],
                        [sx / 2, sy / 2],
                        [-sx / 2, sy / 2],
                    ]
                )
                @ Rz.T
            )
            rect += np.array([cx, cy])

            # Use consistent colors with the segmentation
            color = (
                distinct_colors[i % len(distinct_colors)]
                if distinct_colors
                else [1, 0, 1]
            )
            color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"

            ax2.add_patch(
                plt.Polygon(
                    rect,
                    closed=True,
                    fill=True,
                    alpha=0.3,
                    facecolor=color_hex,
                    edgecolor=color_hex,
                    lw=2,
                )
            )
            ax2.text(
                cx,
                cy,
                box.label,
                color="white",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(facecolor=color_hex, alpha=0.7, pad=2),
            )

        # Convert matplotlib figure to PIL Image
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        return img

    def visualize_points(
        self, pil_image: Image.Image, points: List[Point2D]
    ) -> Image.Image:
        """
        Visualize points on an image.

        Args:
            pil_image: Input PIL Image
            points: List of Point2D objects

        Returns:
            PIL Image with point visualization
        """
        result_img = pil_image.copy()
        img_width, img_height = pil_image.size
        draw = ImageDraw.Draw(result_img)

        # Define colors for different points
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "brown",
            "gray",
            "beige",
            "turquoise",
            "cyan",
            "magenta",
        ]

        # Draw each point
        for i, point in enumerate(points):
            color = colors[i % len(colors)]
            y_norm, x_norm = point.coordinates

            # Convert to pixel coordinates
            x = int(x_norm / 1000.0 * img_width)
            y = int(y_norm / 1000.0 * img_height)

            # Draw point as circle
            r = self.point_radius
            draw.ellipse(
                [(x - r, y - r), (x + r, y + r)], fill=color, outline="white", width=2
            )

            # Add label
            draw.text(
                (x + r + 5, y - r),
                point.label,
                fill="white",
                stroke_fill="black",
                stroke_width=1,
            )

            # Show coordinates for debugging
            if self.show_debug_info:
                coord_text = f"({x_norm},{y_norm})"
                draw.text(
                    (x + r, y + r + 5),
                    coord_text,
                    fill="white",
                    stroke_fill="black",
                    stroke_width=1,
                )

        return result_img

    def visualize_multiview_points(
        self,
        reference_image: Image.Image,
        reference_points: List[Point2D],
        new_image: Image.Image,
        tracked_points: List[MultiviewPoint],
    ) -> Image.Image:
        """
        Visualize point tracking across multiple views.

        Args:
            reference_image: First view image
            reference_points: Points in the first view
            new_image: Second view image
            tracked_points: Points tracked in the second view

        Returns:
            PIL Image with multiview point visualization
        """
        # Create figure for two-panel visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # First image with reference points
        ref_img = self.visualize_points(reference_image, reference_points)
        ax1.imshow(np.array(ref_img))
        ax1.set_title("Reference View")
        ax1.axis("off")

        # Second image with tracked points
        new_img = new_image.copy()
        img_width, img_height = new_img.size
        draw = ImageDraw.Draw(new_img)

        # Define colors for different points
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "brown",
            "gray",
            "beige",
            "turquoise",
            "cyan",
            "magenta",
        ]

        # Create point mapping for easy lookup
        point_map = {p.label: p for p in reference_points}

        # Draw tracked points
        for i, point in enumerate(tracked_points):
            color = colors[i % len(colors)]

            if point.in_frame and point.point is not None:
                # Point is visible in new view
                y_norm, x_norm = point.point

                # Convert to pixel coordinates
                x = int(x_norm / 1000.0 * img_width)
                y = int(y_norm / 1000.0 * img_height)

                # Draw point as circle
                r = self.point_radius
                draw.ellipse(
                    [(x - r, y - r), (x + r, y + r)],
                    fill=color,
                    outline="white",
                    width=2,
                )

                # Add label
                draw.text(
                    (x + r + 5, y - r),
                    point.label,
                    fill="white",
                    stroke_fill="black",
                    stroke_width=1,
                )
            else:
                # Point is not visible in new view
                # Mark this somewhere on the image
                draw.text(
                    (10, 10 + i * 20),
                    f"Point {point.label} not in frame",
                    fill=color,
                    stroke_fill="black",
                    stroke_width=1,
                )

        ax2.imshow(np.array(new_img))
        ax2.set_title("New View with Tracked Points")
        ax2.axis("off")

        # Convert matplotlib figure to PIL Image
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        return img
