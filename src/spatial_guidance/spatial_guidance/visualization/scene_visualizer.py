"""Scene visualization pipeline stage for displaying detection results.

This module provides a pipeline stage for visualizing detection results
alongside depth data for better spatial understanding.
"""

import io
from typing import Any, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field
from zenml.steps import BaseStep

from ..pipeline.data_contracts import DetectedObject, VisualizationIn, VisualizationOut
from ..utils import CONSOLE, BaseConfig


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
        super().__init__(**step_kwargs)
        # config = config or SceneVisualizerConfig()

        # Store visualization-specific configuration
        config = config or SceneVisualizerConfig()
        self.show_debug_info = config.show_debug_info
        self.figure_size = config.figure_size
        self.colorful_depth = config.colorful_depth
        self.point_radius = config.point_radius
        self.dpi = config.dpi

        CONSOLE.log(f"Initialized SceneVisualizer with debug={self.show_debug_info}")

    def entrypoint(self, input_data: VisualizationIn) -> VisualizationOut:
        """Process the input data to produce a visualization.

        Args:
            input_data: Contains RGB image, depth image, and detection output

        Returns:
            VisualizationOutput containing the visualization as a PIL Image
        """
        # Convert PIL images to numpy arrays for processing
        rgb_frame = np.array(input_data.rgb_image)
        depth_frame = np.array(input_data.depth_image)

        # Create visualization
        vis_image = self.visualize_scene(
            rgb_frame, depth_frame, input_data.detection_output.objects
        )

        return VisualizationOut(
            visualization=vis_image,
            object_count=len(input_data.detection_output.objects),
        )

    def visualize_scene(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        objects: List[DetectedObject],
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
        obj: DetectedObject,
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
        y1, x1, y2, x2 = obj.aabb_2d

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
