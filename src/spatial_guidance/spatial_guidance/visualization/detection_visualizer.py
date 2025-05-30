from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image as PILImage
from PIL import ImageColor, ImageDraw, ImageFont

from ..data_contracts.aabb_segmentation import AABBDetection, AABBDetections
from ..data_contracts.obb_detection import OBBDetections


class DetectionVisualizer:
    """Handles the visualization of AABB detections on RGB and depth images."""

    @staticmethod
    def _find_optimal_label_position(
        detection: AABBDetection,
        img_width: int,
        img_height: int,
        font_size: int = 24,
    ) -> Tuple[int, int]:
        y0_abs, x0_abs, y1_abs, x1_abs = (
            detection.box_2d
        )  # These are absolute after processing

        # Use mask centroid if mask is a PIL Image and valid
        if isinstance(detection.mask, PILImage.Image):
            mask_array = np.array(detection.mask.convert("L"))
            mask_indices = np.where(mask_array > 127)

            if len(mask_indices[0]) > 0:
                y_center = int(np.mean(mask_indices[0]))
                x_center = int(np.mean(mask_indices[1]))
            else:
                y_center = (y0_abs + y1_abs) // 2
                x_center = (x0_abs + x1_abs) // 2
        else:
            y_center = (y0_abs + y1_abs) // 2
            x_center = (x0_abs + x1_abs) // 2

        label_y_position = y_center - font_size - 5

        if label_y_position < 5:
            label_y_position = y0_abs + 5
            if label_y_position + font_size > y1_abs:
                label_y_position = y0_abs + 2

        x_pos = max(5, min(img_width - 50, x0_abs + 5))
        y_pos = max(5, min(img_height - font_size - 5, label_y_position))

        return x_pos, y_pos

    @staticmethod
    def _overlay_mask_on_img(
        img: PILImage.Image, mask: PILImage.Image, color: str, alpha: float = 0.7
    ) -> PILImage.Image:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        try:
            color_rgb = ImageColor.getrgb(color)
        except ValueError as e:
            raise ValueError(f"Invalid color '{color}': {e}")

        img_rgba = img.convert("RGBA")
        mask_l_mode = mask.convert("L")
        mask_array = np.array(mask_l_mode)

        width, height = img_rgba.size

        alpha_int = int(alpha * 255)
        overlay_color_rgba = color_rgb + (alpha_int,)

        colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)
        mask_binary = mask_array > 127
        colored_mask_layer_np[mask_binary] = overlay_color_rgba

        mask_layer_pil = PILImage.fromarray(colored_mask_layer_np, "RGBA")
        result_img = PILImage.alpha_composite(img_rgba, mask_layer_pil)
        return result_img.convert("RGB")

    @staticmethod
    def visualize_rgb_detections(
        img: PILImage.Image,
        detections: AABBDetections,
        alpha: float = 0.4,
        show_boxes: bool = True,
        line_width: int = 2,
        font_size: int = 16,
        show_3d_info: bool = False,
    ) -> PILImage.Image:
        result_img = img.copy()
        width, height = img.size
        colors = [
            "#FF3838",
            "#FF9D97",
            "#FF701F",
            "#FFB21D",
            "#CFD231",
            "#48F90A",
            "#92CC17",
            "#3DDB86",
            "#1A9334",
            "#00D4BB",
            "#2C99A8",
            "#00C2FF",
            "#344593",
            "#6473FF",
            "#0018EC",
            "#8400FF",
            "#AA00FF",
            "#FF00AA",
            "#FF0080",
        ]

        for i, obj in enumerate(detections.objects):
            color = colors[i % len(colors)]
            if isinstance(obj.mask, PILImage.Image):
                result_img = DetectionVisualizer._overlay_mask_on_img(
                    result_img, obj.mask, color, alpha
                )

        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        if show_boxes:
            for i, obj in enumerate(detections.objects):
                color = colors[i % len(colors)]
                y0_abs, x0_abs, y1_abs, x1_abs = obj.box_2d  # Absolute coords

                actual_line_width = max(1, min(line_width, int(width / 300)))
                draw.rectangle(
                    ((x0_abs, y0_abs), (x1_abs, y1_abs)),
                    outline=color,
                    width=actual_line_width,
                )

        for i, obj in enumerate(detections.objects):
            if not obj.label:
                continue
            color = colors[i % len(colors)]
            x_pos, y_pos = DetectionVisualizer._find_optimal_label_position(
                obj, width, height, font_size=font_size
            )

            label_text = obj.label
            depth_parts = []
            if obj.min_depth is not None:
                depth_parts.append(f"{obj.min_depth:.1f}")
            if obj.med_depth is not None:
                depth_parts.append(f"{obj.med_depth:.1f}")
            if obj.max_depth is not None:
                depth_parts.append(f"{obj.max_depth:.1f}")
            if depth_parts:
                label_text += f" D:({','.join(depth_parts)})m"

            # Add 3D information if requested and available
            if show_3d_info:
                info_3d = DetectionVisualizer._format_3d_info(obj)
                if info_3d:
                    label_text += f"\n{info_3d}"

            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_bbox_on_image = (
                    x_pos,
                    y_pos,
                    x_pos + text_width,
                    y_pos + text_height,
                )
            except AttributeError:
                text_width, text_height = 10 * len(label_text), font_size
                text_bbox_on_image = (
                    x_pos,
                    y_pos,
                    x_pos + text_width,
                    y_pos + text_height,
                )

            bg_color_rgb = ImageColor.getrgb("black")
            bg_color_rgba = bg_color_rgb + (int(0.6 * 255),)

            temp_draw_img = PILImage.new("RGBA", result_img.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_draw_img)

            padding = 2
            rect_coords = [
                (text_bbox_on_image[0] - padding, text_bbox_on_image[1] - padding),
                (text_bbox_on_image[2] + padding, text_bbox_on_image[3] + padding),
            ]
            temp_draw.rounded_rectangle(rect_coords, radius=3, fill=bg_color_rgba)
            result_img.paste(temp_draw_img, (0, 0), temp_draw_img)

            draw = ImageDraw.Draw(result_img)
            draw.text((x_pos, y_pos), label_text, fill="white", font=font)

        return result_img

    @staticmethod
    def _create_depth_visualization(
        depth_frame_pil: PILImage.Image,
        target_width: int,
        target_height: int,
        colorful_depth: bool = True,
    ) -> PILImage.Image:
        depth_frame_np = np.array(depth_frame_pil.convert("F"))

        if depth_frame_np is not None and np.any(depth_frame_np > 0):
            depth_vis = depth_frame_np.copy()

            valid_depth = depth_vis[depth_vis > 0]
            if valid_depth.size == 0:
                return PILImage.new("RGB", (target_width, target_height), color="black")

            min_depth = np.min(valid_depth)
            max_depth = np.max(valid_depth)

            depth_range = max_depth - min_depth
            if depth_range < 1e-5:
                depth_vis_norm = np.where(depth_vis > 0, 128, 0).astype(np.uint8)
            else:
                depth_vis_norm = np.where(
                    depth_vis > 0,
                    255 - ((depth_vis - min_depth) / depth_range * 255),
                    0,
                ).astype(np.uint8)

            if colorful_depth:
                depth_rgb = np.zeros(
                    (depth_vis_norm.shape[0], depth_vis_norm.shape[1], 3),
                    dtype=np.uint8,
                )
                depth_rgb[:, :, 0] = depth_vis_norm
                depth_rgb[:, :, 2] = 255 - depth_vis_norm
                depth_img_pil = PILImage.fromarray(depth_rgb, mode="RGB")
            else:
                depth_img_pil = PILImage.fromarray(depth_vis_norm, mode="L").convert(
                    "RGB"
                )

            if depth_img_pil.size != (target_width, target_height):
                depth_img_pil = depth_img_pil.resize(
                    (target_width, target_height), PILImage.Resampling.BILINEAR
                )
        else:
            depth_img_pil = PILImage.new(
                "RGB", (target_width, target_height), color="black"
            )

        return depth_img_pil

    @staticmethod
    def visualize_depth_detections(
        depth_image_pil: PILImage.Image,
        detections: AABBDetections,
        img_width: int,
        img_height: int,
        line_width: int = 2,
        font_size: int = 16,
        show_3d_info: bool = False,
    ) -> PILImage.Image:
        depth_vis_img = DetectionVisualizer._create_depth_visualization(
            depth_image_pil, img_width, img_height, colorful_depth=True
        )

        draw = ImageDraw.Draw(depth_vis_img)
        colors = [
            "#FF3838",
            "#FF9D97",
            "#FF701F",
            "#FFB21D",
            "#CFD231",
            "#48F90A",
            "#92CC17",
            "#3DDB86",
            "#1A9334",
            "#00D4BB",
            "#2C99A8",
            "#00C2FF",
            "#344593",
            "#6473FF",
            "#0018EC",
            "#8400FF",
            "#AA00FF",
            "#FF00AA",
            "#FF0080",
        ]

        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        for i, obj in enumerate(detections.objects):
            color = colors[i % len(colors)]
            y0_abs, x0_abs, y1_abs, x1_abs = obj.box_2d  # Absolute coords

            actual_line_width = max(1, min(line_width, int(img_width / 300)))
            draw.rectangle(
                ((x0_abs, y0_abs), (x1_abs, y1_abs)),
                outline=color,
                width=actual_line_width,
            )

            if obj.label:
                x_pos, y_pos = DetectionVisualizer._find_optimal_label_position(
                    obj, img_width, img_height, font_size=font_size
                )

                label_text = obj.label
                depth_parts = []
                if obj.min_depth is not None:
                    depth_parts.append(f"{obj.min_depth:.1f}")
                if obj.med_depth is not None:
                    depth_parts.append(f"{obj.med_depth:.1f}")
                if obj.max_depth is not None:
                    depth_parts.append(f"{obj.max_depth:.1f}")
                if depth_parts:
                    label_text += f" D:({','.join(depth_parts)})m"

                # Add 3D information if requested and available
                if show_3d_info:
                    info_3d = DetectionVisualizer._format_3d_info(obj)
                    if info_3d:
                        label_text += f"\n{info_3d}"

                try:
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_bbox_on_image = (
                        x_pos,
                        y_pos,
                        x_pos + text_width,
                        y_pos + text_height,
                    )
                except AttributeError:
                    text_width, text_height = 10 * len(label_text), font_size
                    text_bbox_on_image = (
                        x_pos,
                        y_pos,
                        x_pos + text_width,
                        y_pos + text_height,
                    )

                bg_color_rgb = ImageColor.getrgb("black")
                bg_color_rgba = bg_color_rgb + (int(0.6 * 255),)

                temp_draw_img = PILImage.new("RGBA", depth_vis_img.size, (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_draw_img)
                padding = 2
                rect_coords = [
                    (text_bbox_on_image[0] - padding, text_bbox_on_image[1] - padding),
                    (text_bbox_on_image[2] + padding, text_bbox_on_image[3] + padding),
                ]
                temp_draw.rounded_rectangle(rect_coords, radius=3, fill=bg_color_rgba)
                depth_vis_img.paste(temp_draw_img, (0, 0), temp_draw_img)

                draw = ImageDraw.Draw(depth_vis_img)
                draw.text((x_pos, y_pos), label_text, fill="white", font=font)

        return depth_vis_img

    @staticmethod
    def _calculate_optimal_label_positions(
        detections_list, img_width: int, img_height: int
    ):
        """
        Calculate optimal label positions to avoid overlaps and clipping.

        Args:
            detections_list: List of detection objects with bev_bbox attribute
            img_width, img_height: Image dimensions

        Returns:
            List of (x, y) label positions
        """
        label_positions = []
        used_positions = []

        for i, det in enumerate(detections_list):
            if det.bev_bbox is None:
                # Fallback to center if no projection available
                label_positions.append((img_width / 2, img_height / 2))
                continue

            pts2d = det.bev_bbox

            # Filter out NaN points
            valid_pts2d = pts2d[~np.isnan(pts2d).any(axis=1)]

            if len(valid_pts2d) > 0:
                # Calculate centroid of valid 2D points (top face center)
                cx2, cy2 = valid_pts2d.mean(axis=0)

                # Ensure label stays within image bounds with margin
                margin = 50
                cx2 = np.clip(cx2, margin, img_width - margin)
                cy2 = np.clip(cy2, margin, img_height - margin)

                # Check for overlap with previous labels
                min_distance = 80  # Minimum distance between labels

                for prev_x, prev_y in used_positions:
                    distance = np.hypot(cx2 - prev_x, cy2 - prev_y)
                    if distance < min_distance:
                        # Move label to avoid overlap
                        angle = np.arctan2(cy2 - prev_y, cx2 - prev_x)
                        cx2 = prev_x + min_distance * np.cos(angle)
                        cy2 = prev_y + min_distance * np.sin(angle)

                        # Re-clamp to image bounds
                        cx2 = np.clip(cx2, margin, img_width - margin)
                        cy2 = np.clip(cy2, margin, img_height - margin)

                label_positions.append((cx2, cy2))
                used_positions.append((cx2, cy2))
            else:
                # Fallback for invalid projections
                fallback_x = img_width / 2 + i * 100  # Offset by index
                fallback_y = img_height / 2
                label_positions.append((fallback_x, fallback_y))
                used_positions.append((fallback_x, fallback_y))

        return label_positions

    @staticmethod
    def visualize_3d_boxes(
        pil_image: PILImage.Image,
        obb_detections: OBBDetections,  # input is OBBDetections
        fov: float = 60.0,
        zoom: float = 1.5,
        figsize: tuple = (12, 5),
        ego_marker_size: float = 0.1,
        show_grid: bool = True,
        use_distinct_bbox_colors: bool = False,
        verbose_labels: bool = True,
        add_labels: bool = True,
    ):
        """
        Draw a static two-panel visualization of 3D bounding boxes,
        with a bo-arrow ego marker and BEV frustum lines.

        Parameters
        ----------
        pil_image : PIL.Image
            Background image for the perspective view.
        obb_detections : OBBDetections
            An OBBDetections object containing the list of detections.
        fov : float
            Horizontal field-of-view in degrees.
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
        verbose_labels : bool
            If True, show detailed labels with measurements (height, distance, area, volume).
            If False, show only the object label name.
        add_labels : bool
            If True, add text labels to the visualization. If False, show only the boxes.
        """
        # 1) parse
        img = np.array(pil_image)
        H, W = img.shape[:2]

        # Generate distinct colors for bounding boxes if requested
        if use_distinct_bbox_colors and len(obb_detections) > 0:
            # Create visually distinct colors using HSV color space
            import matplotlib.colors as mcolors

            def generate_distinct_colors(n):
                # Generate evenly spaced hues
                HSVs = [(i / n, 0.9, 0.9) for i in range(n)]
                # Convert to RGB
                RGBs = [mcolors.hsv_to_rgb(hsv) for hsv in HSVs]
                return RGBs

            box_colors = generate_distinct_colors(len(obb_detections))
        else:
            # Default colors
            box_colors = ["cyan"] * len(obb_detections)  # For perspective view

        # Calculate correct BEV bounds using the working notebook formula
        xs, ys = [], []
        for det in obb_detections.objects:
            if det.box_3d:
                cx, cy, _, sx, sy, _, _, _, _ = det.box_3d
                # Add box extents to bounds calculation
                xs.extend([cx - sx / 2, cx + sx / 2])
                ys.extend([cy - sy / 2, cy + sy / 2])

        # Always include ego position (0, 0)
        xs.append(0)
        ys.append(0)

        if not xs or not ys:
            # Fallback bounds
            xs, ys = [-1, 1], [-1, 1]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Calculate spans with minimum threshold
        dx = max(xmax - xmin, 1e-3)
        dy = max(ymax - ymin, 1e-3)

        # Center on data centroid (CRITICAL FIX: match notebook formula)
        cx_theta = (xmin + xmax) / 2
        cy_theta = (ymin + ymax) / 2

        # Calculate half-spans with zoom
        half_x = (dx / 2) * zoom
        half_y = (dy / 2) * zoom

        # 4) plot
        fig = plt.figure(figsize=figsize)

        # Calculate optimal label positions to avoid overlaps
        label_positions = DetectionVisualizer._calculate_optimal_label_positions(
            obb_detections.objects, W, H
        )

        # 4a) perspective
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"Corrected Perspective (FOV={fov:.0f}°)")

        for i, det in enumerate(
            obb_detections.objects
        ):  # Iterate with index for colors
            label = det.label
            pts2d = det.bev_bbox  # Use processed bev_bbox
            depths = det.depth  # Use processed depth
            height = det.height
            distance = det.center_distance
            area = det.bev_area

            if (
                pts2d is None
                or depths is None
                or np.all(np.isnan(pts2d))
                or np.all(np.isnan(depths))
            ):
                continue

            # Filter out NaN depths for min/max calculation to avoid runtime warnings
            valid_depths = depths[~np.isnan(depths)]
            if len(valid_depths) == 0:
                continue  # Skip if all depths are NaN
            dmin, dmax = valid_depths.min(), valid_depths.max()

            # Use the generated color if distinct colors are requested
            color = box_colors[i]  # Use index i for color selection

            # Draw lines for the 3D box
            # Ensure drawing only if points are not NaN
            for j in range(4):
                # Top face lines
                if not (np.isnan(pts2d[j]).any() or np.isnan(pts2d[(j + 1) % 4]).any()):
                    x0, y0 = pts2d[j]
                    x1, y1 = pts2d[(j + 1) % 4]
                    # Average depth for line width, handle potential NaNs in depths
                    depth_val_0 = depths[j] if not np.isnan(depths[j]) else dmin
                    depth_val_1 = (
                        depths[(j + 1) % 4]
                        if not np.isnan(depths[(j + 1) % 4])
                        else dmin
                    )
                    avg_depth_top = (depth_val_0 + depth_val_1) / 2
                    w = 4 - ((avg_depth_top - dmin) / (dmax - dmin + 1e-5)) * 3
                    ax1.add_line(
                        Line2D([x0, x1], [y0, y1], lw=max(0.5, w), color=color)
                    )

                # Bottom face lines
                if not (
                    np.isnan(pts2d[j + 4]).any()
                    or np.isnan(pts2d[(j + 1) % 4 + 4]).any()
                ):
                    bx0, by0 = pts2d[j + 4]
                    bx1, by1 = pts2d[(j + 1) % 4 + 4]
                    depth_val_b0 = (
                        depths[j + 4] if not np.isnan(depths[j + 4]) else dmin
                    )
                    depth_val_b1 = (
                        depths[(j + 1) % 4 + 4]
                        if not np.isnan(depths[(j + 1) % 4 + 4])
                        else dmin
                    )
                    avg_depth_bottom = (depth_val_b0 + depth_val_b1) / 2
                    w2 = 4 - ((avg_depth_bottom - dmin) / (dmax - dmin + 1e-5)) * 3
                    ax1.add_line(
                        Line2D([bx0, bx1], [by0, by1], lw=max(0.5, w2), color=color)
                    )

                # Vertical lines connecting top and bottom faces
                if not (np.isnan(pts2d[j]).any() or np.isnan(pts2d[j + 4]).any()):
                    x0, y0 = pts2d[j]
                    bx0, by0 = pts2d[j + 4]
                    # Use depth of top point for vertical line width, or dmin if NaN
                    depth_val_vert = depths[j] if not np.isnan(depths[j]) else dmin
                    w_vert = 4 - ((depth_val_vert - dmin) / (dmax - dmin + 1e-5)) * 3
                    ax1.add_line(
                        Line2D([x0, bx0], [y0, by0], lw=max(0.5, w_vert), color=color)
                    )

            # Add labels only if add_labels is True
            if add_labels:
                label_x, label_y = label_positions[i]
                bbox_color = color  # Use the same color for the bbox background

                if verbose_labels:
                    # Detailed labels with measurements
                    label_text = (
                        f"{label}\n"
                        f"H:{height:.2f}m D:{distance:.2f}m\n"
                        f"A:{area:.2f}m² V:{det.volume:.2f}m³"
                    )
                else:
                    # Simple label with just the object name
                    label_text = label

                ax1.text(
                    label_x,
                    label_y,
                    label_text,
                    color="white",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor=bbox_color, alpha=0.7, pad=3),
                    fontsize=8,
                    weight="bold",
                )

        # 4b) BEV
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("equal")
        ax2.set_title(f"Corrected BEV (zoom={zoom:.1f}×)")
        # Apply corrected bounds (centered on data centroid)
        ax2.set_xlim(cx_theta - half_x, cx_theta + half_x)
        ax2.set_ylim(cy_theta - half_y, cy_theta + half_y)

        # Add grid only if show_grid is True
        if show_grid:
            # grid
            xt = np.arange(np.floor(cx_theta - half_x), np.ceil(cx_theta + half_x) + 1)
            yt = np.arange(np.floor(cy_theta - half_y), np.ceil(cy_theta + half_y) + 1)
            for x_val in xt:
                ax2.axvline(x_val, color="gray", lw=0.5)
            for y_val in yt:
                ax2.axhline(y_val, color="gray", lw=0.5)

        # 4c) frustum rays
        theta = np.deg2rad(fov / 2)
        frustum_line_length = max(half_x, half_y) * 2
        for sign in (+1, -1):
            dx = np.sin(sign * theta) * frustum_line_length
            dy = np.cos(theta) * frustum_line_length
            ax2.plot([0, dx], [0, dy], "--", color="blue", lw=1.5)

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

        # 4e) draw box footprints
        for i, det in enumerate(obb_detections.objects):
            if not det.box_3d:
                continue
            cx, cy, _, sx, sy, _, _, _, yaw = det.box_3d  # Unpack relevant parts
            label = det.label
            height = det.height
            distance = det.center_distance
            area = det.bev_area

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

            bev_color = box_colors[i]  # Use the same color scheme as perspective

            ax2.add_patch(
                plt.Polygon(rect, closed=True, fill=False, edgecolor=bev_color, lw=2)
            )
            # label with new fields
            label_text = (
                f"{label}\n"
                f"H:{height:.2f}m D:{distance:.2f}m\n"
                f"A:{area:.2f}m² V:{det.volume:.2f}m³"
            )
            ax2.text(
                cx,
                cy,
                label_text,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor=bev_color, alpha=0.6, pad=2),
                fontsize=8,
            )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _format_3d_info(detection: AABBDetection, include_rotation: bool = True) -> str:
        """
        Format 3D information for display in labels.

        Args:
            detection: AABBDetection object with 3D fields
            include_rotation: Whether to include rotation information

        Returns:
            Formatted string with 3D information
        """
        info_parts = []

        if detection.center_3d_bbox is not None:
            x, y, z = detection.center_3d_bbox
            info_parts.append(f"3D-Bbox: ({x:.1f},{y:.1f},{z:.1f})m")

        if detection.center_3d_mask is not None:
            x, y, z = detection.center_3d_mask
            info_parts.append(f"3D-Mask: ({x:.1f},{y:.1f},{z:.1f})m")

        if include_rotation and detection.rotation_deg is not None:
            rotation_info = f"Rot: {detection.rotation_deg:.1f}°"
            if detection.rotation_clock is not None and detection.rotation_clock:
                clock_positions = ", ".join(
                    [f"{hour}h" for hour in detection.rotation_clock]
                )
                rotation_info += f" ({clock_positions})"
            info_parts.append(rotation_info)

        return "\n".join(info_parts) if info_parts else ""

    @staticmethod
    def print_3d_summary(detections: AABBDetections) -> None:
        """
        Print a summary of all 3D information computed for the detections.

        Args:
            detections: AABBDetections object with computed 3D fields
        """
        print("\n=== 3D Detection Summary ===")
        if not detections.objects:
            print("No detections found.")
            return

        for i, obj in enumerate(detections.objects):
            print(f"\n[{i+1}] {obj.label}")
            print(
                f"  Depth stats: min={obj.min_depth:.2f}m, med={obj.med_depth:.2f}m, max={obj.max_depth:.2f}m"
                if obj.min_depth is not None
                else "  Depth stats: Not available"
            )

            if obj.center_3d_bbox is not None:
                x, y, z = obj.center_3d_bbox
                print(f"  3D Center (BBox): ({x:.2f}, {y:.2f}, {z:.2f})m")
            else:
                print("  3D Center (BBox): Not computed")

            if obj.center_3d_mask is not None:
                x, y, z = obj.center_3d_mask
                print(f"  3D Center (Mask): ({x:.2f}, {y:.2f}, {z:.2f})m")
            else:
                print("  3D Center (Mask): Not computed")

            if obj.rotation_deg is not None:
                print(f"  Rotation: {obj.rotation_deg:.1f}°")
                if obj.rotation_clock is not None and obj.rotation_clock:
                    clock_positions = ", ".join(
                        [f"{hour}h" for hour in obj.rotation_clock]
                    )
                    print(f"  Clock positions: {clock_positions}")
            else:
                print("  Rotation: Not computed")
        print("\n===========================")
