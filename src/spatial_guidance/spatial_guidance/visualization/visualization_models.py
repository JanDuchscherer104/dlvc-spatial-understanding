"""Input and output models for different visualization types."""

from enum import Enum
from typing import Any, List, Optional, Union

from PIL.Image import Image
from pydantic import Field

from ..pipeline.data_contracts import (
    AABBDetection,
    DataModel,
    VisualizationIn,
    VisualizationOut,
)
from ..pipeline.data_contracts_3d import (
    Box3D,
    CombinedDetectionSegmentationOut,
    MultiviewPoint,
    Point2D,
    SegmentationMaskData,
)


class DetectionVisualizationIn(VisualizationIn):
    """Standard input for detection visualization."""

    pass


class BoxesVisualizationIn(DataModel):
    """Input for 3D bounding boxes visualization."""

    rgb_image: Image = Field(..., description="RGB image to visualize")
    boxes_3d: List[Box3D] = Field(..., description="List of 3D bounding boxes")


class SegmentationVisualizationIn(DataModel):
    """Input for segmentation visualization."""

    rgb_image: Image = Field(..., description="RGB image to visualize")
    segmentation: List[SegmentationMaskData] = Field(
        ..., description="List of segmentation masks"
    )


class CombinedVisualizationIn(DataModel):
    """Input for combined segmentation and 3D boxes visualization."""

    rgb_image: Image = Field(..., description="RGB image to visualize")
    segmentation: List[SegmentationMaskData] = Field(
        ..., description="List of segmentation masks"
    )
    boxes_3d: List[Box3D] = Field(..., description="List of 3D bounding boxes")


class PointVisualizationIn(DataModel):
    """Input for point visualization."""

    rgb_image: Image = Field(..., description="RGB image to visualize")
    points: List[Point2D] = Field(..., description="List of 2D points")


class MultiviewPointVisualizationIn(DataModel):
    """Input for multiview point visualization."""

    reference_image: Image = Field(..., description="Reference view image")
    reference_points: List[Point2D] = Field(
        ..., description="Points in the reference view"
    )
    new_image: Image = Field(..., description="New view image")
    tracked_points: List[MultiviewPoint] = Field(
        ..., description="Points tracked in the new view"
    )
