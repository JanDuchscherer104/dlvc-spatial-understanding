"""Data models for pipeline data exchange between stages."""

from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="DataModel")


class DataModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_(cls: Type[T], *args: "DataModel") -> T:
        """Create a new instance using fields from other DataModel instances.

        This method creates a new instance of the class by extracting fields with
        matching names from the provided DataModel instances. If multiple source
        models have the same field, the last one in the args list takes precedence.

        Args:
            *args: One or more DataModel instances to extract fields from

        Returns:
            A new instance of the class with fields populated from source models
        """
        if not args:
            return cls()

        # Collect field names from the target class
        target_fields = set(cls.model_fields.keys())

        # Extract matching fields from source models
        extracted_data = {}
        for source in args:
            source_data = source.model_dump()
            matching_fields = {
                field: value
                for field, value in source_data.items()
                if field in target_fields
            }
            extracted_data.update(matching_fields)

        # Create new instance with extracted data
        return cls(**extracted_data)


class PipelineIn(DataModel):
    """Input data for the input stage."""

    idx: int
    user_prompt: Optional[str] = None


class DataSetOut(DataModel):
    """Input data for detection stage."""

    rgb_image: Image
    depth_image: Image
    user_prompt: Optional[str] = None


class DetectedObject(DataModel):
    """Detected object with bounding box and further information by a VLM."""

    aabb_2d: List[int] = Field(
        ..., description="Normalized coordinates [y1, x1, y2, x2] from 0-1000"
    )
    points_2d: List[Tuple[int, int]] = Field(
        ...,
        description="List of at least 3 points on the object surface. Format [(y1, x1), (y1, x2), (y3, x3)] normalized to 0-1000. (not the AABB!)",
    )
    label: str = Field(
        ...,
        description="Concise and unique instance label with distinguishing characteristics (snake_case)",
    )
    approx_distance: float = Field(
        ...,
        description="Approximate distance to the object in meters (rounded to 1 decimal place)",
    )
    height: Literal[
        "floor-level",
        "knee-height",
        "waist-height",
        "chest-height",
        "eye-level",
        "overhead",
    ] = Field(..., description="Height relative to user")
    description: str = Field(..., description="Position relative to user")
    is_hazard: bool = Field(..., description="Whether object poses collision/trip risk")
    is_dynamic: bool = Field(
        ...,
        description="Whether object might be moving (some vehicle, escalator, rotating door) or stationary",
    )
    hazard_type: Optional[Union[Literal["trip", "collision"], str]] = Field(
        None, description="Type of hazard"
    )


class DetectionStageOutput(DataModel):
    """Complete analysis of a scene for navigation assistance."""

    objects: List[DetectedObject] = Field(
        ..., description="List of detected objects in the scene"
    )


# ===== SEGMENTATION STAGE I/O MODELS =====


class SegmentationStageInput(DataModel):
    """Input data for segmentation stage."""

    rgb_frame: Any = Field(..., description="RGB frame data")
    depth_frame: Optional[Any] = Field(None, description="Depth frame data")
    user_query: Optional[str] = None


class SegmentationMask(DataModel):
    """Segmentation mask with label and confidence.

    Note: The actual mask data is no longer stored in the Pydantic model
    and should be passed separately.
    """

    label: str = Field(..., description="Label for the segmentation mask")
    confidence: float = Field(1.0, description="Confidence score")
    mask_id: str = Field(
        ..., description="ID to associate with the binary mask stored separately"
    )


class SegmentationStageOutput(DataModel):
    """Output data from segmentation stage."""

    masks: List[SegmentationMask] = Field(
        ..., description="List of segmentation masks identified in the scene"
    )


# ===== DEPTH ESTIMATION STAGE I/O MODELS =====


class DepthEstimationInput(DataModel):
    """Input data for depth estimation stage."""

    depth_frame: Optional[Any] = Field(None, description="Depth frame data")
    rgb_frame: Optional[Any] = Field(None, description="RGB frame data")
    detected_objects: List[DetectedObject]
    segmentation_masks: Optional[List[SegmentationMask]] = None


class EnrichedObject(DetectedObject):
    """Detected object enriched with precise depth information."""

    precise_distance: float = Field(
        ..., description="Precise distance to the object in meters"
    )
    confidence: float = Field(
        1.0, description="Confidence level of the depth estimation (0-1)"
    )


class DepthEstimationOutput(DataModel):
    """Output data from depth estimation stage."""

    objects_with_depth: List[EnrichedObject] = Field(
        ..., description="List of detected objects with depth information"
    )


class VisualizationIn(DataModel):
    """Input for the visualization stage."""

    rgb_image: Image
    depth_image: Image
    detection_output: DetectionStageOutput


class VisualizationOutput(DataModel):
    """Output from the visualization stage."""

    visualization: Image
    object_count: int


# ===== REFINEMENT STAGE I/O MODELS =====


class RefinementStageInput(DataModel):
    """Input data for refinement stage."""

    rgb_frame: Any = Field(..., description="RGB frame data")
    objects_with_depth: List[EnrichedObject]
    user_query: Optional[str] = None


class RefinedObject(EnrichedObject):
    """Object with refined metadata and priority information."""

    priority_score: float = Field(
        ...,
        description="Priority score for navigation importance (higher is more important)",
    )
    navigation_advice: str = Field(
        ..., description="Specific navigation advice related to this object"
    )


class RefinementStageOutput(DataModel):
    """Final output from the refinement stage."""

    refined_objects: List[RefinedObject] = Field(
        ..., description="List of refined and prioritized objects"
    )
    scene_description: str = Field(
        ..., description="Natural language description of the scene for navigation"
    )
    path_recommendation: Optional[str] = Field(
        None, description="Recommended path through the scene if applicable"
    )
