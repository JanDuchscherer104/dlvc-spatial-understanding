from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import numpy as np
from pydantic import Field

from spatial_guidance import StrayDatasetConfig
from utils import CONSOLE, BaseConfig

CONSOLE.log("[red] This is red text[/red]")

@dataclass
class Data:
    rgb_frame: np.ndarray
    depth_frame: np.ndarray
    semantic_map: Optional[np.ndarray] = None
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None 
    labels: Optional[List[str]] = None
    prompt: Optional[str] = None
    gemini_response: Optional[str] = None

class PreprocessingStage:
    """
    Should be model specific to preprocess the data and convert it into a more specific form.
    - depending on the model we will need to rescale to specific sizes
    - some models might need normalized image data
    - when we get results (eg. bboxes or masks) we need potentially nned to convert them back.

    """


class PipelineStage(ABC):
    @abstractmethod
    def process(self, data: Data) -> Data:
        """Process input data and return processed output"""
        return data

    def set_next(self, stage: "PipelineStage"):
        """Set next stage in pipeline"""
        self.next_stage = stage
        return stage

    def execute(self, data: Data) -> Data:
        """Execute this stage and forward to next"""
        result = self.process(data)
        if hasattr(self, "next_stage"):
            return self.next_stage.execute(result)
        return result


class PreprocessingForGemini(PreprocessingStage): ...


class ZeroShotModelConfig(BaseConfig["ZeroShotModel"]):
    target: Type["ZeroShotModel"] = Field(default_factory=lambda: ZeroShotModel)


class ZeroShotModel(PipelineStage): ...


class ZeroShotDetectionModel: ...


class DepthAPI: ...


# get depth values given detected bboxes (and potentially masks)


class MergePredictionsAndDepth:
    ...
    # Merge into structured data (pydantic),
    # should be "pastable" into prompt template


class FilterDetections:
    ...
    # filter all detections based on criteria (eg. hazard class, is_dynamic - predicted by VLM, and distance)


class OrderDetections:
    ...
    # same criteria


class FinalVLMInference:
    ...
    # pass structured data with system prompt to VLM and let it
    # provide final NLP output!


class DetectionStage:
    def __init__(self, models=None):
        self.models = models or []

    def add_model(self, model):
        """Add a detection model to this stage"""
        self.models.append(model)

    def process(self, data):
        """Run all detection models and merge results"""
        results = {}
        for model in self.models:
            # Determine what data the model needs (RGB vs RGB-D)
            model_input = self._prepare_model_input(data, model)
            model_results = model.detect(model_input)
            results[model.__class__.__name__] = model_results

        # Add results to data dict
        data["detection_results"] = results
        return data

    def _prepare_model_input(self, data, model):
        """Prepare the right kind of input for each model"""
        # Logic to determine if model needs RGB or RGB-D
        return data
