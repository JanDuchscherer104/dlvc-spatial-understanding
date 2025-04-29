"""Spatial guidance package."""

from spatial_guidance.pipeline.pipeline_stage import PipelineStage  # Add this import"

# from .audio_guidance.stt_engine import AudioConfig, SpeechToTextConfig, WhisperConfig
from .data_handling.stray_scanner.data_parser import StrayScannerDataParserConfig
from .data_handling.stray_scanner.stray_dataset import StrayDataset, StrayDatasetConfig
from .data_handling.stray_scanner.stray_scanner_paths import StrayScannerPaths
from .pipeline.pipeline import PipelineConfig, SpatialUnderstandingPipeline

# from .pipeline.data_contracts import DetectedObject, InputData, VLMSceneDescription
from .scene_understanding.vlm_gemini_detector import GeminiVLMDetectionConfig
from .visualization.scene_visualizer import SceneVisualizerConfig

__all__ = [
    # Audio guidance
    # "AudioConfig",
    # "SpeechToTextConfig",
    # "WhisperConfig",
    # Data handling
    "StrayDataset",
    "StrayDatasetConfig",
    "StrayScannerDataParserConfig",
    "StrayScannerPaths",
    "GeminiVLMDetectionConfig",
    # Pipeline
    "PipelineStage",
    "SpatialUnderstandingPipeline",
    "PipelineConfig",
    # Visualization
    "SceneVisualizerConfig",
]
