"""Spatial guidance package."""

# from .audio_guidance.stt_engine import AudioConfig, SpeechToTextConfig, WhisperConfig
from .data_handling.stray_scanner.data_parser import StrayScannerDataParserConfig
from .data_handling.stray_scanner.stray_dataset import StrayDataset, StrayDatasetConfig
from .data_handling.stray_scanner.stray_scanner_paths import StrayScannerPaths
from .pipeline.pipeline import PipelineConfig
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
    # "PipelineStage",
    "PipelineConfig",
    # Visualization
    "SceneVisualizerConfig",
]
