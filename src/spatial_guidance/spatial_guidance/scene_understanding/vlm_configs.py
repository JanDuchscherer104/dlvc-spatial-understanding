"""Base configuration classes for VLM detectors."""

from enum import Enum
from typing import List, Literal, Tuple, Type

from google.genai import types
from pydantic import Field

from ..utils import BaseConfig
from .gemini_aabb_detection import GeminiAABBDetSeg


class VLMConfig(BaseConfig):
    """Base configuration for all VLM-based detectors."""

    target: Type[GeminiAABBDetSeg] = Field(default_factory=lambda: GeminiAABBDetSeg)

    # Common model configuration for all detectors
    model_name: Literal[
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro-exp-03-25",
    ] = Field("gemini-2.0-flash", description="Name of the Gemini model to use")

    temperature: float = Field(
        0.5,
        description="Controls randomness in the output. Lower values make output more deterministic.",
    )
    top_p: float = Field(
        0.95,
        description="Nucleus sampling: Consider the smallest set of tokens whose probability sum exceeds top_p",
    )
    top_k: int = Field(
        40, description="Only sample from the top k most likely tokens at each step"
    )
    max_objects: int = Field(
        25, description="Maximum number of objects to detect in a scene"
    )

    # Safety settings
    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            (
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "BLOCK_ONLY_HIGH",
            ),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            (
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "BLOCK_ONLY_HIGH",
            ),
        ],
        description="Safety settings for the Gemini model",
    )

    def get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]
