from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

import numpy as np
from google import genai
from google.genai import types
from langchain_core.output_parsers import PydanticOutputParser
from PIL import Image
from pydantic import Field
from zenml import step

from utils import CONSOLE, BaseConfig, PathConfig

from ..pipeline.data_contracts import DataSetOut, DetectionStageOutput
from ..pipeline.pipeline_stage import PipelineStage, PipelineStageConfig


class GeminiVLMDetectionConfig(PipelineStageConfig["GeminiVLMDetection"]):
    """Configuration for Gemini VLM detection model."""

    target: Type["GeminiVLMDetection"] = Field(
        default_factory=lambda: GeminiVLMDetection,
        description="Target class to instantiate",
    )

    # Model-specific configuration
    model_name: Literal["gemini-2.0-flash", "gemini-2.0-flash-lite"] = Field(
        "gemini-2.0-flash", description="Name of the Gemini model to use"
    )
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

    # Base system prompt template - will be formatted with specific instructions
    base_system_prompt: str = Field(
        """
        You are an assistive AI for 3D scene understanding, helping visually impaired users navigate safely through their environment.

        DETECTION PRIORITIES (in order of importance):
        1. Immediate trip hazards (floor-level obstacles, cords, steps, uneven surfaces)
        2. Moving objects (vehicles, objects that might be in motion)
        3. Head-height dangers (hanging objects, branches, signs below 7ft)
        4. Navigation landmarks (doors, hallways, stairs, railings, elevator buttons)
        5. Structural elements (walls, furniture, plants, fixtures)

        FURTHER CONTEXT:
        - Prioritize safety-critical objects that pose immediate risks
        - Focus on practical, actionable spatial information
        - Use concise directional language
        - The system is designed for real-time blind navigation assistance
        - Limit detection to {max_objects} most critical objects

        {format_instructions}
        """,
    )

    # Default human prompt - can be overridden at inference time
    default_prompt: str = Field(
        "Analyze this scene for a blind person. Identify all obstacles, hazards, and important landmarks. "
        "Include their positions relative to the user and provide clear navigation guidance.",
        description="Default prompt to use if none is provided",
    )

    # Output type for structured parsing
    output_type: Type[DetectionStageOutput] = Field(
        default=DetectionStageOutput,
        description="Pydantic model to use for structured output parsing",
    )

    def get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]


class GeminiVLMDetection(PipelineStage[DataSetOut, DetectionStageOutput]):
    """Detection model using Google's Gemini multimodal model with structured output parsing.

    Inherits from PipelineStage with explicitly defined input and output types.
    """

    def __init__(self, config: GeminiVLMDetectionConfig):
        """Initialize the Gemini VLM detection model.

        Args:
            config: Configuration for the detection model
        """
        super().__init__(config=config)
        self.config = config

        # Initialize parser
        self.parser = PydanticOutputParser(pydantic_object=self.config.output_type)

        # Initialize the Google GenAI client
        CONSOLE.log(f"Initialized Gemini detector with model: {self.config.model_name}")

    def entrypoint(self, input_data: DataSetOut) -> DetectionStageOutput:
        """Process a frame through the detection model.

        Args:
            rgb_frame: RGB frame as a numpy array
            user_prompt: Optional user prompt text

        Returns:
            Detection results with object metadata
        """
        return self._detect(input_data.rgb_image, input_data.user_prompt)

    def _detect(
        self, rgb_image, user_prompt: Optional[str] = None
    ) -> DetectionStageOutput:
        """
        Detect objects and analyze a scene using Gemini.

        Args:
            rgb_frame: RGB image as numpy array
            user_prompt: Optional user prompt text

        Returns:
            DetectionStageOutput with structured detection results
        """
        try:

            # Use provided prompt or default from config
            human_prompt = user_prompt or self.config.default_prompt

            # Get the appropriate system prompt based on query type, formatted with instructions
            format_instructions = self.parser.get_format_instructions()
            system_prompt_with_format = self.config.base_system_prompt.format(
                format_instructions=format_instructions,
                max_objects=self.config.max_objects,
            )

            CONSOLE.log(
                f"Running Gemini detection with model: {self.config.model_name}"
            )

            # Generate content using direct API call
            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=[human_prompt, rgb_image],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt_with_format,
                    temperature=self.config.temperature,
                    # top_p=self.top_p,
                    # top_k=self.top_k,
                    safety_settings=self.get_safety_settings(),
                ),
            )

            # Extract text from response
            response_text = response.text

            # Parse response using Pydantic parser
            try:
                scene_description = self.parser.parse(response_text)

                CONSOLE.log(
                    f"{self.config.model_name} detected {len(scene_description.objects)} objects!"
                )

                return scene_description

            except Exception as parse_err:
                CONSOLE.warn(f"[yellow]Failed to parse response: {parse_err}")
                CONSOLE.log(f"Raw response: {response_text}...")
                raise parse_err

        except Exception as e:
            CONSOLE.warn(f"[red]Error in Gemini detection: {str(e)}")
            # Return empty scene description on failure
            raise e

    def get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.config.safety_settings
        ]
