from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, TypeVar

import numpy as np
from google import genai
from google.genai import types
from PIL import Image as PILImage
from pydantic import Field
from zenml.steps import BaseStep

from ..data_contracts import DataModel
from ..utils.console import Console
from ..utils.base_config import BaseConfig

TargetType = TypeVar("TargetType", bound="GeminiBaseModel")
OutputSchemaType = TypeVar("OutputSchemaType", bound=DataModel)


class GeminiBaseConfig(BaseConfig[TargetType]):

    output_schema_type: Type[OutputSchemaType]
    """"""
    output_as_list: bool = False
    """Whether to return output as a list of objects or a single object"""

    # Model-specific configuration
    model_name: Literal[
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
    ] = "gemini-2.0-flash"
    temperature: Optional[float] = 0.5
    """Controls randomness in the output. Lower values make output more deterministic."""
    top_p: Optional[float] = None
    "Nucleus sampling: Consider the smallest set of tokens whose probability sum exceeds top_p"
    top_k: Optional[int] = None
    """Only sample from the top k most likely tokens at each step"""
    candidate_count: Optional[int] = None
    """Number of candidates to generate. If None, defaults to 1."""

    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "BLOCK_ONLY_HIGH"),
        ],
    )
    """Safety settings for the model"""

    ### Task-specific configuration
    base_system_prompt: str

    max_objects: Optional[int] = None

    def _get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]

    def get_generation_config(self) -> types.GenerateContentConfig:
        """Generate the configuration for the Gemini model."""
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            candidate_count=self.candidate_count,
            safety_settings=self._get_safety_settings(),
            response_schema=self.output_schema_type.get_json_schema(
                as_list=self.output_as_list, max_length=self.max_objects
            ),
            response_mime_type="application/json",
            system_instruction=self.base_system_prompt,
        )


class GeminiBaseModel(BaseStep):
    config: GeminiBaseConfig

    def __init__(self, **step_kwargs: Any):

        super().__init__(**step_kwargs)

    def infer_gemini(self, )


    def _parse_response(self, response: types.GenerateContentResponse, console: Console) -> List[OutputSchemaType]:
        """Parse a Gemini response into RawAABBDetSeg objects."""
        if response.parsed is not None:
            parsed_detections = response.parsed
            parsed_detections = [
                self.config.output_schema_type.model_validate(det) for det in parsed_detections
            ]
        else:
            console.log(
                "Gemini did not return parsed results. Attempting to parse raw JSON output."
            )
            raw_json_output = response.text
            if not raw_json_output:
                console.warn("Gemini returned empty text response.")
                return []

            console.dbg(f"Raw Gemini response text: {raw_json_output[:500]}")

            parsed_detections = self.config.output_schema_type.parse_json_list(
                raw_json_output,
                self.config.output_schema_type.with_prefix(console.prefix, "parse_json_list"),
            )

        if not parsed_detections:
            console.warn("Gemini returned no parsable results")
            return []

        console.log(f"Parsed {len(parsed_detections)} objects from response")


        return parsed_detections