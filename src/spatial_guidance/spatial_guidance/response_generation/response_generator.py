"""
Natural Language Response Generator for the Spatial Understanding Agent.
Uses pure Gemini models to generate contextual responses from structured detection data.
NO hardcoded rules or manual NLP processing - everything is inferred by Gemini.
"""

import asyncio
import json
import os
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from google import genai
from google.genai import types

from ..data_contracts.aabb_segmentation import AABBDetection, AABBDetections
from ..gemini_client import OperationalMode
from ..utils import Console, PathConfig


class DirectionalStyle(Enum):
    """Different styles for expressing directions."""

    CLOCK_FACE = "clock"  # "at 2 o'clock"
    RELATIVE = "relative"  # "to your right"
    COMPASS = "compass"  # "northeast"
    DEGREES = "degrees"  # "30 degrees right"


class DistanceStyle(Enum):
    """Different styles for expressing distances."""

    PRECISE = "precise"  # "2.3 meters"
    APPROXIMATE = "approximate"  # "about 2 meters"
    RELATIVE = "relative"  # "arm's reach", "across the room"


class ResponseGenerator:
    """
    Generates natural language responses for spatial understanding using Gemini AI.
    Simplified implementation that uses the structured JSON output from detections.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the response generator with Gemini client."""
        self.console = Console.with_prefix(self.__class__.__name__)

        # Initialize Gemini client
        if api_key is None:
            # Use PathConfig to load API key from .env file
            path_config = PathConfig()
            api_key = path_config.get_api_key("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file or environment variables"
            )

        # Create Gemini client
        self.client = genai.Client(api_key=api_key)

        # Model configuration for response generation
        self.model_name = "gemini-1.5-flash"
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=512,
            candidate_count=1,
        )

        self.directional_style = DirectionalStyle.CLOCK_FACE
        self.distance_style = DistanceStyle.APPROXIMATE

    def set_response_styles(
        self, directional_style: DirectionalStyle, distance_style: DistanceStyle
    ):
        """Set the preferred styles for directional and distance descriptions."""
        self.directional_style = directional_style
        self.distance_style = distance_style

    def generate_response(
        self,
        detections: AABBDetections,
        mode: OperationalMode,
        user_query: Optional[str] = None,
    ) -> str:
        """
        Generate a natural language summary of detection results using Gemini.
        """
        try:
            # Use the structured JSON output from detections
            detections_json = detections.to_json_list()

            # Create simple prompt
            prompt = self._create_prompt(detections_json, mode, user_query)

            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Part.from_text(text=prompt)],
                config=self.generation_config,
            )
            result = str(response.text).strip() if response.text else ""

            self.console.log(
                f"Generated response for {mode.value} mode: {len(result)} characters"
            )
            return result

        except Exception as e:
            self.console.error(f"Error generating response: {e}")
            return self._get_fallback_response(detections)

    def _create_prompt(
        self,
        detections_json: str,
        mode: OperationalMode,
        user_query: Optional[str],
    ) -> str:
        """Create a simple prompt for Gemini using the structured detection JSON."""

        # Mode-specific context
        mode_contexts = {
            OperationalMode.GENERAL_SCENE: "Provide a general overview of the spatial scene",
            OperationalMode.OBJECT_DETECTION: "Focus on object identification and precise location details",
            OperationalMode.NAVIGATION_GUIDANCE: "Focus on navigation assistance, obstacles, and movement guidance",
            OperationalMode.COOKING_ASSISTANCE: "Focus on kitchen/cooking context, safety, and culinary assistance",
            OperationalMode.ACCESSIBILITY_SUPPORT: "Focus on accessibility features, interactive elements, and user assistance",
        }

        prompt = f"""You are a spatial understanding assistant. Analyze the detection data and provide a natural, conversational response.

OPERATIONAL MODE: {mode.value}
CONTEXT: {mode_contexts.get(mode, "General spatial understanding")}

DETECTION DATA (JSON):
{detections_json}

SPATIAL REFERENCE:
- rotation_clock: 12 = straight ahead, 3 = right, 6 = behind, 9 = left
- depth: distance in meters
- bbox: [y0, x0, y1, x1] bounding box coordinates

STYLE PREFERENCES:
- Directional style: {self.directional_style.value}
- Distance style: {self.distance_style.value}
- Use natural, conversational language
- Be helpful and contextually relevant"""

        if user_query:
            prompt += f"\n\nUSER QUERY: {user_query}"

        prompt += "\n\nProvide a natural language response based on the detection data."

        return prompt

    def _get_fallback_response(self, detections: AABBDetections) -> str:
        """Simple fallback when Gemini fails."""
        if not detections.objects:
            return "No objects detected in the current view."

        object_count = len(detections.objects)
        if object_count == 1:
            return f"I can see one {detections.objects[0].label}."
        else:
            return f"I can see {object_count} objects in the scene."
