import json
import traceback
from math import e
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from google import genai
from google.genai import types
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel, Field
from zenml.steps import BaseStep

from ..data_contracts.aabb_segmentation import AABBDetection, AABBDetections
from ..data_contracts.dataset import DatasetOut
from ..data_contracts.scene_description import SceneDescription
from ..utils import BaseConfig, Console, PathConfig


class GeminiSceneDescriptorConfig(BaseConfig["GeminiSceneDescriptor"]):
    """Configuration for Gemini VLM scene description model."""

    target: Type["GeminiSceneDescriptor"] = Field(
        default_factory=lambda: GeminiSceneDescriptor,
    )

    # Model-specific configuration
    model_name: Literal[
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
    ] = "gemini-2.5-pro-preview-05-06"
    """Name of the Gemini model to use"""

    temperature: Optional[float] = 0.5
    """Controls randomness in the output. Lower values make output more deterministic."""

    top_p: Optional[float] = None
    """Nucleus sampling: Consider the smallest set of tokens whose probability sum exceeds top_p"""

    top_k: Optional[int] = None
    """Only sample from the top k most likely tokens at each step"""

    enable_async_processing: bool = True
    """Whether to enable asynchronous processing for multiple candidates."""

    combine_candidates: bool = True
    """Whether to combine results from multiple candidates into a single response."""

    # Safety settings
    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "BLOCK_ONLY_HIGH"),
        ],
    )
    """Safety settings for the model"""

    # Scene description specific settings
    include_distance_estimates: bool = True
    """Whether to include distance estimates in the description."""

    focus_on_navigation: bool = True
    """Whether to focus on navigation-relevant aspects of the scene."""

    detailed_spatial_analysis: bool = True
    """Whether to provide detailed spatial relationship analysis."""

    accessibility_focus: bool = True
    """Whether to include accessibility considerations for visually impaired users."""

    # TODO: base_system_prompt should be configurable given the config params above.
    base_system_prompt: str = (
        "You are an advanced spatial understanding AI assistant specifically designed to help visually impaired users "
        "navigate and understand their environment. Your role is to provide comprehensive, accurate, and helpful "
        "scene descriptions that prioritize safety and navigation guidance.\n\n"
        "You will be provided with:\n"
        "1. An RGB image of the scene\n"
        "2. A JSON list of detected objects with their properties including:\n"
        "   - label: descriptive name of the object\n"
        "   - med_depth: median distance in meters\n"
        "   - rotation_clock: clock position (1-12) indicating direction relative to user\n"
        "   - box_2d: bounding box coordinates in the image\n"
        "   - rotation_deg: precise bearing angle in degrees\n\n"
        "Provide your response as a JSON object with three fields:\n\n"
        "1. **immediate_safety_hazards**: Brief description of the most critical safety hazards that pose immediate risks, "
        "including their distance and clock position. Focus on moving objects, path obstacles, and head-level hazards. "
        "If no hazards exist, state 'No immediate safety hazards detected.'\n\n"
        "2. **scene_description**: Comprehensive description of all relevant detections organized by spatial relationships "
        "using clock positions and distances. Include navigation landmarks, obstacles, and environmental features.\n\n"
        "3. **navigation_guidance**: Specific, actionable guidance for safe movement, recommended paths, areas to avoid, "
        "and direction-specific advice when user destinations are provided.\n\n"
        "Use precise spatial language with clock positions (e.g., '2 o'clock', '11 o'clock') and metric distances. "
        "Prioritize safety-critical information and context relevant to any user prompt provided.\n\n"
        "Respond only with the JSON object containing these three fields."
    )

    def _get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]

    def get_generation_config(self) -> types.GenerateContentConfig:
        """Generate the configuration for the Gemini model.

        Args:
            candidate_count: Override the default candidate count for this specific request
        """
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            safety_settings=self._get_safety_settings(),
            system_instruction=self.base_system_prompt,
            response_mime_type="application/json",
            response_schema=SceneDescription.get_json_schema(),
        )


class GeminiSceneDescriptor(BaseStep):
    """Scene description model using Google's Gemini multimodal model for spatial understanding.

    Provides comprehensive scene analysis and navigation guidance specifically designed
    for visually impaired users, including spatial relationships, safety considerations,
    and detailed environmental descriptions.
    """

    def __init__(
        self, config: Optional[GeminiSceneDescriptorConfig] = None, **step_kwargs: Any
    ):
        """Initialize the Gemini scene descriptor.

        Args:
            config: Configuration for the scene descriptor
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        super().__init__(**step_kwargs)
        self.config = config or GeminiSceneDescriptorConfig()

        CONSOLE.log(
            f"Initialized Gemini scene descriptor with model: {self.config.model_name}"
        )

    def entrypoint(
        self, dataset_out: DatasetOut, detections: AABBDetections
    ) -> SceneDescription:
        """Process a frame through the scene description model.

        Args:
            dataset_out: DatasetOut object containing rgb_image, depth_image, user_prompt, camera_intrinsics, camera_pose
            detections: AABBDetections object containing detected objects with spatial information

        Returns:
            SceneDescription object with structured spatial analysis
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "entrypoint")

        try:
            # Extract components from dataset_out
            rgb_image = dataset_out.rgb_image
            depth_image = dataset_out.depth_image
            user_prompt = dataset_out.user_prompt

            # Analyze the scene with detections
            scene_description = self._analyze_scene_with_detections(
                rgb_image=rgb_image,
                detections=detections,
                user_prompt=user_prompt,
                depth_image=depth_image,
            )

            CONSOLE.log("Scene analysis completed successfully")
            return scene_description

        except Exception as e:
            CONSOLE.error(e, "Error in scene description entrypoint")
            return SceneDescription(
                immediate_safety_hazards="Error occurred during scene analysis.",
                scene_description="Error occurred during scene analysis. Please try again.",
                navigation_guidance="Unable to provide navigation guidance due to system error.",
            )

    def _convert_detections_to_json(self, detections: AABBDetections) -> str:
        """Convert AABBDetections to JSON format for the prompt.

        Args:
            detections: AABBDetections object

        Returns:
            JSON string with detection data
        """
        detection_list = []

        for obj in detections.objects.values():
            detection_dict = {
                "label": obj.label,
                "med_depth": obj.med_depth,
                "rotation_clock": obj.rotation_clock,
                "rotation_deg": obj.rotation_deg,
                "box_2d": obj.box_2d.tolist() if obj.box_2d is not None else None,
                "min_depth": obj.min_depth,
                "max_depth": obj.max_depth,
            }
            detection_list.append(detection_dict)

        return json.dumps(detection_list, indent=2)

    def _analyze_scene_with_detections(
        self,
        rgb_image: PILImage.Image,
        detections: AABBDetections,
        user_prompt: Optional[str] = None,
        depth_image: Optional[PILImage.Image] = None,
    ) -> SceneDescription:
        """
        Analyze scene using Gemini with grounded detection data.

        Args:
            rgb_image: RGB image as PIL Image
            detections: AABBDetections object with spatial information
            user_prompt: Optional user prompt text
            depth_image: Optional depth image

        Returns:
            SceneDescription object with structured analysis
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "_analyze_scene_with_detections"
        )

        try:
            CONSOLE.log(
                f"Running Gemini scene analysis with model: {self.config.model_name}"
            )

            # Convert detections to JSON
            detections_json = self._convert_detections_to_json(detections)

            # Prepare content for the request
            contents = [rgb_image]

            # # Add depth image if available
            # if depth_image is not None:
            #     contents.append(depth_image)
            #     contents.append(
            #         types.Part.from_text(
            #             text="The second image is a depth map corresponding to the RGB image. "
            #             "Use this to enhance spatial understanding, but prioritize the precise "
            #             "distance measurements provided in the detection data below."
            #         )
            #     )

            # Add detection data
            detection_prompt = (
                f"Here are the detected objects in the scene with their spatial properties:\n\n"
                f"```json\n{detections_json}\n```\n\n"
                "Each object includes:\n"
                "- label: descriptive name\n"
                "- med_depth: median distance in meters\n"
                "- rotation_clock: clock position (1-12) relative to your forward direction (12 o'clock)\n"
                "- rotation_deg: precise bearing angle in degrees\n"
                "- box_2d: bounding box coordinates [y0, x0, y1, x1] in pixels\n\n"
                "Use this data to provide precise spatial descriptions and navigation guidance."
            )
            contents.append(types.Part.from_text(text=detection_prompt))

            # Add user prompt if provided
            if user_prompt:
                enhanced_prompt = (
                    f"\nUser's specific request or context: {user_prompt}\n\n"
                    "Please provide a scene description that addresses the user's needs while "
                    "prioritizing safety and navigation guidance. Focus on objects most relevant "
                    "to the user's request."
                )
                contents.append(types.Part.from_text(text=enhanced_prompt))
            else:
                # Default guidance prompt
                default_prompt = (
                    "\nPlease provide a comprehensive scene description focusing on:\n"
                    "1. Immediate safety hazards and obstacles\n"
                    "2. Clear navigation pathways\n"
                    "3. Spatial organization of objects using clock positions and distances\n"
                    "4. Any objects that might be relevant for navigation or safety\n\n"
                    "Prioritize the most important information for safe movement through this environment."
                )
                contents.append(types.Part.from_text(text=default_prompt))

            # Make API call
            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=self.config.get_generation_config(),
            )

            # Parse structured response similar to gemini_aabb_detection.py
            if response.parsed is not None:
                # Direct structured parsing succeeded
                scene_desc = SceneDescription.model_validate(response.parsed)
                CONSOLE.log(
                    "Scene analysis completed successfully with structured parsing"
                )
                return scene_desc
            else:
                # Fallback to manual JSON parsing
                CONSOLE.log(
                    "Gemini did not return parsed results. Attempting to parse raw JSON output."
                )
                raw_json_output = response.text
                if not raw_json_output:
                    CONSOLE.warn("Gemini returned empty text response.")
                    return SceneDescription(
                        immediate_safety_hazards="No immediate safety hazards detected.",
                        scene_description="Unable to analyze scene due to empty response.",
                        navigation_guidance="Unable to provide navigation guidance. Please try again.",
                    )

                CONSOLE.dbg(f"Raw Gemini response text: {raw_json_output[:500]}")

                try:
                    # Try to parse as JSON directly
                    import json

                    parsed_json = json.loads(raw_json_output)
                    scene_desc = SceneDescription.model_validate(parsed_json)
                    CONSOLE.log(
                        "Scene analysis completed successfully with manual JSON parsing"
                    )
                    return scene_desc
                except (json.JSONDecodeError, Exception) as e:
                    CONSOLE.error(e, "Failed to parse JSON response")
                    # Return a fallback response with the raw text in scene_description
                    return SceneDescription(
                        immediate_safety_hazards="Unable to parse safety hazards from response.",
                        scene_description=f"Raw response: {raw_json_output[:500]}...",
                        navigation_guidance="Unable to provide structured navigation guidance. Please try again.",
                    )

        except Exception as e:
            CONSOLE.error(e, "Error during Gemini scene analysis")
            return SceneDescription(
                immediate_safety_hazards="Error occurred during safety hazard analysis.",
                scene_description=f"Error during scene analysis: {str(e)}",
                navigation_guidance="Unable to provide navigation guidance due to analysis error. Please try again.",
            )
