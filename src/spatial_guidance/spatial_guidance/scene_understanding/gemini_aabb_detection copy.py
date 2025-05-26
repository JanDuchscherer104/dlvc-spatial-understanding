from enum import Enum
from pydoc import resolve
from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, Union

from google import genai
from google.genai import types
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field
from zenml.steps import BaseStep

from ..pipeline.data_contracts import AABBDetections, DatasetOut
from ..pipeline.data_contracts_3d import (
    CombinedDetectionSegmentationOut,
    Detection3DStageOut,
    MultiviewPointsStageOut,
    PointDetectionStageOut,
    SegmentationStageOut,
)
from ..utils import BaseConfig, Console, PathConfig


class GeminiAABBDetSegConfig(BaseConfig["GeminiAABBDetSeg"]):
    """Configuration for Gemini VLM detection model."""

    target: Type["GeminiAABBDetSeg"] = Field(
        default_factory=lambda: GeminiAABBDetSeg,
        description="Target class to instantiate",
    )

    # Model-specific configuration
    model_name: Literal[
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-05-06",
    ] = Field(
        "gemini-2.5-flash-preview-04-17", description="Name of the Gemini model to use"
    )

    temperature: float = Field(
        0.35,
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
        10, description="Maximum number of objects to detect in a scene"
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

    # # Base system prompt template - will be formatted with specific instructions
    # system_prompt: str = Field(
    #     """
    #     You are an assistive AI for 3D scene understanding, helping visually impaired users navigate safely through their environment.

    #     DETECTION PRIORITIES (in order of importance):
    #     1. Immediate trip hazards (floor-level obstacles, cords, steps, uneven surfaces)
    #     2. Moving objects (vehicles, objects that might be in motion)
    #     3. Head-height dangers (hanging objects, branches, signs below 7ft)
    #     4. Navigation landmarks (doors, hallways, stairs, railings, elevator buttons)
    #     5. Structural elements (walls, furniture, plants, fixtures)

    #     FURTHER CONTEXT:
    #     - Prioritize safety-critical objects that pose immediate risks
    #     - Focus on practical, actionable spatial information
    #     - Use concise directional language
    #     - The system is designed for real-time blind navigation assistance
    #     - Limit detection to {max_objects} most critical objects

    #     {format_instructions}
    #     """,
    # )
    # system_prompt: str = Field(
    #     """
    #     Provide the segmentation masks and bounding of all obstacles / crowds are relevant in the context of a scene description to guide visually impaired users.
    #     """
    # )
    system_prompt: str = Field(
        """
        You are an assistive AI for real-time navigation assistance for visually impaired users. Identify immediate hazards—obstacles or crowds that pose tripping or collision risk and could block the path. For each hazard, output "label", 2D bounding box [y1, x1, y2, x2], and a segmentation mask. Prioritize hazards closest to the user and highest in danger. Focus on practical, actionable spatial information to guide safe movement.

        - Prioritize the detection of safety-critical objects that pose immediate risks and that are closest to the user
        - Detection priorities: moving hazards (vehicles or other dynamic objects) > immediate trip hazards > head-height dangers (overhead obstacles like low branches, hanging signs)
        - Limit detection to {max_objects} most critical objects
        - Do not include objects that are not relevant to the users navigation
        """
    )
    # Output type for structured parsing
    output_type: Type[AABBDetections] = Field(
        default=AABBDetections,
        description="Pydantic model to use for structured output parsing",
    )

    def get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]


class GeminiAABBDetSeg(BaseStep):
    """Detection model using Google's Gemini multimodal model with structured output parsing.

    Inherits from PipelineStage with explicitly defined input and output types.
    """

    def __init__(
        self, config: Optional[GeminiAABBDetSegConfig] = None, **step_kwargs: Any
    ):
        """Initialize the Gemini VLM detection model.

        Args:
            config: Configuration for the detection model
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        super().__init__(**step_kwargs)
        self.config = config or GeminiAABBDetSegConfig()

        self.parser = PydanticOutputParser(pydantic_object=self.config.output_type)

        CONSOLE.log(f"Initialized Gemini detector with model: {self.config.model_name}")

    def entrypoint(
        self, input_data: DatasetOut
    ) -> Annotated[AABBDetections, "Detection-Results"]:
        """Process a frame through the detection model.

        Args:
            rgb_frame: RGB frame as a numpy array
            user_prompt: Optional user prompt text

        Returns:
            Detection results with object metadata
        """
        return self._detect(input_data.rgb_image, input_data.user_prompt)

    def _detect(self, rgb_image, user_prompt: Optional[str] = None) -> AABBDetections:
        """
        Detect objects and analyze a scene using Gemini.

        Args:
            rgb_frame: RGB image as numpy array
            user_prompt: Optional user prompt text

        Returns:
            DetectionStageOutput with structured detection results
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_detect")
        try:

            # Get the appropriate system prompt, formatted with instructions
            format_instructions = self.parser.get_format_instructions()
            system_prompt_with_format = self.config.system_prompt.format(
                format_instructions="",
                max_objects=self.config.max_objects,
            )

            CONSOLE.log(
                f"Running Gemini detection with model: {self.config.model_name}"
            )

            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=[user_prompt, rgb_image] if user_prompt else [rgb_image],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt_with_format,
                    temperature=self.config.temperature,
                    response_schema=self.config.output_type,
                    response_mime_type="application/json",
                    # top_p=self.top_p,
                    # top_k=self.top_k,
                    safety_settings=self.config.get_safety_settings(),
                ),
            )

            if response.parsed is None:
                CONSOLE.warn(f"Gemini returned no parsed results.")
                CONSOLE.plog(response)
                return AABBDetections(objects=[])

            CONSOLE.log(
                f"{self.config.model_name} detected {len(response.parsed.objects)} objects in the scene."
            )

            return response.parsed

        except Exception as e:
            CONSOLE.warn(f"[red]Error in Gemini detection: {str(e)}")
            # Return empty scene description on failure
            raise e
