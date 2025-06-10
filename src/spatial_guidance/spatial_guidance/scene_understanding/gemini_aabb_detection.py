from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from google.genai import types
from PIL import Image as PILImage
from pydantic import Field

from ..data_contracts.aabb_segmentation import (
    AABBDetection,
    AABBDetections,
    RawAABBDetSeg,
)
from ..data_contracts.dataset import DatasetOut
from ..utils import BaseConfig, Console
from ..gemini_client import GeminiClient, GeminiClientConfig
from ..visualization.detection_visualizer import DetectionVisualizer


class GeminiAABBDetSegConfig(BaseConfig["GeminiAABBDetSeg"]):
    """Configuration for Gemini VLM detection model."""

    target: Type["GeminiAABBDetSeg"] = Field(
        default_factory=lambda: GeminiAABBDetSeg,
    )

    mask_confidence_threshold: float = Field(0.45, gt=0.0, le=1.0)
    """Confidence threshold for the segmentation mask to be considered valid."""

    show_3d_info_in_visualization: bool = True
    """Whether to display 3D information (centers and rotation) in the visualization labels."""

    # Model-specific configuration
    model_name: Literal[
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
    ] = "gemini-2.5-pro-preview-05-06"
    """Name of the Gemini model to use"""
    temperature: Optional[float] = 0.5
    """Controls randomness in the output. Lower values make output more deterministic."""
    top_p: Optional[float] = None  # 0.9
    """
    "Nucleus sampling: Consider the smallest set of tokens whose probability sum exceeds top_p"
    """
    top_k: Optional[int] = None  # 40
    """Only sample from the top k most likely tokens at each step"""
    candidate_count: Optional[int] = None
    """Number of candidates to generate. If None, defaults to 1."""

    max_objects: int = 10
    "Maximum number of objects to detect in a scene"

    request_timeout: Union[int, float] = 25
    """Timeout for the request to the Gemini model in seconds."""

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
    )
    """Safety settings for the model"""

    base_system_prompt: str = (
        "You are an advanced VLM, trained for precise obstacle detection and segmentation to help visually-impaired users navigate. "
        "Based on the image provided, you can operate in two modes:\n\n"
        "1. **FULL DETECTION MODE** (when no specific user request is given): Detect and segment ALL objects relevant for navigation including:\n"
        "   - Moveable hazards: objects which can move (vehicles, cyclists, trains, revolving doors, escalators)\n"
        "   - Trip hazards: cords, curbs, scooters, clutter, obstacles lying on walking surfaces\n"
        "   - Head-level hazards: low signs, branches, overhangs\n"
        "   - Navigation landmarks: doors, stairs, ramps, handrails, crossings, elevator entries\n\n"
        "2. **SUBSET DETECTION MODE** (when user requests specific objects): Focus ONLY on detecting and segmenting the specific objects, categories, or types mentioned in the user's request. Be precise and only return objects that match the user's criteria.\n\n"
        "Always ignore irrelevant scene elements like far-away objects, buildings, sky, or decorative objects unless specifically requested.\n"
        "The output should be a JSON list of objects. Each object must conform to the provided schema with unique and descriptive labels.\n"
    )

    default_prompt: str = (
        "FULL DETECTION MODE: Detect all navigation-relevant objects including:\n"
        "- Moveable hazards: all objects which can move (e.g. vehicles, cyclists, trains)\n"
        "- Trip hazards: cords, curbs, scooters, clutter, arbitrary obstacles lying on the walking surface\n"
        "- Head-level hazards: low signs, branches, overhangs\n"
        "- Navigation landmarks: doors, stairs, ramps, handrails, crossings, elevator entries, escalators\n"
    )

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
            response_schema=RawAABBDetSeg.get_json_schema(
                as_list=True, max_length=self.max_objects
            ),
            response_mime_type="application/json",
            system_instruction=self.base_system_prompt,
        )


class GeminiAABBDetSeg:
    """Detection model using Google's Gemini multimodal model with structured output parsing.

    Inherits from PipelineStage with explicitly defined input and output types.
    """

    def __init__(
        self,
        config: Optional[GeminiAABBDetSegConfig] = None,
        *,
        gemini_client: Optional[GeminiClient] = None,
        **step_kwargs: Any,
    ):
        """Initialize the Gemini VLM detection model.

        Args:
            config: Configuration for the detection model
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        super().__init__(**step_kwargs)
        self.config = config or GeminiAABBDetSegConfig()
        self.visualizer = DetectionVisualizer()
        self.gemini_client = gemini_client or GeminiClient(
            GeminiClientConfig(model_name=self.config.model_name)
        )

        CONSOLE.log(f"Initialized Gemini detector with model: {self.config.model_name}")

    def run_aabb_detection(
        self,
        input_data: DatasetOut,
        user_prompt: Optional[str] = None,
        subset_mode: bool = False,
    ) -> Annotated[AABBDetections, "Detection-Results"]:
        """Process a frame through the detection model.

        Args:
            input_data: DatasetOut object containing rgb_image, depth_image and user_prompt
            user_prompt: Optional user prompt text to guide the detection

        Returns:
            Detection results with object metadata and visualizations
        """
        # Decide which prompt to pass to Gemini
        if subset_mode and user_prompt:
            detection_prompt = (
                f"SUBSET DETECTION MODE: {user_prompt}\n"
                "Only detect and segment objects that match this specific request."
            )
        else:
            detection_prompt = user_prompt

        raw_detections = self._detect(input_data.rgb_image, detection_prompt)

        if not raw_detections:
            # Return empty detections if nothing was found, but still provide original images for context
            return AABBDetections(
                objects=[],
                visualization_rgb=input_data.rgb_image,
                visualization_depth=input_data.depth_image,
            )

        processed_detections = self._process(
            rgb_image=input_data.rgb_image,
            raw_detections=raw_detections,
            depth_image=input_data.depth_image,
            camera_intrinsics=input_data.camera_intrinsics,
            camera_pose=input_data.camera_pose,
        )

        # Generate visualizations using DetectionVisualizer
        processed_detections.visualization_rgb = (
            self.visualizer.visualize_rgb_detections(
                input_data.rgb_image,
                processed_detections,
                show_3d_info=self.config.show_3d_info_in_visualization,
            )
        )
        processed_detections.visualization_depth = (
            self.visualizer.visualize_depth_detections(
                input_data.depth_image,
                processed_detections,
                img_width=input_data.rgb_image.width,
                img_height=input_data.rgb_image.height,
                show_3d_info=self.config.show_3d_info_in_visualization,
            )
        )

        return processed_detections

    def _detect(
        self, rgb_image: PILImage.Image, user_prompt: Optional[str] = None
    ) -> list[RawAABBDetSeg]:
        """
        Detect objects and analyze a scene using Gemini.

        Args:
            rgb_image: RGB image as PIL Image
            user_prompt: Optional user prompt text

        Returns:
            List of RawAABBDetection objects
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_detect")

        try:
            CONSOLE.log(
                f"Running Gemini detection with model: {self.config.model_name}"
            )

            contents = [rgb_image]
            if user_prompt:
                contents.append(types.Part.from_text(text=user_prompt))
            else:
                contents.append(types.Part.from_text(text=self.config.default_prompt))

            self.gemini_client.add_message(
                "user",
                user_prompt or self.config.default_prompt,
                tags=["aabb"],
            )

            response = self.gemini_client.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                generation_config=self.config.get_generation_config(),
            )
            parsed_detections: list[RawAABBDetSeg]
            if response.parsed is not None:
                parsed_detections_raw = response.parsed
                parsed_detections = [
                    RawAABBDetSeg.model_validate(det) for det in parsed_detections_raw
                ]
            else:
                CONSOLE.log(
                    "Gemini did not return parsed results. Attempting to parse raw JSON output."
                )
                raw_json_output = response.text
                if not raw_json_output:
                    CONSOLE.warn("Gemini returned empty text response.")
                    return []

                CONSOLE.dbg(
                    f"Raw Gemini response text: {raw_json_output[:500]}"
                )  # Log beginning of response

                parsed_detections = RawAABBDetSeg.parse_json_list(
                    raw_json_output,
                    Console.with_prefix(CONSOLE.prefix, "parse_json_list"),
                )

            if not parsed_detections:
                CONSOLE.warn(
                    f"Gemini returned no parsable results after _parse_json. Original text was: {raw_json_output[:200] if raw_json_output else 'None'}"
                )
                return []

            CONSOLE.log(
                f"{self.config.model_name} detected and parsed {len(parsed_detections)} objects in the scene."
            )

            self.gemini_client.add_message(
                "assistant",
                response.text or "",
                tags=["aabb"],
            )

            return parsed_detections

        except Exception as e:
            CONSOLE.warn(f"[red]Error in Gemini detection: {str(e)}")
            raise e

    def _process(
        self,
        rgb_image: PILImage.Image,  # Ensure type is PILImage.Image
        raw_detections: list[RawAABBDetSeg],
        depth_image: Optional[PILImage.Image] = None,  # Ensure type is PILImage.Image
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
    ) -> AABBDetections:
        """
        Process the detection results to convert them into a structured format.

        Args:
            rgb_image: RGB image as PIL Image
            raw_detections: List of raw detection results
            depth_image: Optional depth image for depth statistics calculation (PIL Image)
            camera_intrinsics: Optional 3x3 camera intrinsics matrix
            camera_pose: Optional 4x4 world-to-camera transformation matrix

        Returns:
            AABBDetections with structured detection results
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "process")

        # Convert raw detections to AABBDetection objects
        # Convert bboxes to np arrays and convert masks from base64 to PIL Images
        detections_list = []
        for raw_det in raw_detections:
            try:
                detections_list.append(
                    AABBDetection.model_validate(raw_det.model_dump())
                )
            except Exception as e:
                CONSOLE.error(f"Error processing object {raw_det.label}: {e}")
                continue

        processed_detections = AABBDetections(objects=detections_list)

        # process_all will normalize the bbounding boxes and masks and scale them correctly
        processed_detections.process_all(
            img_size=rgb_image.size,  # Pass PIL image size
            confidence_thresh=self.config.mask_confidence_threshold,
            depth_image=depth_image,  # Pass PIL depth image
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
        )

        CONSOLE.log(
            f"Processed {len(processed_detections.objects)} detections with masks and bounding boxes."
        )

        return processed_detections
