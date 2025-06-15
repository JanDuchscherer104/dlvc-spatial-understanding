from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from click import Option
from google import genai
from google.genai import types
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import Field

from spatial_guidance.utils import base_config

from ..data_contracts.aabb_segmentation import (
    AABBDetection,
    AABBDetections,
    RawAABBDetSeg,
)
from ..data_contracts.dataset import DatasetOut
from ..utils import BaseConfig, Console, PathConfig
from ..visualization.detection_visualizer import DetectionVisualizer


class GeminiAABBDetSegConfig(BaseConfig["GeminiAABBDetSeg"]):
    """Configuration for Gemini VLM detection model."""

    target: Type["GeminiAABBDetSeg"] = Field(
        default_factory=lambda: GeminiAABBDetSeg,
    )
    is_debug: bool = True

    mask_confidence_threshold: float = Field(0.45, gt=0.0, le=1.0)
    """Confidence threshold for the segmentation mask to be considered valid."""
    resize: Optional[Tuple[int, int]] = (640, 640)
    """Resize the input image to this size before processing. If None, no resizing is done."""

    visualize_rgb: bool = False
    """Whether to visualize RGB image with detection results."""
    visualize_depth: bool = False
    """Whether to visualize depth information in the output."""
    show_boxes_in_visualization: bool = False
    """Whether to display text boxes in the visualization."""
    show_3d_info_in_visualization: bool = True
    """Whether to display 3D information (centers and rotation) in the visualization labels."""

    # Model-specific configuration
    model_name: Literal[
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
    ] = "gemini-2.5-pro-preview-05-06"
    """Name of the Gemini model to use"""

    # TODO: Tune further by logging schema violation rate vs settings; raise temperature by +0.1 only if detection recall is < 90 % over five frames.
    temperature: Optional[float] = 0.5
    """Controls randomness in the output. Lower values make output more deterministic."""
    top_p: Optional[float] = None
    """
    "Nucleus sampling: Consider the smallest set of tokens whose probability sum exceeds top_p"
    """
    top_k: Optional[int] = None
    """Only sample from the top k most likely tokens at each step. Smaller k speeds up generation."""
    candidate_count: Optional[int] = None
    """Number of candidates to generate. If None, defaults to 1."""

    max_objects: int = 5
    "Maximum number of objects to detect in a scene"

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

    # base_system_prompt: str = (
    #     "You are an advanced VLM, trained for precise obstacle detection and segmentation to help visually-impaired users navigate. "
    #     "Based on the image provided, you can operate in two modes:\n\n"
    #     "1. **FULL DETECTION MODE** (when no specific user request is given): Detect and segment ALL objects relevant for navigation including:\n"
    #     "   - Moveable hazards: objects which can move (vehicles, cyclists, trains, revolving doors, escalators)\n"
    #     "   - Trip hazards: cords, curbs, scooters, clutter, obstacles lying on walking surfaces\n"
    #     "   - Head-level hazards: low signs, branches, overhangs\n"
    #     "   - Navigation landmarks: doors, stairs, ramps, handrails, crossings, elevator entries\n\n"
    #     "2. **SUBSET DETECTION MODE** (when user requests specific objects): Focus ONLY on detecting and segmenting the specific objects, categories, or types mentioned in the user's request. Be precise and only return objects that match the user's criteria.\n\n"
    #     "Always ignore irrelevant scene elements like far-away objects, buildings, sky, or decorative objects unless specifically requested.\n"
    #     "The output should be a JSON list of objects. Each object must conform to the provided schema with unique and descriptive labels.\n"
    #     "If labels are provided in the user request, they must be used as-is. E.g. if the uers quer is 'bicycle and scooter', the labels of the two detected objects must be 'bicycle' and 'scooter'.\n"
    # )
    base_system_prompt: str = """\
You are an advanced VLM, trained for precise object detection and segmentation to assist visually impaired users with navigation. Always provide structured outputs that conform to the provided schema.

Use descriptive labels that help users understand both what objects are and their relevance to navigation (e.g., "wooden handrail along stairs", "parked red car blocking sidewalk", "glass entrance door with metal handle"). If the user specifies labels in their request, use those exact labels (e.g., if the user asks for "bicycle and scooter", label the detected objects as "bicycle" and "scooter").

Always ignore irrelevant scene elements like far-away objects, buildings, floor, grass or decorative objects unless specifically requested.

"""

    full_detection_prompt: str = (
        "Detect all navigation-relevant objects including:\n"
        "- Moveable hazards: all objects which can move (e.g. vehicles, cyclists, trains)\n"
        "- Hazardous areas: train tracks, revolving doors, road crossings and intersections, steep slopes or stairs"
        "- Trip hazards: arbitrary obstacles lying on the walking surface (e.g. clutter, high curbs)\n"
        "- Head-level hazards: low signs, branches, overhangs\n"
        "- Navigation landmarks: public transport stops or platforms, wayfinding signs (e.g. street signs, directional markers, platform signs) doors, stairs, ramps, handrails, crossings, entries, escalators, pedestrian crossings\n"
    )

    category_detection_prompts: dict[str, str] = Field(
        default_factory=lambda: dict(
            hazards=(
                "Detect hazardous objects including: "
                "moveable hazards (vehicles, cyclists, trains), "
                "hazardous areas (train tracks, revolving doors, road crossings, intersections, steep slopes), "
                "trip hazards (clutter, cords, high curbs), "
                "head-level hazards (low signs, branches, overhangs)"
            ),
            navigation_landmarks=(
                "Detect navigation landmarks including: "
                "public transport stops or platforms, "
                "wayfinding signs (street signs, directional markers, platform signs), "
                "doors, stairs, ramps, handrails, crossings, entries, escalators, pedestrian crossings"
            ),
        )
    )

    subset_detection_prompt: str = (
        "Detect and segment ONLY the objects specified in the user request. "
        "Focus on the specific objects, categories, or types mentioned in the request. Be precise and only return objects that match the user's criteria."
    )

    path_description_prompt: str = (
        "\nTASK: Path Planning and Navigation Analysis\n"
        "You are conducting a comprehensive path analysis for navigation guidance. Think step-by-step and try to understand what you are seeing before providing the detection and segmentation results:\n"
        "1. Identifying the destination and reflect upon potential routes\n"
        "2. Select the safest and most efficient route\n"
        "3. Identify all obstacles, hazards and navigation landmarks along the path\n"
        "After completing your thought process, detect potential hazards and landmarks along the path.\n"
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
            max_output_tokens=4096,
            response_mime_type="application/json",
            system_instruction=self.base_system_prompt,
        )


class GeminiAABBDetSeg:
    """Detection model using Google's Gemini multimodal model with structured output parsing.

    Inherits from PipelineStage with explicitly defined input and output types.
    """

    def __init__(self, config: Optional[GeminiAABBDetSegConfig] = None):
        """Initialize the Gemini VLM detection model.

        Args:
            config: Configuration for the detection model
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        self.config = config or GeminiAABBDetSegConfig()
        self.visualizer = DetectionVisualizer()

        CONSOLE.log(f"Initialized Gemini detector with model: {self.config.model_name}")

    def run_aabb_detection(
        self,
        input_data: DatasetOut,
        user_prompt: Optional[str] = None,
        detection_mode: Optional[str] = None,
    ) -> Annotated[AABBDetections, "Detection-Results"]:
        """Process a frame through the detection model.

        Args:
            input_data: DatasetOut object containing rgb_image, depth_image and user_prompt
            user_prompt: Optional user prompt text to guide the detection

        Returns:
            Detection results with object metadata and visualizations
        """
        # Validate and handle detection_mode
        if detection_mode is None:
            # Fallback: if user_prompt is provided, assume subset mode
            if user_prompt:
                detection_mode = "subset"
            else:
                # No user_prompt, do full detection
                detection_mode = None

        # Decide which prompt to pass to Gemini based on detection_mode
        if detection_mode == "subset" and user_prompt:
            detection_prompt = f"{self.config.subset_detection_prompt}: {user_prompt}"
        elif detection_mode == "path_description" and user_prompt:
            detection_prompt = f"{self.config.path_description_prompt}: {user_prompt}"
        elif detection_mode in ("hazards", "navigation_landmarks"):
            detection_prompt = self.config.category_detection_prompts[detection_mode]
        else:
            # Full detection or default mode
            detection_prompt = None

        raw_detections = self._detect(input_data.rgb_image, detection_prompt)

        if not raw_detections:
            # Return empty detections if nothing was found, but still provide original images for context
            return AABBDetections(
                objects={},
                visualization_rgb=input_data.rgb_image,
                visualization_depth=input_data.depth_image,
            )

        processed_detections = self._process(
            rgb_image=input_data.rgb_image,
            raw_detections=raw_detections,
            depth_image=input_data.depth_image,
            camera_intrinsics=input_data.camera_intrinsics,
            camera_pose=input_data.camera_pose,
            ground_plane=input_data.ground_plane,
        )

        # Generate visualizations using DetectionVisualizer
        if self.config.visualize_rgb:
            processed_detections.visualization_rgb = (
                self.visualizer.visualize_rgb_detections(
                    input_data.rgb_image,
                    processed_detections,
                    show_boxes=self.config.show_boxes_in_visualization,
                    show_3d_info=self.config.show_3d_info_in_visualization,
                )
            )
        if self.config.visualize_depth:
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
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_detect").set_debug(
            self.config.is_debug
        )

        try:
            CONSOLE.log(
                f"Running Gemini detection with model: {self.config.model_name}"
            )

            if self.config.resize:
                rgb_image = rgb_image.copy()
                rgb_image.thumbnail(self.config.resize, PILImage.Resampling.LANCZOS)

            contents = [
                rgb_image,
            ]
            if user_prompt:
                contents.append(types.Part.from_text(text=user_prompt))
            else:
                contents.append(
                    types.Part.from_text(text=self.config.full_detection_prompt)
                )

            # TODO: client should be provided by our GeminiClient class!
            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=self.config.get_generation_config(),
            )
            parsed_detections: list[RawAABBDetSeg]
            raw_json_output: Optional[str] = None
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
                    f"Gemini returned no parsable results after _parse_json. Original text was: {response.text[:200] if response.text else 'None'}"
                )
                return []

            CONSOLE.log(
                f"{self.config.model_name} detected and parsed {len(parsed_detections)} objects in the scene."
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
        ground_plane: Optional[Tuple[np.ndarray, float]] = None,
    ) -> AABBDetections:
        """
        Process the detection results to convert them into a structured format.

        Args:
            rgb_image: RGB image as PIL Image
            raw_detections: List of raw detection results
            depth_image: Optional depth image for depth statistics calculation (PIL Image)
            camera_intrinsics: Optional 3x3 camera intrinsics matrix
            camera_pose: Optional 4x4 world-to-camera transformation matrix
            ground_plane: Optional ground plane definition (normal vector and distance)

        Returns:
            AABBDetections with structured detection results
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "process").set_debug(
            self.config.is_debug
        )

        # Convert raw detections to AABBDetection objects
        # Convert bboxes to np arrays and convert masks from base64 to PIL Images
        detections_list = []
        for raw_det in raw_detections:
            try:
                detections_list.append(
                    AABBDetection.model_validate(raw_det.model_dump())
                )
            except Exception as e:
                CONSOLE.error(e, f"Error processing object {raw_det.label}")
                continue

        processed_detections = AABBDetections.from_list(detections_list)

        # process_all will normalize the bbounding boxes and masks and scale them correctly
        processed_detections.process_all(
            img_size=rgb_image.size,  # Pass PIL image size
            confidence_thresh=self.config.mask_confidence_threshold,
            depth_image=depth_image,  # Pass PIL depth image
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            ground_plane=ground_plane,
        )

        CONSOLE.log(
            f"Processed {len(processed_detections.objects)} detections with masks and bounding boxes."
        )

        return processed_detections
