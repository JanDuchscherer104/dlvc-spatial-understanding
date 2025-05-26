from typing import Annotated, Any, List, Literal, Optional, Tuple, Type

import numpy as np
from google import genai
from google.genai import types
from PIL import Image as PILImage
from pydantic import Field
from zenml.steps import BaseStep

from ..data_contracts.aabb_segmentation import AABBDetections
from ..data_contracts.dataset import DatasetOut
from ..data_contracts.obb_detection import OBBDetection, OBBDetections, RawOBBDetection
from ..utils import BaseConfig, Console, PathConfig
from ..visualization.detection_visualizer import DetectionVisualizer


class GeminiOBBDetConfig(BaseConfig["GeminiOBBDet"]):
    """Configuration for Gemini VLM 3D OBB detection model."""

    target: Type["GeminiOBBDet"] = Field(
        default_factory=lambda: GeminiOBBDet,
    )
    use_camera_pose: bool = False

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

    max_objects: int = 10
    "Maximum number of 3D objects to detect in a scene"

    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "BLOCK_ONLY_HIGH"),
        ],
    )
    """Safety settings for the model"""

    base_system_prompt: str = (
        "You are an expert 3D spatial analysis system for visually impaired navigation assistance. "
        "Estimate 3D bounding boxes for detected obstacles using the camera's perspective.\n\n"
        "COORDINATE SYSTEM:\n"
        "- Origin: Camera position\n"
        "- X-axis: Right (positive = right of camera)\n"
        "- Y-axis: Forward (positive = away from camera)\n"
        "- Z-axis: Up (positive = above ground)\n"
        "- Units: meters\n"
        "- Angles: degrees (roll=rotation around Y, pitch=rotation around X, yaw=rotation around Z)\n\n"
        "3D BOUNDING BOX FORMAT:\n"
        "[x_center, y_center, z_center, width, height, depth, roll, pitch, yaw]\n\n"
        "ESTIMATION GUIDELINES:\n"
        "- Be conservative with sizes - better to overestimate than underestimate\n"
        "- Ground-level objects: z_center = height/2\n"
        "- Typical object sizes: person=1.7m height, car=4.5m length, scooter=1.2m length\n"
        "- Most objects have roll=0, pitch=0, yaw depends on orientation\n"
        "- Distance estimation: use relative size and perspective cues\n\n"
        "FOCUS ON ACCURACY:\n"
        "- Only estimate for clearly visible objects\n"
        "- Use realistic proportions and physics\n"
        "- Consider perspective distortion near image edges"
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
            response_schema=RawOBBDetection.get_json_schema(
                as_list=True, max_length=self.max_objects
            ),
            response_mime_type="application/json",
            system_instruction=self.base_system_prompt,
        )


class GeminiOBBDet(BaseStep):
    """3D OBB Detection model using Google's Gemini multimodal model."""

    def __init__(self, config: Optional[GeminiOBBDetConfig] = None, **step_kwargs: Any):
        CONSOLE = Console.with_prefix(self.__class__.__name__)
        super().__init__(**step_kwargs)
        self.config = config or GeminiOBBDetConfig()
        self.visualizer = DetectionVisualizer()

        CONSOLE.log(
            f"Initialized Gemini 3D OBB detector with model: {self.config.model_name}"
        )

    def entrypoint(
        self, input_data: DatasetOut, aabb_detections: Optional[OBBDetections] = None
    ) -> Annotated[OBBDetections, "3D-OBB-Detection-Results"]:
        """Process an image through the 3D OBB detection model."""
        CONSOLE = Console.with_prefix(self.__class__.__name__, "entrypoint")
        raw_detections = self._detect(
            input_data.rgb_image, input_data.user_prompt, aabb_detections
        )

        # Get image dimensions for context, though not strictly for OBB processing itself
        img_width, img_height = input_data.rgb_image.size

        if not raw_detections:
            CONSOLE.log("No raw 3D OBB detections returned from model.")
            return OBBDetections(
                objects=[], image_width=img_width, image_height=img_height
            )

        # Convert and process raw detections
        obb_detections_container = self._process(
            input_data=input_data,
            raw_detections=raw_detections,
        )

        CONSOLE.log(
            f"Processed {len(obb_detections_container.objects)} 3D OBB detections."
        )

        return obb_detections_container

    def _detect(
        self,
        rgb_image: PILImage.Image,
        user_prompt: Optional[str] = None,
        aabb_detections: Optional[AABBDetections] = None,
    ) -> List[RawOBBDetection]:
        """Detect 3D OBBs using Gemini."""
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_detect")
        try:
            CONSOLE.log(
                f"Running Gemini 3D OBB detection with model: {self.config.model_name}"
            )

            contents = [
                rgb_image,
            ]
            if aabb_detections:
                contents.append(
                    types.Part.from_text(
                        f"Provide the 3D bounding boxes for the following objects:\n{aabb_detections.to_json_list()}"
                    )
                )
            if user_prompt:
                contents.append(types.Part.from_text(text=user_prompt))

            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=self.config.get_generation_config(),
            )

            if response.parsed is not None:
                parsed_detections = response.parsed
                parsed_detections = [
                    RawOBBDetection.model_validate(det) for det in parsed_detections
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

                parsed_detections = RawOBBDetection.parse_json_list(
                    raw_json_output,
                    Console.with_prefix(CONSOLE.prefix, "parse_json_list"),
                )

            if not parsed_detections:
                CONSOLE.warn(
                    f"Gemini returned no parsable results after _parse_json. Original text was: {raw_json_output[:200]}"
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
        input_data: DatasetOut,
        raw_detections: List[RawOBBDetection],
    ) -> OBBDetections:
        """Convert raw detections into processed OBBDetections."""
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_process")
        img_width, img_height = input_data.rgb_image.size

        detections_list: List[OBBDetection] = []
        for raw_det in raw_detections:
            try:
                detections_list.append(
                    OBBDetection.model_validate(raw_det.model_dump())
                )
            except Exception as e:
                CONSOLE.error(f"Error processing raw detection '{raw_det.label}': {e}")

        obb_detections = OBBDetections(objects=detections_list)

        # Pass camera intrinsics and pose for proper 3D projection
        obb_detections.process_all(
            img_size=(img_width, img_height),
            camera_intrinsics=input_data.camera_intrinsics,
            camera_pose=input_data.camera_pose if self.config.use_camera_pose else None,
        )

        CONSOLE.log(
            f"Processed {len(obb_detections.objects)} OBB detections with camera intrinsics."
        )
        return obb_detections

