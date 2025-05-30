import asyncio
import time
from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from google import genai
from google.genai import types
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import Field
from zenml.steps import BaseStep

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

    mask_confidence_threshold: float = Field(0.45, gt=0.0, le=1.0)
    """Confidence threshold for the segmentation mask to be considered valid."""

    show_3d_info_in_visualization: bool = False
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
    candidate_count: Optional[int] = 3
    """Number of candidates to generate. If None, defaults to 1."""

    max_concurrent_requests: int = 3
    """Maximum number of concurrent requests to send for multiple candidates."""

    enable_async_processing: bool = True
    """Whether to enable asynchronous processing for multiple candidates."""

    combine_candidates: bool = True
    """Whether to combine results from multiple candidates into a single response."""

    min_valid_mask_size: Tuple[int, int] = (2, 2)
    """Minimum mask size (width, height) to consider a segmentation mask valid."""

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
        # "Your task is to detect and segment objects in the provided image, which are relevant for a visually impaired "
        # "person navigating the scene. "
        # "Prioritize potential hazards like obstacles, entities that might be in motion (e.g. vehicles, revolving doors), "
        # ""
        # "Do not include items that are not relevant to the task, such as far-away objects, buildings, sky, or decorative objects."
        # "{format_instructions}"
        # "The segmentation mask must be provided as a base64 encoded PNG."
        # "Always provide valid AABBs and base64 encoded segmentation masks in the exact way you were trained. "
        # "You are an advanced VLM, trained for object detection and segmentation. "
        "You are an advanced VLM, trained for precise obstacle detection and segmentation to help visually-impaired users navigate. "
        "Only detect objects that:\n"
        # "1. Might move or collide (vehicles, bikes, people).\n"
        # "2. Trip hazards that lie on the walking surface (cords, curbs, scooters).\n"
        # "3. Obstruct head height (overhangs, low signs, branches).  "
        # "4. Serve as navigation landmarks (doors, stairs, ramps, handrails, crossings).  "
        "- Moveable hazards: all objects which can move (e.g. vehicles, cyclists, trains)\n"
        "- Trip hazards: cords, curbs, scooters, clutter, arbitrary obstacles lying on the walking surface\n"
        "- Head-level hazards: low signs, branches, overhangs\n"
        "- Navigation landmarks: doors, stairs, ramps, handrails, crossings, elevator entries, escalators, \n"
        "Ignore other scene elements that are not relevant to the task, such as far-away objects, buildings, sky, or decorative objects.\n"
        "The output should be a JSON list of objects. Each object in the list must conform to the provided schema. Make sure to provide unique and descriptive labels for each object.\n"
    )
    # base_system_prompt: str = (
    #     "You are a component in an AI system for assisting visually impaired users. "
    #     "You are an advanced VLM, trained for obstacle detection and segmentation to help visually-impaired users navigate. "
    #     "Based on the image provided, detect and segment objects that are relevant for navigation. "
    #     "Only detect objects that:\n"
    #     "  - Are moveable hazards: all objects which can move (e.g. vehicles, cyclists, trains, revolving doors, escalators).\n"
    #     "  - Are trip hazards: cords, curbs, scooters, clutter, arbitrary obstacles lying on the walking surface.\n"
    #     "  - Are head-level hazards: low signs, branches, overhangs.\n"
    #     "  - Serve as navigation landmarks: doors, stairs, ramps, handrails, crossings, elevator entries, escalators.\n"
    #     "Ignore other scene elements that are not relevant to the task, such as far-away objects, buildings, sky, or decorative objects.\n"
    #     # "Your output must strictly adhere to the provided JSON schema to ensure system compatibility. "
    #     "The output should be a JSON list of objects. Each object in the list must conform to the provided schema. Make sure to provide unique and descriptive labels for each object.\n"
    #     # "For the \'mask\' field, provide ONLY the valid base64 encoded string of the PNG image. Do not include any prefixes like \'data:image/png;base64,\' or any other text. "
    #     # "The base64 string for \'mask\' must only contain valid ASCII characters (A-Z, a-z, 0-9, +, /, =). "
    # )

    def _get_safety_settings(self) -> List[types.SafetySetting]:
        """Convert safety settings from config to genai types."""
        return [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in self.safety_settings
        ]

    def get_generation_config(
        self, candidate_count: Optional[int] = None
    ) -> types.GenerateContentConfig:
        """Generate the configuration for the Gemini model.

        Args:
            candidate_count: Override the default candidate count for this specific request
        """
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            candidate_count=candidate_count or self.candidate_count,
            safety_settings=self._get_safety_settings(),
            response_schema=RawAABBDetSeg.get_json_schema(
                as_list=True, max_length=self.max_objects
            ),
            response_mime_type="application/json",
            system_instruction=self.base_system_prompt,
        )


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
        self.visualizer = DetectionVisualizer()

        CONSOLE.log(f"Initialized Gemini detector with model: {self.config.model_name}")

    def entrypoint(
        self, input_data: DatasetOut
    ) -> Annotated[AABBDetections, "Detection-Results"]:
        """Process a frame through the detection model.

        Args:
            input_data: DatasetOut object containing rgb_image, depth_image and user_prompt

        Returns:
            Detection results with object metadata and visualizations
        """
        raw_detections = self._detect(input_data.rgb_image, input_data.user_prompt)

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
        if (
            self.config.enable_async_processing
            and self.config.candidate_count
            and self.config.candidate_count > 1
        ):
            # Use async processing for multiple candidates
            return self._run_async_in_sync_context(
                self._detect_async_multiple_candidates(rgb_image, user_prompt)
            )
        else:
            # Use synchronous processing for single candidate
            return self._detect_single_candidate(rgb_image, user_prompt)

    def _run_async_in_sync_context(self, coro):
        """
        Run an async coroutine in a synchronous context, handling both
        interactive environments (with existing event loops) and standalone scripts.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine execution
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an interactive environment with a running loop
                # Use nest_asyncio if available, otherwise run in executor
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    return asyncio.run(coro)
                except ImportError:
                    # nest_asyncio not available, run in thread pool
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, coro)
                        return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run() safely
            return asyncio.run(coro)

    def _detect_single_candidate(
        self, rgb_image: PILImage.Image, user_prompt: Optional[str] = None
    ) -> list[RawAABBDetSeg]:
        """
        Detect objects using a single request to Gemini.

        Args:
            rgb_image: RGB image as PIL Image
            user_prompt: Optional user prompt text

        Returns:
            List of RawAABBDetection objects
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "_detect_single_candidate"
        )

        try:
            CONSOLE.log(
                f"Running Gemini detection with model: {self.config.model_name}"
            )

            contents = [rgb_image]
            if user_prompt:
                contents.append(types.Part.from_text(text=user_prompt))

            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            response = client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=self.config.get_generation_config(candidate_count=1),
            )

            return self._parse_response(response, CONSOLE)

        except Exception as e:
            CONSOLE.warn(f"[red]Error in Gemini detection: {str(e)}")
            raise e

    async def _detect_async_multiple_candidates(
        self, rgb_image: PILImage.Image, user_prompt: Optional[str] = None
    ) -> list[RawAABBDetSeg]:
        """
        Detect objects using multiple concurrent requests to Gemini for better results.

        Args:
            rgb_image: RGB image as PIL Image
            user_prompt: Optional user prompt text

        Returns:
            Combined list of RawAABBDetection objects from all valid candidates
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "_detect_async_multiple_candidates"
        )

        try:
            CONSOLE.log(
                f"Running async Gemini detection with {self.config.candidate_count} candidates, "
                f"max {self.config.max_concurrent_requests} concurrent requests"
            )

            contents = [rgb_image]
            if user_prompt:
                contents.append(types.Part.from_text(text=user_prompt))

            # Create multiple concurrent requests
            tasks = []
            for i in range(
                min(
                    self.config.candidate_count or 1,
                    self.config.max_concurrent_requests,
                )
            ):
                task = asyncio.create_task(self._generate_candidate_async(contents, i))
                tasks.append(task)

            # Wait for responses with timeout
            start_time = time.time()
            completed_responses = []

            try:
                # Use asyncio.wait with timeout to get partial results
                # Don't use ALL_COMPLETED since we want to respect timeout
                done, pending = await asyncio.wait(
                    tasks, timeout=self.config.request_timeout
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    CONSOLE.warn(f"Task cancelled due to timeout")

                # Collect results from completed tasks
                for task in done:
                    try:
                        response = await task
                        if response is not None:
                            completed_responses.append(response)
                    except Exception as e:
                        CONSOLE.warn(f"Task failed: {e}")

            except Exception as e:
                CONSOLE.error(f"Unexpected error in async processing: {e}")
                # Cancel all remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()

            elapsed_time = time.time() - start_time
            CONSOLE.log(
                f"Completed {len(completed_responses)} candidates in {elapsed_time:.2f}s"
            )

            if not completed_responses:
                CONSOLE.warn("No valid responses received from any candidate")
                return []

            # Parse and filter valid detections from all candidates
            all_detections = []
            for i, response in enumerate(completed_responses):
                try:
                    detections = self._parse_response(
                        response, Console.with_prefix(CONSOLE.prefix, f"candidate_{i}")
                    )

                    # Filter valid masks
                    valid_detections = self._filter_valid_detections(detections)
                    all_detections.extend(valid_detections)

                    CONSOLE.log(
                        f"Candidate {i}: {len(valid_detections)}/{len(detections)} valid detections"
                    )

                except Exception as e:
                    CONSOLE.warn(f"Failed to parse candidate {i}: {e}")

            if self.config.combine_candidates:
                # Combine and deduplicate results
                combined_detections = self._combine_candidate_results(all_detections)
                CONSOLE.log(
                    f"Combined {len(all_detections)} detections into {len(combined_detections)} unique objects"
                )
                return combined_detections
            else:
                return all_detections

        except Exception as e:
            CONSOLE.error(f"Error in async Gemini detection: {str(e)}")
            raise e

    async def _generate_candidate_async(self, contents: List, candidate_id: int):
        """Generate a single candidate response asynchronously."""
        try:
            client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

            # Note: The current genai client may not support true async operations
            # This is a synchronous call wrapped in an async function
            # For true async, we'd need to use aiohttp or similar
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=self.config.get_generation_config(candidate_count=1),
                ),
            )
            return response

        except Exception as e:
            console = Console.with_prefix(
                self.__class__.__name__, f"candidate_{candidate_id}"
            )
            console.warn(f"Candidate {candidate_id} failed: {e}")
            return None  # Return None instead of raising exception

    def _parse_response(self, response, console: Console) -> list[RawAABBDetSeg]:
        """Parse a Gemini response into RawAABBDetSeg objects."""
        if response.parsed is not None:
            parsed_detections = response.parsed
            parsed_detections = [
                RawAABBDetSeg.model_validate(det) for det in parsed_detections
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

            parsed_detections = RawAABBDetSeg.parse_json_list(
                raw_json_output,
                Console.with_prefix(console.prefix, "parse_json_list"),
            )

        if not parsed_detections:
            console.warn("Gemini returned no parsable results")
            return []

        console.log(f"Parsed {len(parsed_detections)} objects from response")
        return parsed_detections

    def _filter_valid_detections(
        self, detections: list[RawAABBDetSeg]
    ) -> list[RawAABBDetSeg]:
        """Filter out detections with invalid segmentation masks."""
        valid_detections = []

        for detection in detections:
            try:
                # Validate the detection by creating an AABBDetection object
                temp_detection = AABBDetection.model_validate(detection.model_dump())

                # Check if mask is valid (not a 1x1 or too small image)
                mask_size = temp_detection.mask.size
                min_width, min_height = self.config.min_valid_mask_size

                if mask_size[0] >= min_width and mask_size[1] >= min_height:
                    valid_detections.append(detection)
                else:
                    Console.with_prefix(
                        self.__class__.__name__, "_filter_valid_detections"
                    ).warn(
                        f"Filtered out detection '{detection.label}' with invalid mask size: {mask_size}"
                    )

            except Exception as e:
                Console.with_prefix(
                    self.__class__.__name__, "_filter_valid_detections"
                ).warn(f"Failed to validate detection '{detection.label}': {e}")

        return valid_detections

    def _combine_candidate_results(
        self, all_detections: list[RawAABBDetSeg]
    ) -> list[RawAABBDetSeg]:
        """Combine and deduplicate detection results from multiple candidates."""
        if not all_detections:
            return []

        # Simple deduplication based on label and bounding box similarity
        unique_detections = []
        similarity_threshold = (
            0.3  # IoU threshold for considering detections as duplicates
        )

        for detection in all_detections:
            is_duplicate = False

            for existing in unique_detections:
                # Check if labels are similar and bounding boxes overlap significantly
                if self._are_detections_similar(
                    detection, existing, similarity_threshold
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_detections.append(detection)

        return unique_detections

    def _are_detections_similar(
        self, det1: RawAABBDetSeg, det2: RawAABBDetSeg, iou_threshold: float
    ) -> bool:
        """Check if two detections are similar enough to be considered duplicates."""
        # Check label similarity (simple string matching) # TODO: use semantic similarity score!
        if det1.label.lower() == det2.label.lower():
            # Calculate IoU of bounding boxes
            iou = self._calculate_iou(det1.box_2d, det2.box_2d)
            return iou > iou_threshold
        return False

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        # Convert from [y0, x0, y1, x1] format
        y1_1, x1_1, y2_1, x2_1 = box1
        y1_2, x1_2, y2_2, x2_2 = box2

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate union
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

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
