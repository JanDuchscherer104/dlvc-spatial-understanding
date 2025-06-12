from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types

from ..data_contracts.aabb_segmentation import AABBDetections
from ..data_contracts.dataset import DatasetOut
from ..data_handling.stray_scanner.data_parser import StrayScannerDataParserConfig
from ..data_handling.stray_scanner.stray_dataset import StrayDataset, StrayDatasetConfig
from ..data_handling.stray_scanner.stray_scanner_paths import StrayScannerPaths
from ..gemini_client import GeminiClient, GeminiClientConfig, OperationalMode
from ..response_generation import DirectionalStyle, DistanceStyle, ResponseGenerator
from ..utils import Console, PathConfig
from .gemini_aabb_detection import GeminiAABBDetSeg, GeminiAABBDetSegConfig

# Models allowed for vision‑language tasks (no Live‑API model included).
MODEL_OPTIONS: dict[str, str] = {
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview(05-20) - adaptive thinking, cost-efficient",
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro Preview (05-06) - enhanced reasoning, multimodal",
}


class ScenePipeline:
    """Enhanced orchestrator for Streamlit app with dynamic operational modes."""

    def __init__(
        self, dataset_dir: Path, model_name: str, is_rotated: bool = True
    ) -> None:
        self.console = Console.with_prefix(self.__class__.__name__, "__init__")
        self.dataset = self._create_dataset(dataset_dir, is_rotated)
        self.detector = self._create_detector(model_name)
        self._frame_cache: Dict[int, DatasetOut] = {}
        self._det_cache: Dict[int, AABBDetections] = {}
        self._subset_det_cache: Dict[Tuple[int, str], AABBDetections] = {}

        # Initialize the new modular components
        gemini_config = GeminiClientConfig(
            api_key=PathConfig().get_api_key("GOOGLE_API_KEY"),
            model_name=model_name,
            max_history_length=20,
            context_window_tokens=8000,
            auto_mode_detection=True,
            default_mode=OperationalMode.GENERAL_SCENE,
        )
        self.gemini_client = GeminiClient(gemini_config)
        self.response_generator = ResponseGenerator(
            api_key=PathConfig().get_api_key("GOOGLE_API_KEY")
        )
        # Set preferred response styles
        self.response_generator.set_response_styles(
            DirectionalStyle.CLOCK_FACE, DistanceStyle.APPROXIMATE
        )

        # Keep track of current operational mode
        self.current_mode = OperationalMode.GENERAL_SCENE

    def _create_dataset(self, dataset_dir: Path, is_rotated: bool) -> StrayDataset:
        cfg = StrayDatasetConfig(
            is_rotated=is_rotated,
            data_parser=StrayScannerDataParserConfig(
                paths=StrayScannerPaths(dataset_dir=dataset_dir)
            ),
        )
        return cfg.setup_target()

    def _create_detector(self, model_name: str) -> GeminiAABBDetSeg:
        cfg = GeminiAABBDetSegConfig(model_name=model_name)
        return cfg.setup_target()

    def load_frame(self, idx: int) -> DatasetOut:
        if idx not in self._frame_cache:
            self._frame_cache[idx] = self.dataset[idx]
        return self._frame_cache[idx]

    def run_detection(self, idx: int) -> AABBDetections:
        frame = self.load_frame(idx)
        if idx not in self._det_cache:
            self._det_cache[idx] = self.detector.run_aabb_detection(frame)
        return self._det_cache[idx]

    def run_subset_detection(self, idx: int, user_prompt: str) -> AABBDetections:
        """Run subset detection for specific objects based on user prompt."""
        cache_key = (idx, user_prompt)
        if cache_key not in self._subset_det_cache:
            frame = self.load_frame(idx)
            self._subset_det_cache[cache_key] = self.detector.run_aabb_detection(
                frame, user_prompt, subset_mode=True
            )
        return self._subset_det_cache[cache_key]

    def clear_chat_history(self) -> None:
        """Clear the persistent chat history."""
        self.gemini_client.clear_history()

    def add_to_chat_history(self, user_message: str, assistant_response: str) -> None:
        """Add messages to the persistent chat history."""
        self.gemini_client.add_user_message(user_message)
        self.gemini_client.add_assistant_message(assistant_response)

    def set_operational_mode(self, mode: OperationalMode) -> None:
        """Set the current operational mode."""
        self.current_mode = mode
        self.gemini_client.set_mode(mode)

    def get_operational_mode(self) -> OperationalMode:
        """Get the current operational mode."""
        return self.current_mode

    def answer_query_with_context(
        self, user_input: Union[str, bytes], idx: int, include_history: bool = True
    ) -> Tuple[str, Optional[bytes]]:
        """Answer query with enhanced context and mode-specific responses."""
        frame = self.load_frame(idx)

        # Handle audio input - for now, convert to text query
        if isinstance(user_input, bytes):
            # TODO: Implement audio transcription
            text_query = "Audio input received - please process the scene"
        else:
            text_query = user_input

        # Detect and switch mode if auto-detection is enabled
        if self.gemini_client.config.auto_mode_detection:
            detected_mode = self.gemini_client.detect_mode(text_query)
            if detected_mode != self.current_mode:
                self.set_operational_mode(detected_mode)

        # Check if this is a detection request that needs object data
        if self.is_detection_request(text_query):
            # Run detection and get objects
            detections = self.run_detection(idx)
            # Generate natural language response with detection results
            detection_response = self.response_generator.generate_response(
                detections, self.current_mode, text_query
            )
            # Use detection-enhanced response
            enhanced_query = f"{text_query}\n\nDetected objects: {detection_response}"
            response_text = self.gemini_client.generate_response(
                enhanced_query, frame.rgb_image
            )
        else:
            # Regular query processing
            response_text = self.gemini_client.generate_response(
                text_query, frame.rgb_image
            )

        # For now, audio output is not implemented
        audio_output = None

        return response_text, audio_output

    def is_detection_request(self, user_input: str) -> bool:
        """Check if user input is requesting object detection using enhanced keywords."""
        detection_keywords = [
            "detect",
            "find",
            "show",
            "identify",
            "locate",
            "where",
            "what",
            "objects",
            "things",
            "items",
            "see",
            "spot",
            "point out",
            "chair",
            "table",
            "person",
            "car",
            "door",
            "stairs",
            "obstacle",
            "furniture",
            "appliance",
            "tool",
            "food",
            "kitchen",
            "bathroom",
            "bedroom",
            "living room",
            "count",
            "how many",
            "list",
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in detection_keywords)

    def run_subset_detection_with_response(
        self, idx: int, user_prompt: str
    ) -> Tuple[AABBDetections, str]:
        """Run subset detection and generate natural language response."""
        detections = self.run_subset_detection(idx, user_prompt)

        # Generate natural language response
        if detections.objects:
            response = self.response_generator.generate_response(
                detections, self.current_mode, user_prompt
            )
        else:
            response = (
                f"I couldn't find any objects matching your request: '{user_prompt}'"
            )

        return detections, response
