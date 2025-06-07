from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from google import genai
from google.genai import types

from ..data_contracts.aabb_segmentation import AABBDetections
from ..data_contracts.dataset import DatasetOut, PipelineIn
from ..data_handling.stray_scanner.data_parser import StrayScannerDataParserConfig
from ..data_handling.stray_scanner.stray_dataset import StrayDataset, StrayDatasetConfig
from ..data_handling.stray_scanner.stray_scanner_paths import StrayScannerPaths
from ..utils import Console, PathConfig
from .gemini_aabb_detection import GeminiAABBDetSeg, GeminiAABBDetSegConfig

# Models allowed for vision‑language tasks (no Live‑API model included).
MODEL_OPTIONS: dict[str, str] = {
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview(05-20) - adaptive thinking, cost-efficient",
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro Preview (05-06) - enhanced reasoning, multimodal",
}


class ScenePipeline:
    """Minimal orchestrator for Streamlit app."""

    def __init__(self, dataset_dir: Path, model_name: str) -> None:
        self.console = Console.with_prefix(self.__class__.__name__, "__init__")
        self.dataset = self._create_dataset(dataset_dir)
        self.detector = self._create_detector(model_name)
        self._frame_cache: Dict[int, DatasetOut] = {}
        self._det_cache: Dict[int, AABBDetections] = {}
        self.client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

    def _create_dataset(self, dataset_dir: Path) -> StrayDataset:
        cfg = StrayDatasetConfig(
            data_parser_config=StrayScannerDataParserConfig(
                paths=StrayScannerPaths(dataset_dir=dataset_dir)
            )
        )
        return cfg.setup_target()

    def _create_detector(self, model_name: str) -> GeminiAABBDetSeg:
        cfg = GeminiAABBDetSegConfig(model_name=model_name)
        return cfg.setup_target()

    def load_frame(self, idx: int) -> DatasetOut:
        if idx not in self._frame_cache:
            self._frame_cache[idx] = self.dataset.entrypoint(PipelineIn(idx=idx))
        return self._frame_cache[idx]

    def run_detection(self, idx: int) -> AABBDetections:
        frame = self.load_frame(idx)
        if idx not in self._det_cache:
            self._det_cache[idx] = self.detector.entrypoint(frame)
        return self._det_cache[idx]

    def answer_query(
        self, user_input: Union[str, bytes], idx: int
    ) -> Tuple[str, Optional[bytes]]:
        frame = self.load_frame(idx)
        contents: list[types.Content] = [frame.rgb_image]
        if isinstance(user_input, bytes):
            contents.append(types.Blob(data=user_input, mime_type="audio/wav"))
        else:
            contents.append(types.Part.from_text(text=user_input))
        response = self.client.models.generate_content(
            model=self.detector.config.model_name,
            contents=contents,
        )
        text = response.text or ""
        audio: Optional[bytes] = None
        for cand in response.candidates:
            for part in cand.content.parts:
                if isinstance(part, bytes):
                    audio = bytes(part)
                    break
        return text, audio
