"""Gemini Live API model wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from google import genai
from google.genai import types

from ..utils import Console
from ..data_contracts.dataset import DatasetOut
from .gemini_client import GeminiClient, GeminiClientConfig


class GeminiLive:
    """Wrapper around the Gemini Live API using :class:`GeminiClient`."""

    def __init__(
        self,
        config: Optional[GeminiClientConfig] = None,
        *,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.config = config or GeminiClientConfig(
            model_name="gemini-2.0-flash-live-001"
        )
        self.console = Console.with_prefix(self.__class__.__name__)
        self.gemini_client = gemini_client or GeminiClient(self.config)
        self.model = genai.GenerativeModel(self.config.model_name)

        self.console.log(
            f"Initialized GeminiLive with model: {self.config.model_name}"
        )

    # ------------------------------------------------------------------
    # Function calling
    # ------------------------------------------------------------------
    def _build_tools(self) -> List[types.Tool]:
        """Return default tools for expert model function calls."""

        detect_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "image": types.Schema(type=types.Type.BYTES, format="jpeg"),
                "prompt": types.Schema(type=types.Type.STRING),
            },
            required=["image"],
        )

        aabb_func = types.FunctionDeclaration(
            name="run_aabb_detection",
            description="Detect AABBs in an image",
            parameters=detect_schema,
        )
        obb_func = types.FunctionDeclaration(
            name="run_obb_detection",
            description="Detect OBBs in an image",
            parameters=detect_schema,
        )

        return [types.Tool(function_declarations=[aabb_func, obb_func])]

    def _dispatch_function(self, name: str, args: Dict[str, Any]) -> str:
        """Call expert models based on Gemini function calls."""
        image = args.get("image")
        prompt = args.get("prompt")
        import numpy as np

        dataset = DatasetOut(
            rgb_image=image,
            depth_image=image.convert("L"),
            camera_intrinsics=args.get("camera_intrinsics") or np.eye(3),
            camera_pose=args.get("camera_pose") or np.eye(4),
        )

        if name == "run_aabb_detection":
            from ..scene_understanding import gemini_aabb_detection

            detector = gemini_aabb_detection.GeminiAABBDetSeg(gemini_aabb_detection.GeminiAABBDetSegConfig(), gemini_client=self.gemini_client)
            dets = detector.run_aabb_detection(dataset, prompt, subset_mode=bool(prompt))
            return dets.to_json_list()
        if name == "run_obb_detection":
            from ..scene_understanding import gemini_obb_detection

            detector = gemini_obb_detection.GeminiOBBDet(gemini_obb_detection.GeminiOBBDetConfig(),)
            dets = detector.entrypoint(dataset)
            return dets.to_json_list()

        self.console.warn(f"Unknown function call: {name}")
        return ""

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def stream(
        self,
        parts: Iterable[types.Part],
        *,
        tags: Optional[List[str]] = None,
    ) -> Iterable[types.GenerateContentResponse]:
        """Stream responses from the live API and handle function calls."""
        tools = self._build_tools()
        chat = self.model.start_chat(tools=tools)
        for response in chat.send_iter(parts):
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if isinstance(part, types.FunctionCall):
                    result = self._dispatch_function(part.name, part.args)
                    yield from chat.send_iter([
                        types.Part.from_function_response(
                            name=part.name, response=result
                        )
                    ])
                    continue
            self.gemini_client.add_message(
                "assistant", response.text or "", tags=tags
            )
            yield response

    def chat(self, text: str, frame: DatasetOut) -> str:
        """Simple helper for text+image chat with the live model."""
        self.gemini_client.add_message("user", text, tags=["live"])
        parts = [frame.rgb_image, types.Part.from_text(text)]
        responses = list(self.stream(parts, tags=["live"]))
        return responses[-1].text if responses else ""

