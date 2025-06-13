import asyncio
import contextlib
import io
import sys
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from google.genai import types

if TYPE_CHECKING:
    from .gemini_live_agent import GeminiLiveAgent


class _ExecAPI:
    def __init__(self, agent: "GeminiLiveAgent"):
        self._agent = agent

    async def _execute_python(self, code: str) -> str:
        """
        Execute a snippet of Python code (with access to `default_api` and `detections`) and capture stdout.
        """

        def runner():
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec_globals = {
                        "default_api": self,
                    }
                    exec(code, exec_globals, {})
            except Exception as e:
                buf.write(f"Error executing code: {e}")
            return buf.getvalue()

        # run in thread to avoid blocking event loop
        return await asyncio.to_thread(runner)

    def get_last_detections(
        self, frame_idx: int, labels: list[str]
    ) -> list[dict[str, Any]]:
        """Return cached detections for the given frame and labels."""
        result = self._make_tool_call(
            name="get_last_detections",
            args={"frame_idx": frame_idx, "labels": labels},
        )
        return result.get("detections", [])

    def list_all_detections(self) -> dict[int, list[str]]:
        """Return an overview of all cached detections: mapping frame index to list of labels."""
        result = self._make_tool_call(
            name="list_all_detections",
            args={},
        )
        return result.get("overview", {})

    @property
    def current_frame_idx(self) -> Optional[int]:
        """Returns the current frame index."""
        return self._agent.current_frame_idx

    def get_camera_pose(self, frame_idx: int) -> np.ndarray:
        """Returns the camera pose for the given frame index."""
        return self._agent.dataset[frame_idx].camera_pose  # type: ignore

    def run_aabb_detection(
        self, user_prompt: str, subset_mode: bool
    ) -> list[dict[str, Any]]:
        """Proxy to agent's AABB detector for exec-code snippets."""
        result = self._make_tool_call(
            name="run_aabb_detection",
            args={"user_prompt": user_prompt, "subset_mode": subset_mode},
        )
        return result.get("detections", [])

    def _make_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Synchronously invoke the agent's tool-call coroutine and return its JSON response."""
        fut = asyncio.run_coroutine_threadsafe(
            self._agent._handle_tool_call(
                tool_call=types.LiveServerToolCall(
                    function_calls=[types.FunctionCall(name=name, args=args)]
                ),
                return_no_send=True,
            ),
            self._agent.loop,
        )
        func_resp = fut.result()
        return func_resp.response
