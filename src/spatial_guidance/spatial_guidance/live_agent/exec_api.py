import asyncio
import contextlib
import io
import sys
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

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

    @property
    def last_detections(self) -> dict[str, Any]:
        """Returns the last detections for code-execution snippets."""
        # Flatten dict for backward compatibility
        return {
            f"{idx}_{lab}": det
            for (idx, lab), det in self._agent._last_detections.items()
        }

    @property
    def current_frame_idx(self) -> Optional[int]:
        """Returns the current frame index."""
        return self._agent.current_frame_idx

    def get_camera_pose(self, frame_idx: int) -> np.ndarray:
        """Returns the camera pose for the given frame index."""
        return self._agent.dataset[frame_idx].camera_pose  # type: ignore

    def run_aabb_detection(self, *args, **kwargs):
        """Proxy to agent's AABB detector for exec-code snippets."""
        return self._agent.aabb_detector.run_aabb_detection(
            self._agent.dataset[self._agent.current_frame_idx], *args, **kwargs
        )
