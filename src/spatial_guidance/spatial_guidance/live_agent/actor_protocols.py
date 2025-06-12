from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _Cmd:
    """Base command class for actor communication."""

    pass


@dataclass
class AskCmd(_Cmd):
    """Command to ask a text question."""

    text: str
    frame_idx: Optional[int] = None


@dataclass
class AudioCmd(_Cmd):
    """Command to send audio data."""

    pcm_data: bytes


@dataclass
class SetFrameCmd(_Cmd):
    """Command to set current frame context."""

    frame_idx: int


@dataclass
class _Evt:
    """Base event class for actor communication."""

    pass


@dataclass
class TextEvt(_Evt):
    """Event containing text response."""

    text: str


@dataclass
class AudioEvt(_Evt):
    """Event containing audio response."""

    audio_data: bytes


@dataclass
class DetectionsEvt(_Evt):
    """Event containing detection results."""

    detections: Any  # AABBDetections type
    frame_idx: int


@dataclass
class ErrorEvt(_Evt):
    """Event containing error information."""

    error: str
