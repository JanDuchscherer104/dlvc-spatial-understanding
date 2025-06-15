import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Cmd:
    """Base command class for actor communication."""

    query_id: Optional[str] = field(default=None)
    timestamp: Optional[float] = field(default=None)

    def __post_init__(self):
        if self.query_id is None:
            self.query_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()


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

    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class TextEvt(_Evt):
    """Event containing text response."""

    text: str
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class AudioEvt(_Evt):
    """Event containing audio response."""

    audio_data: bytes
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class DetectionsEvt(_Evt):
    """Event containing detection results."""

    detections: Any  # AABBDetections type
    frame_idx: int
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class ErrorEvt(_Evt):
    """Event containing error information."""

    error: str
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None
