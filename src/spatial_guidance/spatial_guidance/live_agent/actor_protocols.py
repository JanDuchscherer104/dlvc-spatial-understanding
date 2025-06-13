import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..data_contracts import AABBDetections


@dataclass
class _Cmd:
    """Base command class for actor communication."""

    def __post_init__(self):
        # Add timing info at creation
        if not hasattr(self, "query_id"):
            self.query_id = str(uuid.uuid4())
        if not hasattr(self, "timestamp"):
            self.timestamp = time.time()


@dataclass
class AskCmd(_Cmd):
    """Command to ask a text question."""

    text: str
    frame_idx: Optional[int] = None
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioCmd(_Cmd):
    """Command to send audio data."""

    pcm_data: bytes
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class SetFrameCmd(_Cmd):
    """Command to set current frame context."""

    frame_idx: int
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class _Evt:
    """Base event class for actor communication."""

    pass


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

    detections: AABBDetections
    frame_idx: int
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class ErrorEvt(_Evt):
    """Event containing error information."""

    error: str
    query_id: Optional[str] = None
    response_time_ms: Optional[float] = None
