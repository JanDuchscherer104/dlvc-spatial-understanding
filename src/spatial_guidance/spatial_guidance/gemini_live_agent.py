import asyncio
import base64
import io
import sys
import threading
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from queue import Empty  # <-- Added per instruction
from queue import Queue
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pyaudio
from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .data_contracts.aabb_segmentation import (
    AABBDetection,
    AABBDetections,
    RawAABBDetSeg,
)
from .data_contracts.dataset import DatasetOut
from .data_handling.stray_scanner.data_parser import StrayScannerDataParserConfig
from .data_handling.stray_scanner.stray_dataset import StrayDataset, StrayDatasetConfig
from .response_generation import DirectionalStyle, DistanceStyle, ResponseGenerator
from .scene_understanding.gemini_aabb_detection import (
    GeminiAABBDetSeg,
    GeminiAABBDetSegConfig,
)
from .utils import BaseConfig, Console, PathConfig
from .visualization.detection_visualizer import DetectionVisualizer

MODEL_OPTIONS: dict[str, str] = {
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview(05-20) - adaptive thinking, cost-efficient",
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro Preview (05-06) - enhanced reasoning, multimodal",
}


class InteractionMode(Enum):
    TEXT = auto()
    VOICE = auto()


# Actor pattern command and event classes
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


class GeminiLiveAgentConfig(BaseConfig["GeminiLiveAgent"]):
    target: Type["GeminiLiveAgent"] = Field(default_factory=lambda: GeminiLiveAgent)

    # Dataset configuration
    dataset: StrayDatasetConfig = Field(default_factory=StrayDatasetConfig)

    # Expert Models
    aabb_detseg: GeminiAABBDetSegConfig = GeminiAABBDetSegConfig()
    num_detseg_attemts: int = 3

    # Tools
    tools: Annotated[List[types.Tool], Field(None)]

    # Live API model configuration
    interaction_mode: InteractionMode = InteractionMode.TEXT
    live_model: Literal["gemini-2.0-flash-live-001"] = "gemini-2.0-flash-live-001"
    live_model_config: types.LiveConnectConfig = Field(
        default_factory=lambda: types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
            system_instruction=None,
            tools=None,
        )
    )
    http_options: types.HttpOptions = Field(
        default_factory=lambda: types.HttpOptions(api_version="v1beta")
    )

    # Audio configuration
    format: int = pyaudio.paInt16
    channels: int = 1
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    chunk_size: int = 1024

    @field_validator("tools", mode="before")
    @classmethod
    def make_tools(cls, _, info: ValidationInfo) -> List[types.Tool]:
        tools = []
        aabb_detseg_config = info.data.get("aabb_detseg")
        assert isinstance(aabb_detseg_config, GeminiAABBDetSegConfig)
        tools.append(aabb_detseg_config.make_tool())
        return tools

    @model_validator(mode="after")
    def validate_live_model_config(self) -> "GeminiLiveAgentConfig":
        """Add tools to the live model configuration and set system instruction."""
        self.live_model_config.tools = self.tools

        self.live_model_config.system_instruction = (
            "You are an agentic AI assistant specializing in spatial understanding and navigation assistance for visually impaired users. "
            "You can analyze RGB-D (color + depth) images to detect and describe objects, provide spatial guidance, "
            "and help with navigation tasks. You have access to advanced computer vision tools that can identify "
            "objects, measure distances, and determine spatial relationships in the scene. "
            "Whenever a precise spatial description (including distances and directions) is required, use the provided tools."
        )

        params_to_log = {
            "live_model": self.live_model,
            "interaction_mode": self.interaction_mode,
            "response_modalities": self.live_model_config.response_modalities,
            "tools": [
                {
                    tool.function_declarations[0]
                    .name: tool.function_declarations[0]
                    .description
                }
                for tool in self.tools
            ],
        }
        console = Console.with_prefix(self.__class__.__name__)
        console.log(f"Live model configuration:")
        console.plog(params_to_log)

        return self


class GeminiLiveAgent:
    """Actor-pattern based Gemini Live Agent with single-loop async backend."""

    def __init__(
        self,
        config: GeminiLiveAgentConfig,
        is_rotated: Optional[bool] = None,
        dataset_dir: Optional[Path] = None,
    ):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

        # Initialize the live client
        self.live_client = genai.Client(
            http_options=self.config.http_options,
            api_key=PathConfig().get_api_key("GOOGLE_API_KEY"),
        )

        # Initialize expert models
        self.aabb_detector = self.config.aabb_detseg.setup_target()

        # Setup dataset
        self.update_dataset(dataset_dir=dataset_dir, is_rotated=is_rotated)
        self.update_interaction_mode()

        # Actor communication queues
        self.in_q: Queue[_Cmd] = Queue()
        self.out_q: Queue[_Evt] = Queue()

        # Actor thread and loop
        self.actor_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session: Optional[AsyncSession] = None
        self.current_frame_idx: Optional[int] = None
        self.detection_callback: Optional[Callable[[int, Any], None]] = None

        # Audio setup
        self.audio_in_queue: Optional[asyncio.Queue[bytes]] = None
        self.out_queue: Optional[asyncio.Queue[dict[str, Any]]] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.pya: Optional[pyaudio.PyAudio] = None

        # Remove initial frame context enqueue; set current_frame_idx if dataset non-empty
        if len(self.dataset) > 0:
            self.current_frame_idx = 0

        # Start the actor
        self._start_actor()

    def _start_actor(self):
        """Start the background actor thread."""
        self.actor_thread = threading.Thread(target=self._run_actor, daemon=True)
        self.actor_thread.start()

    def _run_actor(self):
        """Actor main loop - runs in background thread with own async loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_async())
        except Exception as e:
            self.console.error(f"Actor loop error: {e}")
            traceback.print_exc()
        finally:
            self.loop.close()

    async def _main_async(self):
        """Main async method - establishes session and runs consumer/producer."""
        try:
            async with self.live_client.aio.live.connect(
                model=self.config.live_model,
                config=self.config.live_model_config,
            ) as session:
                self.session = session
                # Send initial frame context once session is live
                if self.current_frame_idx is not None:
                    try:
                        await self._send_frame_context(self.current_frame_idx)
                    except Exception as e:
                        self.console.error(f"Failed sending initial frame: {e}")
                self.console.log("Live API session started")

                # Create task group for consumer and producer
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._consumer())
                    tg.create_task(self._producer())

                    # Setup audio tasks if in voice mode
                    if (
                        types.Modality.AUDIO
                        in self.config.live_model_config.response_modalities
                    ):
                        self.audio_in_queue = asyncio.Queue()
                        self.out_queue = asyncio.Queue(maxsize=5)
                        tg.create_task(self._send_realtime())
                        tg.create_task(self._listen_audio())
                        tg.create_task(self._play_audio())

        except Exception as e:
            self.console.error(f"Main async error: {e}")
            self.out_q.put(ErrorEvt(error=str(e)))

    async def _consumer(self):
        """Consumes commands from in_q and executes them."""
        while True:
            try:
                # Non-blocking check for commands
                try:
                    cmd = self.in_q.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    continue

                if isinstance(cmd, AskCmd):
                    await self._handle_ask_cmd(cmd)
                elif isinstance(cmd, AudioCmd):
                    await self._handle_audio_cmd(cmd)
                elif isinstance(cmd, SetFrameCmd):
                    await self._handle_set_frame_cmd(cmd)

            except Exception as e:
                self.console.error(f"Consumer error: {e}")
                self.out_q.put(ErrorEvt(error=str(e)))

    async def _producer(self):
        """Produces events from Live API responses and puts them in out_q."""
        while True:
            try:
                if self.session is None:
                    await asyncio.sleep(0.1)
                    continue

                async for response in self.session.receive():
                    # ───── DEBUG LOG ─────
                    try:
                        self.console.plog(
                            {
                                "text": response.text,
                                "tool_call?": bool(response.tool_call),
                                "data?": bool(response.data),
                                "server_content": (
                                    response.server_content.to_json_dict()
                                    if response.server_content
                                    else None
                                ),
                            }
                        )
                    except Exception:
                        pass
                    # ─────────────────────

                    # Explicitly handle non-text parts to suppress SDK warnings
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Log generated code
                            if part.executable_code is not None:
                                self.console.plog(
                                    {"code": part.executable_code.code},
                                )
                            # Log code execution results
                            if part.code_execution_result is not None:
                                self.console.plog(
                                    {"result": part.code_execution_result.output},
                                )

                    # Handle tool calls
                    if response.tool_call:
                        await self._handle_tool_call(response.tool_call)
                        continue

                    # Immediate text chunks
                    if response.text is not None:
                        self.console.plog(
                            {
                                "text": response.text,
                            }
                        )
                        self.out_q.put(TextEvt(text=response.text))

                    # Audio passthrough
                    if response.data is not None and self.audio_in_queue:
                        await self.audio_in_queue.put(response.data)

            except Exception as e:
                self.console.error(f"Producer error: {e}")
                self.out_q.put(ErrorEvt(error=str(e)))
                await asyncio.sleep(0.1)

    async def _handle_ask_cmd(self, cmd: AskCmd):
        """Handle text question command."""
        if self.session is None:
            return

        # Set frame if provided
        if cmd.frame_idx is not None:
            await self._send_frame_context(cmd.frame_idx)

        # Send text query as completed turn
        self.console.log(f"Sending user input to Live API: '{cmd.text}'")
        await self.session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=cmd.text)]),
            turn_complete=True,
        )

    async def _handle_audio_cmd(self, cmd: AudioCmd):
        """Handle audio data command."""
        if self.session is None or self.out_queue is None:
            return

        await self.out_queue.put({"data": cmd.pcm_data, "mime_type": "audio/pcm"})

    async def _handle_set_frame_cmd(self, cmd: SetFrameCmd):
        """Handle set frame command."""
        if self.session is not None and cmd.frame_idx != self.current_frame_idx:
            self.current_frame_idx = cmd.frame_idx
            await self._send_frame_context(cmd.frame_idx)

    async def _send_frame_context(self, frame_idx: int) -> None:
        """Send RGB image context to Live API as a completed turn."""
        if self.session is None:
            return
        try:
            frame = self.dataset[frame_idx]
            # This is an error in the genai SDK!
            # buf = io.BytesIO()
            # frame.rgb_image.save(buf, format="JPEG")
            # # blob = types.Blob(data=buf.getvalue(), mime_type="image/jpeg")
            # self.session.send_realtime_input()
            # await self.session.send_client_content(
            #     turns=types.Content(
            #         role="user",
            #         parts=[
            #             # self.dataset[frame_idx].rgb_image,
            #             # types.Part(inline_data=blob),
            #             types.Part(
            #                 inline_data=types.Blob(
            #                     data=base64.b64encode(buf.getvalue()).decode("utf-8"),
            #                     mime_type="image/jpeg",
            #                 ),
            #             ),
            #             types.Part.from_text(
            #                 text=(
            #                     f"I'm now looking at frame {frame_idx} from the dataset. "
            #                     "I'm seeing what the visually impaired user is seeing."
            #                 )
            #             ),
            #         ],
            #     ),
            #     turn_complete=True,
            # )
            await self.session.send_realtime_input(
                media=frame.rgb_image,
            )
            await self.session.send_realtime_input(
                text=f"[SYSTEM] I'm now looking at frame {frame_idx} from the dataset. "
                "I'm seeing what the visually impaired user is seeing."
            )

            self.console.log(f"Sent frame {frame_idx} context to Live API")
        except Exception as e:
            self.console.error(f"Error sending frame context: {e}")
            raise e

    async def _handle_tool_call(self, tool_call: types.LiveServerToolCall):
        """Handle tool calls from the model."""
        self.console.log(f"[TOOL CALL] Received function call")
        self.console.plog(tool_call)

        for fc in tool_call.function_calls:
            if fc.name == "run_aabb_detection":
                user_prompt = fc.args.get("user_prompt", "")
                subset_mode = fc.args.get("subset_mode", False)

                # Get current frame data
                if self.current_frame_idx is None:
                    response_payload = {
                        "error": "No current frame set. Please select a frame first."
                    }
                else:
                    frame = self.dataset[self.current_frame_idx]
                    # Attempt detection up to N times
                    succ = False
                    last_exc = None
                    for _ in range(self.config.num_detseg_attemts):
                        try:
                            detections = await asyncio.to_thread(
                                self.aabb_detector.run_aabb_detection,
                                frame,
                                user_prompt,
                                subset_mode=subset_mode,
                            )
                            succ = True
                            break
                        except Exception as e:
                            last_exc = e
                            self.console.error(f"AABB detection attempt failed: {e}")

                    # Build a single payload
                    if succ:
                        # Emit to UI thread
                        self.out_q.put(
                            DetectionsEvt(
                                detections=detections,
                                frame_idx=self.current_frame_idx,
                            )
                        )
                        response_payload = {"detections": detections.to_list_dict()}
                        self.console.log(
                            f"Found {len(detections.objects)} detections for frame {self.current_frame_idx}"
                        )
                    else:
                        response_payload = {
                            "error": f"run_aabb_detection failed after {self.config.num_detseg_attemts} attempts: {last_exc}"
                        }

                # Send exactly one FunctionResponse and exit
                await self.session.send_tool_response(
                    function_responses=[
                        types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response=response_payload,
                        )
                    ]
                )
                return

    # Audio handling methods (for voice mode)
    async def _send_realtime(self):
        """Sends audio data from the queue to the API."""
        while True:
            if self.out_queue is None:
                break
            msg = await self.out_queue.get()
            if self.session:
                await self.session.send_realtime_input(**msg)

    async def _listen_audio(self):
        """Captures audio from microphone and puts it in the queue."""
        if self.pya is None or self.out_queue is None:
            return

        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.send_sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.config.chunk_size,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(
                self.audio_stream.read, self.config.chunk_size, **kwargs
            )
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def _play_audio(self):
        """Plays the received audio."""
        if self.pya is None or self.audio_in_queue is None:
            return

        stream = await asyncio.to_thread(
            self.pya.open,
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.receive_sample_rate,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
            # Emit audio event for Streamlit
            self.out_q.put(AudioEvt(audio_data=bytestream))

    # Public API methods (thread-safe, callable from Streamlit)
    def ask(self, text: str, frame_idx: Optional[int] = None):
        """Ask a text question (thread-safe)."""
        # Flush any pending events so new query isn't mixed with old responses
        try:
            while True:
                self.out_q.get_nowait()
        except Empty:
            pass
        # Enqueue ask command
        self.in_q.put(AskCmd(text=text, frame_idx=frame_idx))

    def push_audio(self, pcm_data: bytes):
        """Push audio data (thread-safe)."""
        self.in_q.put(AudioCmd(pcm_data=pcm_data))

    def set_current_frame(self, frame_idx: int):
        """Set current frame (thread-safe)."""
        self.in_q.put(SetFrameCmd(frame_idx=frame_idx))

    def next_event(self) -> Optional[_Evt]:
        """Get next event (thread-safe, non-blocking)."""
        try:
            return self.out_q.get_nowait()
        except Empty:
            return None

    def set_detection_callback(self, callback: Callable[[int, Any], None]):
        """Set detection result callback."""
        self.detection_callback = callback

    # Setup methods (keep existing implementation)
    def update_dataset(self, dataset_dir: Optional[Path], is_rotated: Optional[bool]):
        if isinstance(is_rotated, bool):
            self.config.dataset.is_rotated = is_rotated

        if dataset_dir is not None:
            self.config.dataset.data_parser.paths.dataset_dir = dataset_dir

        self.dataset = self.config.dataset.setup_target()

    def update_interaction_mode(
        self, interaction_mode: Optional[InteractionMode] = None
    ):
        """Setup the live interaction mode."""
        interaction_mode = interaction_mode or self.config.interaction_mode

        if interaction_mode == InteractionMode.VOICE:
            import pyaudio

            self.pya = pyaudio.PyAudio()
            # Set audio modality
            self.config.live_model_config.response_modalities = [types.Modality.AUDIO]
        else:
            self.pya = None
            # Set text modality
            self.config.live_model_config.response_modalities = [types.Modality.TEXT]

    def update_aabb_detector(self, model_name: str = "gemini-2.5-flash-preview-05-20"):
        """Create an AABB detector with the specified model name."""
        self.config.aabb_detseg.model_name = model_name
        self.aabb_detector = self.config.aabb_detseg.setup_target()
