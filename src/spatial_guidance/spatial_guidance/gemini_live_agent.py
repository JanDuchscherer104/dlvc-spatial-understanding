import asyncio
import sys
import traceback
from enum import Enum, auto
from pathlib import Path
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


class GeminiLiveAgentConfig(BaseConfig["GeminiLiveAgent"]):
    target: Type["GeminiLiveAgent"] = Field(default_factory=lambda: GeminiLiveAgent)

    # Dataset configuration
    dataset: StrayDatasetConfig = Field(default_factory=StrayDatasetConfig)

    # Expert Models
    aabb_detseg: GeminiAABBDetSegConfig = GeminiAABBDetSegConfig()

    # Tools
    # TODO: Init remaining tools and fiels of each tool!
    tools: Annotated[List[types.Tool], Field(None)]

    # Live API model configuration
    interaction_mode: InteractionMode = InteractionMode.TEXT
    live_model: Literal["gemini-2.0-flash-live-001"] = "gemini-2.0-flash-live-001"
    live_model_config: types.LiveConnectConfig = Field(
        default_factory=lambda: types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=None,  # TODO
            tools=None,  # Will be set in `validate_live_model_config` method
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
        # TODO: Add all tools to the live model configuration
        tools = []

        aabb_detseg_config = info.data.get("aabb_detseg")
        assert isinstance(aabb_detseg_config, GeminiAABBDetSegConfig)

        tools.append(aabb_detseg_config.make_tool())

        return tools

    @model_validator(mode="after")
    def validate_live_model_config(self) -> "GeminiLiveAgentConfig":
        """Add tools to the live model configuration and set system instruction."""
        self.live_model_config.tools = self.tools

        # # Set response modalities based on interaction mode
        # if self.interaction_mode == InteractionMode.VOICE:
        #     self.live_model_config.response_modalities = [
        #         types.Modality.TEXT,
        #         types.Modality.AUDIO,
        #     ]
        # else:
        #     self.live_model_config.response_modalities = [types.Modality.TEXT]

        self.live_model_config.system_instruction = (
            "You are an agentic AI assistant specializing in spatial understanding and navigation assistance for visually impaired users. "
            "You can analyze RGB-D (color + depth) images to detect and describe objects, provide spatial guidance, "
            "and help with navigation tasks. You have access to advanced computer vision tools that can identify "
            "objects, measure distances, and determine spatial relationships in the scene."
            "Whenever a precise spatial description (including distances and directions) is required, use the provided tools."
            "If a "
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
        console.log(f"Live model configuration: ")
        console.plog(params_to_log)

        return self


class GeminiLiveAgent:
    # Type hints for all attributes
    config: GeminiLiveAgentConfig
    dataset: StrayDataset
    live_client: genai.Client
    aabb_detector: GeminiAABBDetSeg

    # Session attributes (always present)
    session: Optional[AsyncSession]
    current_frame_idx: Optional[int]

    # Callback for storing detection results (for Streamlit integration)
    detection_callback: Optional[Callable[[int, AABBDetections], None]]

    # Audio-specific attributes (only for VOICE mode)
    audio_in_queue: Optional[asyncio.Queue[bytes]]
    out_queue: Optional[asyncio.Queue[dict[str, Any]]]
    audio_stream: Optional[pyaudio.Stream]
    pya: Optional[pyaudio.PyAudio]

    def __init__(
        self,
        config: GeminiLiveAgentConfig,
        is_rotated: Optional[bool] = None,
        dataset_dir: Optional[Path] = None,
    ):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

        # Event‑loop that owns the Live‑API session
        self.session_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize the live client
        self.live_client = genai.Client(
            http_options=self.config.http_options,
            api_key=PathConfig().get_api_key("GOOGLE_API_KEY"),
        )

        # Initialize session (always present)
        self.session = None
        self.current_frame_idx = None
        self.detection_callback = None

        # Initialize expert models
        self.aabb_detector = self.config.aabb_detseg.setup_target()

        self.setup_dataset(
            dataset_dir=dataset_dir,
            is_rotated=is_rotated,
        )
        self.setup_interaction_mode()

    async def run(self):
        """Run the Gemini Live Agent in an asynchronous loop.

        ```python
        asyncio.run(GeminiLiveAgentConfig.setup_target().run())
        ```
        """
        # Check if we have AUDIO modality for voice interaction
        if types.Modality.AUDIO in self.config.live_model_config.response_modalities:
            await self._run_voice_interaction()
        else:
            await self._run_text_interaction()

    async def _run_voice_interaction(self):
        """Main loop that starts and manages all asynchronous voice tasks."""
        try:
            async with self.live_client.aio.live.connect(
                model=self.config.live_model, config=self.config.live_model_config
            ) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                async with asyncio.TaskGroup() as tg:
                    send_text_task = tg.create_task(self._send_text())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())

                    await send_text_task
                    raise asyncio.CancelledError("User has exited the program.")

        except asyncio.CancelledError:
            pass
        except Exception as eg:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exc()
        finally:
            if self.pya:
                self.pya.terminate()

    async def _send_text(self):
        """Enables sending text messages to the model."""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            # print(f"[INFO] Sending text to API: '{text}'")
            self.console.log(f"Sending text to API: '{text}'")
            if self.session:

                await self.session.send(input=text or ".", end_of_turn=True)

    async def _send_realtime(self):
        """Sends audio data from the queue to the API."""
        while True:
            if self.out_queue is None:
                break
            msg = await self.out_queue.get()
            if self.session:
                await self.session.send(input=msg)

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

    async def _receive_audio(self):
        """Receives responses (audio, text, function calls) from the API."""
        while True:
            if self.session is None:
                break
            turn = self.session.receive()
            async for response in turn:
                if tool_call := response.tool_call:
                    await self.handle_tool_call(tool_call)
                    continue
                if data := response.data:
                    if self.audio_in_queue:
                        self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(f"\n[MODEL OUTPUT] {text}", end="")

            if self.audio_in_queue:
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

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

    async def handle_model_responses_streamlit(self) -> str:
        """Receives and processes responses (text, function calls) from the API for Streamlit integration."""
        if self.session is None:
            return "No active session"

        response_text = ""
        turn = self.session.receive()
        async for response in turn:
            if tool_call := response.tool_call:
                await self.handle_tool_call(tool_call)
                continue
            if text := response.text:
                response_text += text

        return response_text

    async def start_live_session(self) -> bool:
        """Start a Live API session for Streamlit integration."""
        if self.session is not None:
            return True  # Already active

        try:
            # Use manual context management with proper exit pairing
            self._session_context = self.live_client.aio.live.connect(
                model=self.config.live_model,
                config=self.config.live_model_config,
            )
            self.session = await self._session_context.__aenter__()
            # Remember which loop the session belongs to
            self.session_loop = asyncio.get_running_loop()

            self.console.log("Live API session started for Streamlit")
            return True
        except Exception as e:
            self.console.error(f"Failed to start Live API session: {e}")
            self.session = None
            return False

    async def stop_live_session(self):
        """Stop the Live API session."""
        if self.session and hasattr(self, "_session_context"):
            try:
                await self._session_context.__aexit__(None, None, None)
                self.console.log("Live API session stopped")
            except Exception as e:
                self.console.error(f"Error stopping Live API session: {e}")
            finally:
                self.session = None
                self._session_context = None

    async def _handle_streamlit_query_async(
        self, user_input: str, frame_idx: Optional[int] = None
    ) -> str:
        """Handle a single query from Streamlit chat interface."""
        if not user_input.strip():
            return "Please provide a valid input."

        # Ensure session is active
        if self.session is None:
            session_started = await self.start_live_session()
            if not session_started:
                return "Failed to establish connection with Live API."

        assert isinstance(self.session, AsyncSession)
        try:
            # Set frame context if provided
            # if frame_idx is not None:
            # await self.set_current_frame(frame_idx)
            await self.send_frame_context(frame_idx)

            # Send user input to Live API
            self.console.log(
                f"Sending user input to Live API: {user_input}, type: {type(user_input)}"
            )
            await self.session.send_realtime_input(text=user_input)

            # Get and return response
            response_text = await self.handle_model_responses_streamlit()

            return response_text or "No response received from the model."

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.console.error(error_msg)
            return error_msg

    def handle_streamlit_query(
        self, user_input: str, frame_idx: Optional[int] = None
    ) -> str:
        """
        Thread‑safe wrapper that schedules the actual async coroutine on the
        event‑loop where the Live‑API session was created, avoiding cross‑loop
        “Future attached to a different loop” errors.
        """
        # If no session loop yet (first call) fall back to running the coroutine directly
        if self.session_loop is None or not self.session_loop.is_running():
            return asyncio.run(
                self._handle_streamlit_query_async(user_input, frame_idx)
            )

        # Always execute the coroutine on the original session loop
        fut = asyncio.run_coroutine_threadsafe(
            self._handle_streamlit_query_async(user_input, frame_idx),
            self.session_loop,
        )
        return fut.result()

    def setup_dataset(self, dataset_dir: Optional[Path], is_rotated: Optional[bool]):
        if isinstance(is_rotated, bool):
            self.config.dataset.is_rotated = is_rotated

        if dataset_dir is not None:
            assert isinstance(dataset_dir, Path), "dataset_dir must be a Path object"
            assert (
                dataset_dir.is_dir()
            ), f"Dataset directory {dataset_dir} does not exist or is not a directory"
            self.config.dataset.data_parser.paths.dataset_dir = dataset_dir

        self.dataset = self.config.dataset.setup_target()

    def setup_aabb_detector(self, model_name: str = "gemini-2.5-flash-preview-05-20"):
        """Create an AABB detector with the specified model name."""
        self.config.aabb_detseg.model_name = model_name
        self.aabb_detector = self.config.aabb_detseg.setup_target()

    def setup_interaction_mode(
        self, interaction_mode: Optional[InteractionMode] = None
    ):
        """Setup the live interaction mode."""
        interaction_mode = interaction_mode or self.config.interaction_mode

        if interaction_mode == InteractionMode.VOICE:
            self.audio_in_queue = None
            self.out_queue = None
            self.audio_stream = None
            self.pya = pyaudio.PyAudio()
        else:
            self.audio_in_queue = None
            self.out_queue = None
            self.audio_stream = None
            self.pya = None

    async def set_current_frame(self, frame_idx: int) -> bool:
        """Set the current frame index and send visual context to Live API if session is active."""
        # if frame_idx == self.current_frame_idx:
        #     return False  # No change needed

        self.current_frame_idx = frame_idx

        # If we have an active session, send the frame context automatically
        if self.session is not None:
            await self.send_frame_context(frame_idx)
            return True

        return False

    async def send_frame_context(self, frame_idx: Optional[int] = None) -> None:
        """Send RGB image and brief context description to Live API."""
        if self.session is None:
            Console.with_prefix("GeminiLiveAgent").warn(
                "No active session to send frame context"
            )
            return

        frame_idx = frame_idx or self.current_frame_idx
        if frame_idx is None:
            Console.with_prefix("GeminiLiveAgent").warn("No frame index set")
            return

        try:
            context_text = (
                f"I'm now looking at frame {frame_idx} from the dataset. "
                "I'm seeing what the visually imparied user is seeing. "
            )

            await self.session.send_realtime_input(
                media=self.dataset[frame_idx].rgb_image
            )
            await self.session.send_realtime_input(text=context_text)

            Console.with_prefix("GeminiLiveAgent").log(
                f"Sent frame {frame_idx} context to Live API"
            )

        except Exception as e:
            Console.with_prefix("GeminiLiveAgent").error(
                f"Failed to send frame context: {e}"
            )

    def get_current_frame_data(self):
        """Get the current frame data for tool calls."""
        if self.current_frame_idx is None:
            Console.with_prefix("GeminiLiveAgent").warn("No current frame index set")
            return None
        return self.dataset[self.current_frame_idx]

    async def handle_tool_call(self, tool_call: types.LiveServerToolCall):
        """Processes function calls from the model."""
        self.console.log(f"[TOOL CALL] Received function call")
        self.console.plog(tool_call)
        function_responses = []
        for fc in tool_call.function_calls:
            result = None

            # Handle AABB detection tool call
            if fc.name == "run_aabb_detection":
                try:
                    user_prompt = fc.args.get("user_prompt", "")
                    subset_mode = fc.args.get("subset_mode", False)

                    # Get current frame data
                    frame = self.get_current_frame_data()
                    if frame is None:
                        result = {
                            "error": "No current frame set. Please select a frame first."
                        }
                    else:
                        # Run AABB detection
                        detections = await asyncio.to_thread(
                            self.aabb_detector.run_aabb_detection,
                            frame,
                            user_prompt,
                            subset_mode=subset_mode,
                        )

                        # Store detection results via callback if available
                        if (
                            self.detection_callback
                            and self.current_frame_idx is not None
                        ):
                            self.detection_callback(self.current_frame_idx, detections)

                        # Convert to JSON format
                        result = detections.to_json_list()
                        print(
                            f"[AABB DETECTION] Found {len(detections.objects)} detections for frame {self.current_frame_idx}"
                        )

                except Exception as e:
                    print(f"[ERROR] AABB detection failed: {e}")
                    result = {"error": str(e)}

            if result:
                function_responses.append(
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"result": result},
                    )
                )

        if function_responses and self.session:
            await self.session.send_tool_response(function_responses=function_responses)
