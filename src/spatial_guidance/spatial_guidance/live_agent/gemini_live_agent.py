import asyncio
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pyaudio
from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from spatial_guidance.data_contracts.aabb_segmentation import (
    compute_rotation_from_3d_position,
)
from spatial_guidance.data_contracts.dataset import DatasetOut

from ..data_contracts.aabb_segmentation import AABBDetection, AABBDetections
from ..utils import Console, PathConfig
from .actor_protocols import (
    AskCmd,
    AudioCmd,
    AudioEvt,
    DetectionsEvt,
    ErrorEvt,
    SetFrameCmd,
    TextEvt,
    _Cmd,
    _Evt,
)
from .live_agent_enums import (
    DirectionalStyle,
    DistanceStyle,
    GenState,
    InteractionMode,
    ResponseStyle,
)

if TYPE_CHECKING:
    from .live_agent_config import GeminiLiveAgentConfig

# TODO: Both of the styles should be combined into a single style and then injected into the system instructions.
# TODO: The system instrucitons should be updated to encourage the model to use past detection results whenever possible
# TODO: The model should rather use the Tool rather than code execution for simpel queries like "How is the scooter positioned relative to me?"
# TODO: Add example Q&A pairs to the system prompt. Eg. Where is the bicycle? should be treated equivalently to a more specic query whose anser format should depend on the DirectionalStyle and DistanceStyle.
# TODO: The agent should always provide a reponse when it receives a new frame where it sees potential hazards. If there are no immediate hazards, it should not say anything
# TODO: Modify the system prompt such that it includes quantitaive inforamtion (relative distnaces / positions) of obstacles along the path.
# TODO: Voice2Voice mode (best w/ transcription -> chat)
# TODO: modify system prompts to be customizable with the enums
# TODO: provide model with metadata (eg. overview of previous detections, if no available, has to run the tool)
# TODO: modify system prompts to encourage tool usage over code execution


class GeminiLiveAgent:
    """Actor-pattern based Gemini Live Agent with single-loop async backend."""

    def __init__(
        self,
        config: "GeminiLiveAgentConfig",
        is_rotated: Optional[bool] = None,
        dataset_dir: Optional[Path] = None,
    ):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_debug(
            self.config.is_debug
        )

        # Initialize the live client
        self.live_client = genai.Client(
            http_options=self.config.http_options,
            api_key=PathConfig().get_api_key("GOOGLE_API_KEY"),
        )

        # Initialize expert models
        self.aabb_detector = self.config.gemini_aabb_detseg.setup_target()

        # Setup dataset
        self.update_dataset(dataset_dir=dataset_dir, is_rotated=is_rotated)
        self.last_frame_idx: Optional[int] = None
        self.current_frame_idx: int = 0
        self.current_frame: Optional[DatasetOut] = None

        self.update_interaction_mode()

        # Actor communication queues
        self.in_q: Queue[_Cmd] = Queue()
        self.out_q: Queue[_Evt] = Queue()

        # Actor thread and loop
        self.actor_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session: Optional[AsyncSession] = None

        # Audio setup
        self.audio_in_queue: Optional[asyncio.Queue[bytes]] = None
        self.out_queue: Optional[asyncio.Queue[dict[str, Any]]] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.pya: Optional[pyaudio.PyAudio] = None

        # Detection cache: maps (frame_idx, detection_mode, user_prompt) to AABBDetections
        self._detection_cache: dict[tuple[int, Optional[str], str], AABBDetections] = {}

        # Detection cache: maps (frame_idx, label) -> detection_json dict
        self._last_detections: dict[tuple[int, str], dict] = {}

        # Buffer that accumulates streamed text chunks for the current assistant
        # turn.  Flushed when we receive `generation_complete`/`turn_complete`
        # from the server, so only the *final* answer is emitted to the UI.
        self._text_buffer: list[str] = []
        # State for text collection: only emit final answer after tools
        self._phase = GenState.COLLECT_FIRST

        # Track pending queries for response time calculation
        self._pending_queries: dict[str, float] = {}

        # Start the actor
        self._start_actor()

    def update_response_style(
        self, dir_style: DirectionalStyle, dist_style: DistanceStyle
    ):
        """Update combined response style, regenerate system instruction, and restart session."""
        new_style = ResponseStyle(dir_style, dist_style)
        if new_style == self.config.response_style:
            return
        self.config.live_model_config.system_instruction = (
            self.config.system_instruction_template.make_prompt(
                response_style=new_style
            )
        )
        self.console.log(
            f"Updated system instruction with new response style: {new_style}."
        )
        self._restart_live_session()

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
            self.console.error(e, "Actor loop error")
            traceback.print_exc()
        finally:
            self.loop.close()

    def _restart_live_session(self):
        """Restart the Live API session to apply updated system instruction."""
        self.console.log("Restarting Live API session with updated 'live_model_config'")
        # Close existing session if active
        if self.session is not None:
            try:
                self.session.close()
            except Exception as e:
                self.console.error(e, "Error closing session")
            self.session = None
        # Stop current event loop to exit actor
        if self.loop is not None and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception as e:
                self.console.error(e, "Error stopping event loop")
        # Start a new actor thread to reconnect
        self._start_actor()

    async def _main_async(self):
        """Main async method - establishes session and runs consumer/producer."""
        try:
            async with self.live_client.aio.live.connect(
                model=self.config.live_model,
                config=self.config.live_model_config,
            ) as session:
                self.session = session
                # Send initial frame context once session is live
                if (
                    self.last_frame_idx is None
                    or self.last_frame_idx != self.current_frame_idx
                ):
                    try:
                        await self._send_frame_context(self.current_frame_idx)
                    except Exception as e:
                        self.console.error(e, "Failed sending initial frame")
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
            self.console.error(e, "Main async error")
            self.out_q.put(ErrorEvt(error=str(e)))

    async def _consumer(self):
        """Consumes commands from in_q and executes them."""
        while True:
            try:
                # Block in thread until a command is available
                cmd = await asyncio.to_thread(self.in_q.get)

                if isinstance(cmd, SetFrameCmd):
                    await self._handle_set_frame_cmd(cmd)
                elif isinstance(cmd, AskCmd):
                    self._pending_queries[cmd.query_id] = cmd.timestamp
                    await self._handle_ask_cmd(cmd)
                elif isinstance(cmd, AudioCmd):
                    await self._handle_audio_cmd(cmd)

            except Exception as e:
                self.console.error(e, "Consumer error")
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
                    if self.config.is_debug:
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
                                },
                                title="Live API Response",
                            )
                        except Exception:
                            pass
                    # ─────────────────────
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Log code execution results (if present from server - should not happen with disabled code execution)
                            if part.code_execution_result is not None:
                                self.console.plog(
                                    {"result": part.code_execution_result.output},
                                    title="Code Execution Result",
                                )
                                # Do not surface internal execution output to the chat UI
                                self.console.plog(
                                    {
                                        "debug_execution_result": part.code_execution_result.output
                                    },
                                    title="Debug Execution Output",
                                )

                    # ── State‐machine buffering ──
                    if response.tool_call:
                        # drop any pre‐tool chatter and wait for tool
                        self._phase = GenState.WAIT_TOOL
                        self._text_buffer.clear()
                        await self._handle_tool_call(response.tool_call)
                        # now collect only the final answer
                        self._phase = GenState.COLLECT_FINAL
                        continue

                    # buffer text only in the appropriate phase
                    if response.text is not None and self._phase in (
                        GenState.COLLECT_FIRST,
                        GenState.COLLECT_FINAL,
                    ):
                        self._text_buffer.append(response.text)

                    # flush buffer on end‐of‐turn
                    if response.server_content and (
                        getattr(response.server_content, "generation_complete", False)
                        or getattr(response.server_content, "turn_complete", False)
                    ):
                        if (
                            self._phase
                            in (GenState.COLLECT_FIRST, GenState.COLLECT_FINAL)
                            and self._text_buffer
                        ):
                            final_text = "".join(self._text_buffer).strip()
                            if final_text:
                                # Calculate response time for the most recent query
                                query_id = None
                                response_time_ms = None
                                if self._pending_queries:
                                    # Get the most recent query (latest timestamp)
                                    query_id = max(
                                        self._pending_queries.keys(),
                                        key=lambda k: self._pending_queries[k],
                                    )
                                    start_time = self._pending_queries.pop(query_id)
                                    response_time_ms = (time.time() - start_time) * 1000

                                self.out_q.put(
                                    TextEvt(
                                        text=final_text,
                                        query_id=query_id,
                                        response_time_ms=response_time_ms,
                                    )
                                )
                        self._text_buffer.clear()
                        # reset to initial for next user turn
                        self._phase = GenState.COLLECT_FIRST

                    # Audio passthrough
                    if response.data is not None and self.audio_in_queue:
                        await self.audio_in_queue.put(response.data)

            except Exception as e:
                self.console.error(e, "Producer error")
                self.out_q.put(ErrorEvt(error=str(e)))
                await asyncio.sleep(0.1)

    async def _handle_ask_cmd(self, cmd: AskCmd):
        """Handle text question command."""
        if self.session is None:
            return

        # Set frame if provided
        if cmd.frame_idx is not None and cmd.frame_idx != self.current_frame_idx:
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
            await self._send_frame_context(cmd.frame_idx)

    async def _send_frame_context(self, frame_idx: int) -> None:
        """Send RGB image context to Live API as a completed turn."""
        if self.session is None:
            return
        try:
            frame = self.dataset[frame_idx]
            # TODO: Use frame.camera_pose to provide spatial context on how we moved relative to the last frame
            self.console.plog(frame.ground_plane, title="Frame Ground Plane")

            await self.session.send_realtime_input(
                media=frame.rgb_image,
            )
            await asyncio.sleep(0.1)

            frame_change_info = f"<SYSTEM>[NEW FRAME] Updated frame_idx to '{frame_idx}'. You are seeing what the user sees.</SYSTEM>"
            if self.current_frame is not None:
                try:
                    # Describe the relative movement from the last frame to the current one
                    rel_move = self.current_frame.rel_move_description(
                        other=frame,
                        idx_self=self.current_frame_idx,
                        idx_other=frame_idx,
                    )
                except Exception as e:
                    self.console.error(
                        e, "Error describing relative movement between frames"
                    )
                else:
                    frame_change_info = (
                        frame_change_info.replace("</SYSTEM>", "") + rel_move
                    )

            self.console.log(frame_change_info)
            await self.session.send_realtime_input(text=frame_change_info)
            if self._last_detections:
                await self.session.send_realtime_input(
                    text=f"<System>Cached detections: {self._last_detections.keys()} </System>"
                )
            # rovide a concise description of all relevant objects in the scene to aquire an improved understanding of the environment. This will improve your capacities to answer potential user queries about the scene."
            #  If you see any potential hazards, call 'warn_user' TODO

            # self._phase = ...
            # TODO: The model should reply with a response if and only if it sees any potential hazards in the frame and warn the user about them!
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            self.last_frame_idx = frame_idx

            self.console.log(f"Sent frame {frame_idx} context to Live API")
        except Exception as e:
            self.console.error(e, "Error sending frame context")
            raise e

    async def _handle_tool_call(
        self, tool_call: types.LiveServerToolCall, return_no_send: bool = False
    ) -> Optional[types.FunctionResponse]:
        """Handle tool calls from the model."""
        self.console.log(f"[TOOL CALL] Received function call")
        self.console.plog(tool_call, title="Tool Call Details")

        fc_responses: list[types.FunctionResponse] = []
        for fc in tool_call.function_calls:
            if fc.name == "get_last_detections":
                frame_index = fc.args["frame_idx"]
                labels = fc.args["labels"]
                # Prepare poses for transform
                pose_prev = self.dataset[frame_index].camera_pose
                pose_curr = (
                    self.current_frame.camera_pose if self.current_frame else None
                )

                if self.config.is_debug:
                    self.console.log(
                        f"Transform debug: frame_index={frame_index}, current_frame_idx={self.current_frame_idx}"
                    )
                    self.console.log(
                        f"Pose prev shape: {pose_prev.shape if pose_prev is not None else None}"
                    )
                    self.console.log(
                        f"Pose curr shape: {pose_curr.shape if pose_curr is not None else None}"
                    )

                det_list = []
                for lbl in labels:
                    key = (frame_index, lbl)
                    if key not in self._last_detections:
                        continue
                    raw_det = self._last_detections[key]

                    # Only transform if we have both poses and they're different frames
                    if frame_index != self.current_frame_idx:
                        det_tr = AABBDetections.transform_cached_det(
                            raw_det,
                            pose_from=pose_prev,
                            pose_to=pose_curr,
                            console=self.console,
                        )

                        if self.config.is_debug:
                            self.console.log(
                                f"Transformed detection {lbl}: {raw_det.get('center_point_3d')} -> {det_tr.get('center_point_3d')}"
                            )
                    else:
                        det_tr = raw_det.copy()  # No transformation needed

                    if det_tr is not None:
                        det_list.append(det_tr)
                    else:
                        self.console.warn(
                            f"Failed to transform detection for label '{lbl}' in frame {frame_index}"
                        )
                self.console.plog(det_list, title="Cached detections (transformed)")
                fc_responses.append(
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"detections": det_list},
                    )
                )

            elif fc.name == "run_aabb_detection":
                # Check if frame is set
                if self.current_frame_idx is None:
                    fc_responses.append(
                        types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response={"detections": {}},
                        )
                    )
                    error_msg = "No current frame set. Please select a frame first."
                    self.console.error(RuntimeError(error_msg), error_msg)
                    continue

                # Get current frame data
                user_prompt = fc.args.get("user_prompt", "")
                detection_mode = fc.args.get("detection_mode", None)
                cache_key = (self.current_frame_idx, detection_mode, user_prompt)
                frame = self.dataset[self.current_frame_idx]
                # reuse cached detections if identical request
                if cache_key in self._detection_cache:
                    detections = self._detection_cache[cache_key]
                    succ = True
                else:
                    succ = False
                    last_exc = None
                    # Attempt detection up to N times
                    for _ in range(self.config.num_detseg_attempts):
                        try:
                            detections = await asyncio.to_thread(
                                self.aabb_detector.run_aabb_detection,
                                frame,
                                user_prompt,
                                detection_mode=detection_mode,
                            )
                            if detections is None or len(detections) == 0:
                                self.console.error(
                                    RuntimeError("AABB detection returned no results")
                                )
                                continue
                            succ = True
                            break
                        except Exception as e:
                            last_exc = e
                            self.console.error(e, "AABB detection attempt failed")
                            self.console.plog(
                                detections, title="Detection Attempt Failed"
                            )
                    if succ:
                        # cache the results
                        self._detection_cache[cache_key] = detections

                # Build a single payload
                if succ:
                    # remember the latest detections for code-execution snippets
                    # Ensure detections is AABBDetections type
                    assert isinstance(
                        detections, AABBDetections
                    ), f"Expected AABBDetections, got {type(detections)}"

                    # Use the to_dict() method to get dictionary for tool response
                    detections_dict = detections.to_dict()
                    # detections_dict["frame_idx"] = self.current_frame_idx
                    # Add frame index to each detection
                    for det in detections_dict.values():
                        det["frame_idx"] = self.current_frame_idx

                    # Return dictionary format as requested by user
                    response_payload = {"detections": list(detections_dict.values())}

                    # cache each detection under (frame_idx, label)
                    if self.current_frame_idx is not None:
                        for lbl, det_json in detections_dict.items():
                            self._last_detections[(self.current_frame_idx, lbl)] = (
                                det_json
                            )
                        self.console.plog(
                            self._last_detections.keys(), title="Last Detections Cache"
                        )

                    # Only emit to UI thread if this is a real tool call (not from code execution)
                    if not return_no_send and self.current_frame_idx is not None:
                        self.out_q.put(
                            DetectionsEvt(
                                detections=detections,
                                frame_idx=self.current_frame_idx,
                            )
                        )
                else:
                    # For error cases, return empty detections dictionary
                    response_payload = {"detections": {}}
                    if self.config.is_debug:
                        error_msg = f"run_aabb_detection failed after {self.config.num_detseg_attempts} attempts"
                        self.console.error(last_exc, error_msg)

                # Send exactly one FunctionResponse and exit
                fc_responses.append(
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response=response_payload,
                    )
                )

        if return_no_send:
            return fc_responses[0]
        else:
            # Send the function response back to the model
            if self.session:
                try:
                    await self.session.send_tool_response(
                        function_responses=fc_responses
                    )
                except Exception as e:
                    self.console.error(e, "Failed to send tool response")
                    self.console.plog(fc_responses, title="Tool Responses")
            return None

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
        self.config.gemini_aabb_detseg.model_name = model_name
        self.aabb_detector = self.config.gemini_aabb_detseg.setup_target()
