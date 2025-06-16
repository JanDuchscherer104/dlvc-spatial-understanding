import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import nest_asyncio
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import io

# Enable nested asyncio for Streamlit compatibility
nest_asyncio.apply()

# WebRTC imports
try:
    import av
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning(
        "streamlit-webrtc not available. Install it for voice mode: pip install streamlit-webrtc"
    )
from spatial_guidance.visualization.detection_visualizer import DetectionVisualizer
from spatial_guidance.data_contracts import AABBDetections
from spatial_guidance.live_agent import (
    MODEL_OPTIONS,
    AudioEvt,
    DetectionsEvt,
    DirectionalStyle,
    DistanceStyle,
    ErrorEvt,
    GeminiLiveAgent,
    GeminiLiveAgentConfig,
    InteractionMode,
    OperationalMode,
    TextEvt,
)
from spatial_guidance.utils import Console, PathConfig

CONSOLE = Console.with_prefix("streamlit_app_live")

# Global containers for thread-safe communication
live_agent_lock = threading.Lock()
live_agent_container = {"agent": None, "response": None, "detections": None}


# Configuration and Setup
@st.cache_resource(max_entries=1)
def get_live_agent() -> GeminiLiveAgent:
    """Create and cache the live agent with default configuration."""
    CONSOLE.log("Initializing live agent with default configuration")
    with st.spinner("Initializing Live Agent..."):
        config = GeminiLiveAgentConfig(interaction_mode=InteractionMode.TEXT)
        agent = GeminiLiveAgent(config=config)
    return agent


def list_datasets(root: Path) -> List[str]:
    """List available datasets."""
    return [p.name for p in root.iterdir() if p.is_dir()]


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = {}
    if "live_session_active" not in st.session_state:
        st.session_state.live_session_active = False
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "current_interaction_mode" not in st.session_state:
        st.session_state.current_interaction_mode = None


def check_dataset_change(dataset_dir: Path):
    """Check if dataset changed and clear session if needed."""
    if st.session_state.current_dataset != str(dataset_dir):
        st.session_state.current_dataset = str(dataset_dir)
        st.session_state.chat_history = []
        st.session_state.live_session_active = False
        CONSOLE.log(f"Dataset changed to {dataset_dir}, cleared session")


# UI Components
def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Spatial Understanding Live Agent", page_icon="🤖", layout="wide"
    )

    # Enhanced CSS for mode indicators and voice mode
    st.markdown(
        """
    <style>
    .mode-indicator {
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
        font-weight: 500;
        text-align: center;
    }
    .general { background-color: #e3f2fd; color: #1565c0; }
    .object { background-color: #f3e5f5; color: #7b1fa2; }
    .cooking { background-color: #fff3e0; color: #ef6c00; }
    .navigation { background-color: #e8f5e8; color: #388e3c; }
    .accessibility { background-color: #fce4ec; color: #c2185b; }

    .voice-indicator {
        background-color: #ffebee;
        color: #c62828;
        padding: 8px;
        border-radius: 4px;
        margin: 8px 0;
        text-align: center;
        font-weight: bold;
    }

    .text-indicator {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 8px;
        border-radius: 4px;
        margin: 8px 0;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_sidebar() -> Tuple[
    GeminiLiveAgent,
    Path,
    str,
    OperationalMode,
    DirectionalStyle,
    DistanceStyle,
    int,
    bool,
    InteractionMode,
]:
    """Create sidebar controls and return configuration."""
    st.sidebar.title("🤖 Spatial Understanding Live Agent")

    # Interaction Mode Selection
    st.sidebar.header("🎙️ Interaction Mode")
    if WEBRTC_AVAILABLE:
        interaction_mode = st.sidebar.radio(
            "Select mode:",
            [InteractionMode.TEXT, InteractionMode.VOICE],
            format_func=lambda x: (
                "📝 Text Mode" if x == InteractionMode.TEXT else "🎙️ Voice Mode"
            ),
            help="Text mode for traditional chat, Voice mode for real-time audio interaction",
        )
    else:
        interaction_mode = InteractionMode.TEXT
        st.sidebar.info("Voice mode requires streamlit-webrtc package")

    # Dataset selection
    paths = PathConfig()
    scenes = list_datasets(paths.data)
    if not scenes:
        st.error(f"No scenes found in {paths.data}")
        st.stop()

    scene_name = st.sidebar.selectbox("Scene", scenes)
    dataset_dir = paths.data / scene_name
    check_dataset_change(dataset_dir)

    # Model selection with Live API specific default
    model_keys = list(MODEL_OPTIONS.keys())
    default_idx = (
        model_keys.index("gemini-2.5-pro-preview-05-06")
        if "gemini-2.5-pro-preview-05-06" in model_keys
        else 0
    )
    model_name = st.sidebar.selectbox(
        "Model",
        model_keys,
        index=default_idx,
        format_func=lambda k: MODEL_OPTIONS[k].split(" - ")[0],
    )

    # Mode selection
    st.sidebar.header("🎯 Operational Mode")
    mode_options = {
        OperationalMode.GENERAL_SCENE: "🔍 General Scene",
        OperationalMode.OBJECT_DETECTION: "📦 Object Detection",
        OperationalMode.COOKING_ASSISTANCE: "👨‍🍳 Cooking Help",
        OperationalMode.NAVIGATION_GUIDANCE: "🗺️ Navigation",
        OperationalMode.ACCESSIBILITY_SUPPORT: "♿ Accessibility",
    }

    selected_mode = st.sidebar.selectbox(
        "Select operational mode:",
        list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
    )

    # Response styles
    st.sidebar.header("⚙️ Settings")

    # Image rotation toggle
    is_rotated = st.sidebar.checkbox(
        "🔄 Rotate images 90° clockwise",
        value=True,
        help="When enabled, RGB and depth images are rotated 90 degrees clockwise.",
    )

    directional_style = st.sidebar.selectbox(
        "Direction style:",
        list(DirectionalStyle),
        format_func=lambda x: x.value.replace("_", " ").title(),
    )

    distance_style = st.sidebar.selectbox(
        "Distance style:",
        list(DistanceStyle),
        format_func=lambda x: x.value.replace("_", " ").title(),
    )

    # Create live agent (cached - only created once)
    live_agent = get_live_agent()

    # Update live agent settings only when they actually change
    CONSOLE.log(f"Updating live agent for dataset: {dataset_dir}")
    live_agent.update_dataset(dataset_dir, is_rotated)

    # Only update AABB detector if model changed
    if st.session_state.current_model != model_name:
        CONSOLE.log(
            f"Model changed from {st.session_state.current_model} to {model_name}, updating detector"
        )
        live_agent.update_aabb_detector(model_name)
        st.session_state.current_model = model_name

    # Only update interaction mode if it changed
    if st.session_state.current_interaction_mode != interaction_mode:
        CONSOLE.log(f"Interaction mode changed to {interaction_mode}")
        live_agent.update_interaction_mode(interaction_mode)
        st.session_state.current_interaction_mode = interaction_mode

    # Frame selection
    max_idx = len(live_agent.dataset) - 1
    frame_idx = st.sidebar.number_input(
        f"Frame (0-{max_idx})", min_value=0, max_value=max_idx, value=0
    )

    return (
        live_agent,
        dataset_dir,
        model_name,
        selected_mode,
        directional_style,
        distance_style,
        frame_idx,
        is_rotated,
        interaction_mode,
    )


def display_mode_indicator(mode: OperationalMode, interaction_mode: InteractionMode):
    """Display current operational and interaction modes."""
    mode_styles = {
        OperationalMode.GENERAL_SCENE: ("general", "🔍 General Scene Understanding"),
        OperationalMode.OBJECT_DETECTION: ("object", "📦 Object Detection & Analysis"),
        OperationalMode.COOKING_ASSISTANCE: ("cooking", "👨‍🍳 Cooking Assistance"),
        OperationalMode.NAVIGATION_GUIDANCE: ("navigation", "🗺️ Navigation Guidance"),
        OperationalMode.ACCESSIBILITY_SUPPORT: (
            "accessibility",
            "♿ Accessibility Support",
        ),
    }

    style_class, display_text = mode_styles.get(mode, ("general", "Unknown Mode"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div class="mode-indicator {style_class}">
            Current Mode: {display_text}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if interaction_mode == InteractionMode.VOICE:
            st.markdown(
                """
            <div class="voice-indicator">
                🎙️ Voice Mode Active
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="text-indicator">
                📝 Text Mode Active
            </div>
            """,
                unsafe_allow_html=True,
            )


def load_frame(live_agent: GeminiLiveAgent, frame_idx: int):
    """Load frame data from the live agent."""
    return live_agent.dataset[frame_idx]


def display_frame_visualizations(live_agent: GeminiLiveAgent, frame_idx: int):
    """Display frame images, detection results, and ground plane visualization."""
    frame = load_frame(live_agent, frame_idx)
    if frame is None:
        st.error("Failed to load frame")
        return

    # Add a new tab for Ground Plane visualization
    rgb_tab, depth_tab, det_tab, ground_tab = st.tabs(["RGB", "Depth", "Detections", "Ground Plane"])

    with rgb_tab:
        st.image(frame.rgb_image, caption="RGB Image", use_container_width=True)

    with depth_tab:
        depth_arr = np.array(frame.depth_image, dtype=np.float32)
        valid_mask = (depth_arr > 0) & (depth_arr < 50.0)
        if np.any(valid_mask):
            min_depth: float = float(np.min(depth_arr[valid_mask]))
            max_depth: float = float(np.max(depth_arr[valid_mask]))
            depth_display = np.zeros_like(depth_arr, dtype=np.uint8)
            if max_depth > min_depth:
                normalized = (depth_arr - min_depth) / (max_depth - min_depth) * 255
                depth_display[valid_mask] = normalized[valid_mask].astype(np.uint8)
            st.image(
                depth_display,
                caption=f"Depth Image (range: {min_depth:.2f}m - {max_depth:.2f}m)",
                use_container_width=True,
            )
        else:
            st.warning("No valid depth data found in this frame")

    with det_tab:
        available_det_keys = []
        for key in st.session_state.detection_results.keys():
            if key.startswith("det_"):
                try:
                    frame_num = int(key.split("_")[1])
                    available_det_keys.append((frame_num, key))
                except (IndexError, ValueError):
                    continue
        available_det_keys.sort(key=lambda x: x[0])
        detection_keys = [key for _, key in available_det_keys]
        if "det_nav_key_idx" not in st.session_state:
            current_det_key = f"det_{frame_idx}"
            if current_det_key in detection_keys:
                st.session_state.det_nav_key_idx = detection_keys.index(current_det_key)
            elif detection_keys:
                st.session_state.det_nav_key_idx = 0
            else:
                st.session_state.det_nav_key_idx = None
        if detection_keys:
            if st.session_state.det_nav_key_idx is None:
                current_det_key = f"det_{frame_idx}"
                if current_det_key in detection_keys:
                    st.session_state.det_nav_key_idx = detection_keys.index(
                        current_det_key
                    )
                else:
                    st.session_state.det_nav_key_idx = 0
            elif st.session_state.det_nav_key_idx >= len(detection_keys):
                st.session_state.det_nav_key_idx = len(detection_keys) - 1
            elif st.session_state.det_nav_key_idx < 0:
                st.session_state.det_nav_key_idx = 0
        else:
            st.session_state.det_nav_key_idx = None
        if detection_keys:
            current_idx = (
                st.session_state.det_nav_key_idx
                if st.session_state.det_nav_key_idx is not None
                else 0
            )
            current_key = detection_keys[current_idx]
            current_frame_num = int(current_key.split("_")[1])
            st.info(
                f"📍 Viewing detection from Frame {current_frame_num} ({current_idx + 1}/{len(detection_keys)} total detections)"
            )
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button(
                "⬅️ Previous Detection",
                key=f"prev_det_{frame_idx}",
                disabled=not detection_keys,
            ):
                if (
                    detection_keys
                    and st.session_state.det_nav_key_idx is not None
                    and st.session_state.det_nav_key_idx > 0
                ):
                    st.session_state.det_nav_key_idx -= 1
                    st.rerun()
        with nav_col2:
            if st.button("🔄 Current Frame", key=f"curr_det_{frame_idx}"):
                current_det_key = f"det_{frame_idx}"
                if current_det_key in detection_keys:
                    st.session_state.det_nav_key_idx = detection_keys.index(
                        current_det_key
                    )
                    st.rerun()
                else:
                    st.warning(f"No detection available for frame {frame_idx}")
        with nav_col3:
            if st.button(
                "➡️ Next Detection",
                key=f"next_det_{frame_idx}",
                disabled=not detection_keys,
            ):
                if (
                    detection_keys
                    and st.session_state.det_nav_key_idx is not None
                    and st.session_state.det_nav_key_idx < len(detection_keys) - 1
                ):
                    st.session_state.det_nav_key_idx += 1
                    st.rerun()
        if detection_keys and st.session_state.det_nav_key_idx is not None:
            current_det_key = detection_keys[st.session_state.det_nav_key_idx]
            detections = st.session_state.detection_results[current_det_key]
            current_frame_num = int(current_det_key.split("_")[1])
            assert isinstance(
                detections, AABBDetections
            ), f"Expected AABBDetections type, got {type(detections)}"
            st.image(
                detections.visualization_rgb,
                caption=f"Detections from Frame {current_frame_num}",
                use_container_width=True,
            )
            if detections.objects:
                st.subheader(
                    f"Frame {current_frame_num} - Detected Objects ({len(detections.objects)})"
                )
                rows = []
                for obj in detections.objects.values():
                    rows.append(
                        {
                            "Label": obj.label,
                            "Median depth [m]": obj.med_depth,
                            "Min depth [m]": obj.min_depth,
                            "Max depth [m]": obj.max_depth,
                            "Center 3D (bbox)": (
                                obj.center_3d_bbox.tolist()
                                if obj.center_3d_bbox is not None
                                else None
                            ),
                            "Center 3D (mask)": (
                                obj.center_3d_mask.tolist()
                                if obj.center_3d_mask is not None
                                else None
                            ),
                            "Rotation [deg]": obj.rotation_deg,
                            "Rotation clock": obj.rotation_clock,
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.info(f"No objects detected in Frame {current_frame_num}")
        else:
            st.info(
                "No detection results available. Run object detection to see results here."
            )

    with ground_tab:
        buf = DetectionVisualizer.plot_ground_plane_on_rgb(frame)
        st.image(buf, caption=None, use_container_width=True)


def get_suggested_prompts(mode: OperationalMode) -> List[str]:
    """Get mode-specific suggested prompts."""
    prompts = {
        OperationalMode.GENERAL_SCENE: [
            "What objects do you see?",
            "Provide a qualitative scene description!",
            "What tools are available?",
        ],
        OperationalMode.OBJECT_DETECTION: [
            "Find all chairs in this scene",
            "Detect kitchen appliances",
            "Show me any safety hazards",
        ],
        OperationalMode.COOKING_ASSISTANCE: [
            "What ingredients are visible?",
            "Are there cooking tools nearby?",
            "Is the workspace safe for cooking?",
        ],
        OperationalMode.NAVIGATION_GUIDANCE: [
            "What obstacles should I avoid?",
            "Describe the path ahead",
            "Which way to the exit?",
        ],
        OperationalMode.ACCESSIBILITY_SUPPORT: [
            "Are there accessibility barriers?",
            "What assistance is needed here?",
            "Describe accessibility features",
        ],
    }
    return prompts.get(mode, prompts[OperationalMode.GENERAL_SCENE])


def handle_chat_interaction(
    live_agent: GeminiLiveAgent, frame_idx: int, user_input: str
):
    """Process user input through Live Agent using actor pattern and generate response."""
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Live Agent is processing..."):
            try:
                # Send query to the actor-based live agent
                live_agent.ask(user_input, frame_idx)

                # Collect text chunks until quiescence (no new chunks for 0.5s)
                text_chunks: List[str] = []
                detection_result = None
                idle_start: Optional[float] = None
                response_time_ms: Optional[float] = None

                try:
                    while live_agent.next_event() is not None:
                        pass
                except Empty:
                    pass

                while True:
                    event = live_agent.next_event()
                    if event is None:
                        if text_chunks:
                            # once we have text, start idle timer
                            if idle_start is None:
                                idle_start = time.time()
                            elif time.time() - idle_start > 0.5:
                                break
                        else:
                            # no text yet, poll quickly
                            time.sleep(0.05)
                        continue

                    # reset idle timer on new event
                    idle_start = None

                    if isinstance(event, TextEvt):
                        text_chunks.append(event.text)
                        if event.response_time_ms is not None:
                            response_time_ms = event.response_time_ms
                    elif isinstance(event, DetectionsEvt):
                        detection_result = event.detections
                        det_key = f"det_{event.frame_idx}"
                        st.session_state.detection_results[det_key] = detection_result
                        if event.response_time_ms is not None:
                            response_time_ms = event.response_time_ms
                    elif isinstance(event, ErrorEvt):
                        st.error(f"Live Agent Error: {event.error}")
                        text_chunks = [f"Sorry, I encountered an error: {event.error}"]
                        break

                # DEBUG: log collected chunks
                try:
                    CONSOLE.log(f"Collected text chunks: {text_chunks}")
                except Exception:
                    pass

                response_text = "".join(text_chunks).strip()
                if not response_text:
                    response_text = "I'm processing your request; please wait a moment."

                # Display response with timing information
                if response_time_ms is not None:
                    # Store response time for statistics
                    if "response_times" not in st.session_state:
                        st.session_state.response_times = []
                    st.session_state.response_times.append(response_time_ms)

                    # Create a container with response text and timing badge
                    response_container = st.container()
                    with response_container:
                        st.markdown(response_text)

                        # Display timing information elegantly
                        timing_color = (
                            "#4CAF50"
                            if response_time_ms < 2000
                            else "#FF9800" if response_time_ms < 5000 else "#F44336"
                        )
                        st.markdown(
                            f"""
                            <div style="
                                display: inline-block;
                                background-color: {timing_color}20;
                                color: {timing_color};
                                padding: 2px 8px;
                                border-radius: 12px;
                                font-size: 0.8em;
                                font-weight: 500;
                                margin-top: 8px;
                                border: 1px solid {timing_color}40;
                            ">
                                ⏱️ {response_time_ms:.0f}ms
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(response_text)

            except Exception as e:
                CONSOLE.error(e, "Error communicating with Live Agent")
                st.error(f"Error communicating with Live Agent: {e}")
                response_text = f"Sorry, I encountered an error: {str(e)}"

        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", response_text))


def run_direct_detection(live_agent: GeminiLiveAgent, frame_idx: int) -> AABBDetections:
    """Run direct detection using the AABB detector (not through Live API)."""
    frame = load_frame(live_agent, frame_idx)
    return live_agent.aabb_detector.run_aabb_detection(frame)


def display_chat_interface(
    live_agent: GeminiLiveAgent,
    frame_idx: int,
    mode: OperationalMode,
    interaction_mode: InteractionMode,
):
    """Display the chat interface."""
    st.header("💬 Live Agent Chat Interface")

    # Suggested prompts
    with st.expander("💡 Suggested Prompts"):
        prompts = get_suggested_prompts(mode)
        cols = st.columns(len(prompts))
        for i, prompt in enumerate(prompts):
            if cols[i].button(prompt, key=f"prompt_{i}", use_container_width=True):
                handle_chat_interaction(live_agent, frame_idx, prompt)
                st.rerun()

    # Display chat history with enhanced styling
    for i, (role, message) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            if role == "assistant":
                # Check if we have timing information for this response
                if "response_times" in st.session_state and i // 2 < len(
                    st.session_state.response_times
                ):
                    response_time_ms = st.session_state.response_times[i // 2]

                    # Display message with timing badge
                    st.markdown(message)

                    # Show timing information for historical messages
                    timing_color = (
                        "#4CAF50"
                        if response_time_ms < 2000
                        else "#FF9800" if response_time_ms < 5000 else "#F44336"
                    )
                    st.markdown(
                        f"""
                        <div style="
                            display: inline-block;
                            background-color: {timing_color}15;
                            color: {timing_color};
                            padding: 1px 6px;
                            border-radius: 8px;
                            font-size: 0.7em;
                            font-weight: 400;
                            margin-top: 4px;
                            border: 1px solid {timing_color}30;
                        ">
                            ⏱️ {response_time_ms:.0f}ms
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(message)
            else:
                st.markdown(message)

    # Chat input based on interaction mode
    if interaction_mode == InteractionMode.TEXT:
        if prompt := st.chat_input("Ask about the scene..."):
            handle_chat_interaction(live_agent, frame_idx, prompt)
            st.rerun()


def display_voice_interface(live_agent: GeminiLiveAgent, frame_idx: int):
    """Display the voice interface using streamlit-webrtc."""
    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc not available. Install it for voice mode.")
        return

    st.header("🎙️ Voice Interface")

    # WebRTC configuration for production deployment
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        raise NotImplementedError()
        """Process audio frames from microphone."""
        # Convert audio frame to raw audio data
        audio_data = frame.to_ndarray()

        # Store audio data for processing
        with live_agent_lock:
            live_agent_container["audio_data"] = audio_data

        return frame

    # Create WebRTC streamer for audio
    webrtc_ctx = webrtc_streamer(
        key="live-agent-audio",
        mode=WebRtcMode.SENDRECV,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    # Display voice interaction status
    if webrtc_ctx.state.playing:
        st.success("🎙️ Voice interaction active - speak naturally!")

        # Process accumulated audio data
        with live_agent_lock:
            if "audio_data" in live_agent_container:
                # Here we would implement:
                # 1. Audio transcription (speech-to-text)
                # 2. Live Agent processing
                # 3. Text-to-speech for response
                # For now, we show a placeholder
                st.info("Audio processing would happen here...")
    else:
        st.info("Click 'START' to begin voice interaction")


def display_quick_actions(live_agent: GeminiLiveAgent, frame_idx: int):
    """Display quick action buttons."""
    st.sidebar.header("🚀 Quick Actions")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("🔍 Detect", use_container_width=True):
        with st.spinner("Running direct detection..."):
            detections = run_direct_detection(live_agent, frame_idx)
            det_key = f"det_{frame_idx}"
            st.session_state.detection_results[det_key] = detections
        st.sidebar.success("Detection complete!")
        st.rerun()

    if col2.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.live_session_active = False
        st.session_state.live_responses = []
        # Clear detection results
        st.session_state.detection_results = {}
        st.session_state.subset_results = {}
        st.sidebar.success("History cleared!")
        st.rerun()


def display_session_info(live_agent: GeminiLiveAgent):
    """Display session statistics and health."""
    st.sidebar.header("📊 Session Info")

    # Chat history stats
    history = st.session_state.chat_history
    user_msgs = len([msg for role, msg in history if role == "user"])
    assistant_msgs = len([msg for role, msg in history if role == "assistant"])

    col1, col2 = st.sidebar.columns(2)
    col1.metric("User", user_msgs)
    col2.metric("Assistant", assistant_msgs)

    # Response time statistics
    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    if st.session_state.response_times:
        avg_response_time = sum(st.session_state.response_times) / len(
            st.session_state.response_times
        )
        min_response_time = min(st.session_state.response_times)
        max_response_time = max(st.session_state.response_times)

        st.sidebar.subheader("⏱️ Response Times")

        # Display metrics in a nice layout
        time_col1, time_col2 = st.sidebar.columns(2)
        time_col1.metric("Avg", f"{avg_response_time:.0f}ms")
        time_col2.metric("Count", len(st.session_state.response_times))

        time_col3, time_col4 = st.sidebar.columns(2)
        time_col3.metric("Min", f"{min_response_time:.0f}ms")
        time_col4.metric("Max", f"{max_response_time:.0f}ms")

        # Show a simple chart of recent response times
        if len(st.session_state.response_times) > 1:
            recent_times = st.session_state.response_times[-10:]  # Last 10 responses
            st.sidebar.line_chart(recent_times)

    # Actor status with health check
    if hasattr(live_agent, "actor_thread") and live_agent.actor_thread is not None:
        if live_agent.actor_thread.is_alive():
            st.sidebar.success("🟢 Actor: Running")
        else:
            st.sidebar.error("🔴 Actor: Stopped")
    else:
        st.sidebar.warning("🟡 Actor: Not started")


def handle_frame_context_change(live_agent: GeminiLiveAgent, frame_idx: int):
    """Handle frame context changes using the actor pattern."""
    # Track the last frame index in session state
    if "last_frame_idx" not in st.session_state:
        # Initialize with the current frame to avoid duplicate initial send
        st.session_state.last_frame_idx = live_agent.current_frame_idx

    # Check if frame has actually changed
    if st.session_state.last_frame_idx != frame_idx:
        st.session_state.last_frame_idx = frame_idx

        # Set the current frame using the actor pattern (thread-safe)
        try:
            live_agent.set_current_frame(frame_idx)
            CONSOLE.log(f"Set frame {frame_idx} context via actor pattern")
        except Exception as e:
            CONSOLE.error(e, "Failed to set frame context")


# Main Application
def main():
    """Main application function."""
    setup_page_config()
    init_session_state()

    # Get configuration from sidebar
    (
        live_agent,
        dataset_dir,
        model_name,
        mode,
        dir_style,
        dist_style,
        frame_idx,
        is_rotated,
        interaction_mode,
    ) = create_sidebar()

    # Set up detection callback to store results in session state
    def store_detection_results(frame_idx: int, detections: AABBDetections):
        """Callback to store detection results in Streamlit session state."""
        det_key = f"det_{frame_idx}"
        st.session_state.detection_results[det_key] = detections
        CONSOLE.log(f"Stored detection results for frame {frame_idx}")

    live_agent.detection_callback = store_detection_results

    # Handle frame context changes
    handle_frame_context_change(live_agent, frame_idx)

    # Display quick actions and session info
    display_quick_actions(live_agent, frame_idx)
    display_session_info(live_agent)

    # Main content area
    display_mode_indicator(mode, interaction_mode)

    # Voice interface for voice mode
    if interaction_mode == InteractionMode.VOICE:
        display_voice_interface(live_agent, frame_idx)
        st.divider()

    # Create layout columns
    col1, col2 = st.columns([3, 2])

    with col1:
        display_frame_visualizations(live_agent, frame_idx)

    with col2:
        display_chat_interface(live_agent, frame_idx, mode, interaction_mode)


if __name__ == "__main__":
    main()
