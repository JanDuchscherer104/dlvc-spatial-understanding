import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nest_asyncio
import numpy as np
import pandas as pd
import streamlit as st

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

from spatial_guidance.data_contracts.aabb_segmentation import AABBDetections
from spatial_guidance.gemini_client import OperationalMode
from spatial_guidance.gemini_live_agent import (
    MODEL_OPTIONS,
    GeminiLiveAgent,
    GeminiLiveAgentConfig,
    InteractionMode,
)
from spatial_guidance.response_generation import DirectionalStyle, DistanceStyle
from spatial_guidance.utils import Console, PathConfig

CONSOLE = Console.with_prefix("streamlit_app_live")

# Global containers for thread-safe communication
live_agent_lock = threading.Lock()
live_agent_container = {"agent": None, "response": None, "detections": None}


# Configuration and Setup
@st.cache_resource(hash_funcs={Path: lambda p: (str(p), p.stat().st_mtime_ns)})
def get_live_agent(
    dataset_dir: Path,
    model_name: str,
    is_rotated: bool,
    interaction_mode: InteractionMode,
) -> GeminiLiveAgent:
    """Create and cache the live agent."""
    CONSOLE.log(
        f"Initializing live agent for {dataset_dir} with {model_name}, is_rotated={is_rotated}, mode={interaction_mode}"
    )
    with st.spinner(
        f"Initializing Live Agent for {dataset_dir.name} with {model_name}..."
    ):
        config = GeminiLiveAgentConfig(interaction_mode=interaction_mode)
        agent = GeminiLiveAgent(
            config=config, dataset_dir=dataset_dir, is_rotated=is_rotated
        )
        # Setup the detector with the selected model
        agent.setup_aabb_detector(model_name)
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
    if "subset_results" not in st.session_state:
        st.session_state.subset_results = {}
    if "live_session_active" not in st.session_state:
        st.session_state.live_session_active = False
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "live_responses" not in st.session_state:
        st.session_state.live_responses = []


def check_dataset_change(dataset_dir: Path):
    """Check if dataset changed and clear session if needed."""
    if st.session_state.current_dataset != str(dataset_dir):
        st.session_state.current_dataset = str(dataset_dir)
        st.session_state.chat_history = []
        st.session_state.live_session_active = False
        st.session_state.live_responses = []
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

    # Create live agent
    live_agent = get_live_agent(dataset_dir, model_name, is_rotated, interaction_mode)

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
    """Display frame images and detection results."""
    frame = load_frame(live_agent, frame_idx)
    if frame is None:
        st.error("Failed to load frame")
        return

    # Create tabs for different views
    rgb_tab, depth_tab, det_tab = st.tabs(["RGB", "Depth", "Detections"])

    with rgb_tab:
        st.image(frame.rgb_image, caption="RGB Image", use_container_width=True)

    with depth_tab:
        # Convert depth image to numpy array (values are in meters)
        depth_arr = np.array(frame.depth_image, dtype=np.float32)

        # Normalize depth values for display (convert meters to 0-255 range)
        # Filter out invalid depth values (0 or very large values)
        valid_mask = (depth_arr > 0) & (depth_arr < 50.0)  # Assume max 50m depth

        if np.any(valid_mask):
            # Normalize to 0-255 range for display
            min_depth: float = float(np.min(depth_arr[valid_mask]))
            max_depth: float = float(np.max(depth_arr[valid_mask]))

            # Create display depth image
            depth_display = np.zeros_like(depth_arr, dtype=np.uint8)
            if max_depth > min_depth:
                # Normalize valid depth values to 0-255
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
        # Show cached detection results if available
        det_key = f"det_{frame_idx}"
        if det_key in st.session_state.detection_results:
            detections = st.session_state.detection_results[det_key]
            assert isinstance(
                detections, AABBDetections
            ), f"Expected AABBDetections type, got {type(detections)}"
            st.image(
                detections.visualization_rgb,
                caption="Detections",
                use_container_width=True,
            )

            # Show object details in an interactive table
            if detections.objects:
                st.subheader(f"Detected Objects ({len(detections.objects)})")

                rows = []
                for obj in detections.objects:
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
            st.info("Run object detection to see results here")


def get_suggested_prompts(mode: OperationalMode) -> List[str]:
    """Get mode-specific suggested prompts."""
    prompts = {
        OperationalMode.GENERAL_SCENE: [
            "What objects do you see?",
            "Describe the spatial layout",
            "What's happening in this scene?",
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
    """Process user input through Live Agent and generate response."""
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Processing with Live Agent..."):
            try:
                # Use the new Streamlit integration method
                response_text = live_agent.handle_streamlit_query(user_input, frame_idx)
            except Exception as e:
                CONSOLE.error(f"Error communicating with Live Agent: {e}")
                st.error(f"Error communicating with Live Agent: {e}")
                response_text = f"Sorry, I encountered an error: {str(e)}"

            # Check if any detections were cached during the interaction
            det_key = f"det_{frame_idx}"
            if det_key in st.session_state.detection_results:
                detections = st.session_state.detection_results[det_key]
                if detections.objects:
                    st.image(
                        detections.visualization_rgb,
                        caption="Detection Results",
                        use_container_width=True,
                    )

            st.markdown(response_text)

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

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input based on interaction mode
    if interaction_mode == InteractionMode.TEXT:
        if user_input := st.chat_input("Ask about this scene..."):
            handle_chat_interaction(live_agent, frame_idx, user_input)
            st.rerun()
    else:
        # Voice mode chat input will be handled by WebRTC component
        st.info("🎙️ Voice mode: Use the microphone above to interact")


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
        # Also stop the Live API session
        try:
            run_async_in_thread(live_agent.stop_live_session())
        except Exception as e:
            CONSOLE.error(f"Error stopping Live session: {e}")
        st.sidebar.success("History cleared!")
        st.rerun()


def display_session_info(live_agent: GeminiLiveAgent):
    """Display session statistics."""
    st.sidebar.header("📊 Session Info")

    history = st.session_state.chat_history
    user_msgs = len([msg for role, msg in history if role == "user"])
    assistant_msgs = len([msg for role, msg in history if role == "assistant"])

    col1, col2 = st.sidebar.columns(2)
    col1.metric("User", user_msgs)
    col2.metric("Assistant", assistant_msgs)

    # Show Live API session status based on actual session state
    if hasattr(live_agent, "session") and live_agent.session is not None:
        st.sidebar.success("🟢 Live Session Active")
    else:
        st.sidebar.info("🔴 Live Session Inactive")


def run_async_in_thread(coro):
    """Run async function in a separate thread to avoid event loop conflicts."""

    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()


def handle_frame_context_change(live_agent: GeminiLiveAgent, frame_idx: int):
    """Handle frame context changes and automatically send to Live API if session is active."""
    # Track the last frame index in session state
    if "last_frame_idx" not in st.session_state:
        st.session_state.last_frame_idx = None

    # Check if frame has changed
    if st.session_state.last_frame_idx != frame_idx:
        st.session_state.last_frame_idx = frame_idx

        # Set the current frame in the live agent (this will automatically send context if session is active)
        try:
            # Run the async operation in a separate thread to avoid event loop conflicts
            if hasattr(live_agent, "set_current_frame"):
                context_sent = run_async_in_thread(
                    live_agent.set_current_frame(frame_idx)
                )
                if context_sent:
                    CONSOLE.log(f"Sent frame {frame_idx} context to Live API")
        except Exception as e:
            CONSOLE.error(f"Failed to set frame context: {e}")


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

    # Update live agent settings when sidebar changes
    if hasattr(live_agent, "setup_aabb_detector"):
        live_agent.setup_aabb_detector(model_name)
    if hasattr(live_agent, "setup_interaction_mode"):
        live_agent.setup_interaction_mode(interaction_mode)

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
