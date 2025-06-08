from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from spatial_guidance.data_contracts.aabb_segmentation import AABBDetections
from spatial_guidance.gemini_client import OperationalMode
from spatial_guidance.response_generation import DirectionalStyle, DistanceStyle
from spatial_guidance.scene_understanding.scene_pipeline import (
    MODEL_OPTIONS,
    ScenePipeline,
)
from spatial_guidance.utils import Console, PathConfig

CONSOLE = Console.with_prefix("streamlit_app_refactored")


# Configuration and Setup
@st.cache_resource(hash_funcs={Path: lambda p: (str(p), p.stat().st_mtime_ns)})
def get_pipeline(dataset_dir: Path, model_name: str, is_rotated: bool) -> ScenePipeline:
    """Create and cache the scene pipeline."""
    CONSOLE.log(
        f"Initializing pipeline for {dataset_dir} with {model_name}, is_rotated={is_rotated}"
    )
    with st.spinner(
        f"Initializing pipeline for {dataset_dir.name} with {model_name}..."
    ):
        pipeline = ScenePipeline(dataset_dir, model_name, is_rotated=is_rotated)
    return pipeline


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


# UI Components
def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Spatial Understanding Agent", page_icon="🤖", layout="wide"
    )

    # Simple CSS for mode indicators
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
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_sidebar() -> tuple[
    ScenePipeline,
    Path,
    str,
    OperationalMode,
    DirectionalStyle,
    DistanceStyle,
    int,
    bool,
]:
    """Create sidebar controls and return configuration."""
    st.sidebar.title("🤖 Spatial Understanding")

    # Dataset selection
    paths = PathConfig()
    scenes = list_datasets(paths.data)
    if not scenes:
        st.error(f"No scenes found in {paths.data}")
        st.stop()

    scene_name = st.sidebar.selectbox("Scene", scenes)
    dataset_dir = paths.data / scene_name

    # ‑‑ Model selection with sensible default for detection ‑‑
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
    st.sidebar.header("🎯 Mode")
    mode_options = {
        OperationalMode.GENERAL_SCENE: "🔍 General Scene",
        OperationalMode.OBJECT_DETECTION: "📦 Object Detection",
        OperationalMode.COOKING_ASSISTANCE: "👨‍🍳 Cooking Help",
        OperationalMode.NAVIGATION_GUIDANCE: "🗺️ Navigation",
        OperationalMode.ACCESSIBILITY_SUPPORT: "♿ Accessibility",
    }

    selected_mode = st.sidebar.selectbox(
        "Operational mode:",
        list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
    )

    # Response styles
    st.sidebar.header("⚙️ Settings")

    # Image rotation toggle
    is_rotated = st.sidebar.checkbox(
        "🔄 Rotate images 90° clockwise",
        value=True,
        help="When enabled, RGB and depth images are rotated 90 degrees clockwise (unless already from the rotated directory). This is useful for datasets where images need rotation to match the expected orientation.",
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

    # Frame selection
    pipeline = get_pipeline(dataset_dir, model_name, is_rotated)
    max_idx = len(pipeline.dataset) - 1
    frame_idx = st.sidebar.number_input(
        f"Frame (0-{max_idx})", min_value=0, max_value=max_idx, value=0
    )

    return (
        pipeline,
        dataset_dir,
        model_name,
        selected_mode,
        directional_style,
        distance_style,
        frame_idx,
        is_rotated,
    )


def display_mode_indicator(mode: OperationalMode):
    """Display current operational mode."""
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
    st.markdown(
        f"""
    <div class="mode-indicator {style_class}">
        Current Mode: {display_text}
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_frame_visualizations(pipeline: ScenePipeline, frame_idx: int):
    """Display frame images and detection results."""
    frame = pipeline.load_frame(frame_idx)
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

            # Show object details
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
            "Find all chairs",
            "Detect kitchen appliances",
            "Show me safety hazards",
        ],
        OperationalMode.COOKING_ASSISTANCE: [
            "What ingredients are visible?",
            "Are there cooking tools nearby?",
            "Is the workspace safe?",
        ],
        OperationalMode.NAVIGATION_GUIDANCE: [
            "What obstacles should I avoid?",
            "Describe the path ahead",
            "Which way to the exit?",
        ],
        OperationalMode.ACCESSIBILITY_SUPPORT: [
            "Are there accessibility barriers?",
            "What assistance is needed?",
            "Describe accessibility features",
        ],
    }
    return prompts.get(mode, prompts[OperationalMode.GENERAL_SCENE])


def handle_chat_interaction(pipeline: ScenePipeline, frame_idx: int, user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if pipeline.is_detection_request(user_input):
            # Handle detection requests
            with st.spinner("🔍 Detecting objects..."):
                subset_detections, response_text = (
                    pipeline.run_subset_detection_with_response(frame_idx, user_input)
                )

                if subset_detections.objects:
                    # Cache results
                    subset_key = f"subset_{frame_idx}_{hash(user_input)}"
                    st.session_state.subset_results[subset_key] = subset_detections

                    # Display response and visualization
                    st.markdown(response_text)
                    st.image(
                        subset_detections.visualization_rgb,
                        caption="Detection Results",
                        use_container_width=True,
                    )
                else:
                    response_text = f"No objects found matching: '{user_input}'"
                    st.markdown(response_text)
        else:
            # Handle general queries
            with st.spinner("🤖 Generating response..."):
                response_text, _ = pipeline.answer_query_with_context(
                    user_input, frame_idx
                )
                st.markdown(response_text)

        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", response_text))


def display_chat_interface(
    pipeline: ScenePipeline, frame_idx: int, mode: OperationalMode
):
    """Display the chat interface."""
    st.header("💬 Chat Interface")

    # Suggested prompts
    with st.expander("💡 Suggested Prompts"):
        prompts = get_suggested_prompts(mode)
        cols = st.columns(len(prompts))
        for i, prompt in enumerate(prompts):
            if cols[i].button(prompt, key=f"prompt_{i}", use_container_width=True):
                handle_chat_interaction(pipeline, frame_idx, prompt)
                st.rerun()

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    if user_input := st.chat_input("Ask about this scene..."):
        handle_chat_interaction(pipeline, frame_idx, user_input)
        st.rerun()


def display_quick_actions(pipeline: ScenePipeline, frame_idx: int):
    """Display quick action buttons."""
    st.sidebar.header("🚀 Quick Actions")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("🔍 Detect", use_container_width=True):
        with st.spinner("Running detection..."):
            detections = pipeline.run_detection(frame_idx)
            det_key = f"det_{frame_idx}"
            st.session_state.detection_results[det_key] = detections
        st.sidebar.success("Detection complete!")
        st.rerun()

    if col2.button("🗑️ Clear", use_container_width=True):
        pipeline.clear_chat_history()
        st.session_state.chat_history = []
        st.sidebar.success("History cleared!")
        st.rerun()


def display_session_info():
    """Display session statistics."""
    st.sidebar.header("📊 Session Info")

    history = st.session_state.chat_history
    user_msgs = len([msg for role, msg in history if role == "user"])
    assistant_msgs = len([msg for role, msg in history if role == "assistant"])

    col1, col2 = st.sidebar.columns(2)
    col1.metric("User", user_msgs)
    col2.metric("Assistant", assistant_msgs)


# Main Application
def main():
    """Main application function."""
    setup_page_config()
    init_session_state()

    # Get configuration from sidebar
    (
        pipeline,
        dataset_dir,
        model_name,
        mode,
        dir_style,
        dist_style,
        frame_idx,
        is_rotated,
    ) = create_sidebar()

    pipeline.set_operational_mode(mode)
    pipeline.response_generator.directional_style = dir_style
    pipeline.response_generator.distance_style = dist_style

    # Display quick actions and session info
    display_quick_actions(pipeline, frame_idx)
    display_session_info()

    # Main content area
    display_mode_indicator(mode)

    # Create layout columns
    col1, col2 = st.columns([2, 1])

    with col1:
        display_frame_visualizations(pipeline, frame_idx)

    with col2:
        display_chat_interface(pipeline, frame_idx, mode)


if __name__ == "__main__":
    main()
