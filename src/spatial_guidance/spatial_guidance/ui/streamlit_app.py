from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import streamlit as st

from spatial_guidance.scene_understanding.scene_pipeline import (
    MODEL_OPTIONS,
    ScenePipeline,
)
from spatial_guidance.utils import Console, PathConfig

CONSOLE = Console.with_prefix("streamlit_app")


@st.cache_resource(hash_funcs={Path: str})
def get_pipeline(dataset_dir: Path, model_name: str) -> ScenePipeline:
    CONSOLE.log(f"Initializing pipeline for {dataset_dir} with {model_name}")
    with st.spinner(
        f"Initializing pipeline for {dataset_dir.name} with {model_name}..."
    ):
        pipeline = ScenePipeline(dataset_dir, model_name)
    return pipeline


def list_datasets(root: Path) -> List[str]:
    return [p.name for p in root.iterdir() if p.is_dir()]


st.set_page_config(page_title="Scene Guidance")

paths = PathConfig()

# Simplified dataset selection: scenes are directly under root_dir
scenes = list_datasets(paths.data)
if not scenes:
    st.error(f"No scenes found in {paths.data}. Please add scene data.")
    st.stop()

scene_sel = st.sidebar.selectbox("Scene", scenes)
dataset_dir = paths.data / scene_sel

model_name = st.sidebar.selectbox(
    "Gemini model",
    list(MODEL_OPTIONS.keys()),
    format_func=lambda k: MODEL_OPTIONS[k],
)

pipeline = get_pipeline(dataset_dir, model_name)

max_idx = len(pipeline.dataset) - 1
idx_label = f"Frame index (0 to {max_idx})"
idx = st.sidebar.number_input(idx_label, 0, max_idx, 0, 1)

# Action selection
action_options = ["Detect Objects", "Describe Scene"]
selected_action = st.sidebar.selectbox("Action", action_options)

frame = pipeline.load_frame(idx)
if frame is None:
    st.error("Frame could not be loaded - check dataset contents.")
    st.stop()

rgb_tab, depth_tab, det_tab = st.tabs(["RGB", "Depth", "Detections"])
with rgb_tab:
    st.image(frame.rgb_image, caption="RGB")
with depth_tab:
    depth_arr = np.array(frame.depth_image, dtype=np.float32)
    st.image(depth_arr, caption="Depth", clamp=True)

history = st.session_state.setdefault("chat_history", [])

if st.sidebar.button("Run Action"):
    if selected_action == "Detect Objects":
        with st.spinner("Running object detection..."):
            detections_result = pipeline.run_detection(idx)
        st.session_state["dets"] = detections_result
        st.session_state["dets_idx"] = idx
        # Clear previous description if any
        st.session_state.pop("scene_description", None)
    elif selected_action == "Describe Scene":
        with st.spinner("Generating scene description..."):
            # Use a generic prompt for scene description
            description_prompt = "Describe all objects you see in this image, their appearances, and their spatial relationships to each other."
            text, _ = pipeline.answer_query(description_prompt, idx)
        st.session_state["scene_description"] = text
        st.session_state["scene_description_idx"] = idx
        history.append(("assistant", text))
        # Clear previous detections if any
        st.session_state.pop("dets", None)


detections = (
    st.session_state.get("dets") if st.session_state.get("dets_idx") == idx else None
)

scene_description = (
    st.session_state.get("scene_description")
    if st.session_state.get("scene_description_idx") == idx
    else None
)

if detections:
    with det_tab:
        st.image(detections.visualization_rgb, caption="Detections")
        labels = [d.label for d in detections.objects]
        if labels:
            sel = st.selectbox("Select object", labels)
            obj = next(o for o in detections.objects if o.label == sel)
            st.write(
                f"Depth m(min/med/max): {obj.min_depth:.2f}/{obj.med_depth:.2f}/{obj.max_depth:.2f}"
            )

# Display scene description if available (e.g., in the chat or a dedicated area)
# The chat history will automatically update if "Describe Scene" was run.

for m in history:
    with st.chat_message(m[0]):
        st.markdown(m[1])

prompt = st.chat_input("Ask about this frame")
if prompt:
    history.append(("user", prompt))
    with st.chat_message("assistant"):
        text, _ = pipeline.answer_query(prompt, idx)
        st.markdown(text)
        history.append(("assistant", text))
