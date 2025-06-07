from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from spatial_guidance.services.scene_pipeline import (
    ScenePipeline,
    VALID_LIVE_MODELS,
)
from spatial_guidance.utils import Console, PathConfig

CONSOLE = Console.with_prefix("streamlit_app")


@st.cache_resource
def get_pipeline(dataset_dir: Path, model_name: str) -> ScenePipeline:
    CONSOLE.log(f"Initializing pipeline for {dataset_dir} with {model_name}")
    return ScenePipeline(dataset_dir, model_name)


def list_datasets(root: Path) -> List[str]:
    return [p.name for p in root.iterdir() if p.is_dir()]


st.set_page_config(page_title="Scene Guidance")

paths = PathConfig()
root_dir = paths.data

available = list_datasets(root_dir)
sel_ds = st.sidebar.selectbox("Dataset folder", available)
model_name = st.sidebar.selectbox("Gemini model", VALID_LIVE_MODELS)

pipeline = get_pipeline(root_dir / sel_ds, model_name)

max_idx = len(pipeline.dataset) - 1
idx = st.sidebar.number_input("Frame index", 0, max_idx, 0, 1)

frame = pipeline.load_frame(idx)

rgb_tab, depth_tab, det_tab = st.tabs(["RGB", "Depth", "Detections"])
with rgb_tab:
    st.image(frame.rgb_image, caption="RGB")
with depth_tab:
    st.image(frame.depth_image, caption="Depth", clamp=True)

if st.sidebar.button("Run Detection"):
    detections = pipeline.run_detection(idx)
    st.session_state["dets"] = detections

detections = st.session_state.get("dets")

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

history = st.session_state.setdefault("chat_history", [])
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
