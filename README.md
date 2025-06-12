# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

SpatialGuidance is a research prototype that combines depth‑based scene analysis
with Google Gemini models to describe the environment for visually impaired
users.  The project integrates multiple components for data parsing, object
segmentation, 3‑D geometry reasoning and natural language generation.  The goal
is a real‑time system that can detect relevant objects, compute their relative
positions in world space and provide concise spoken directions.

- **3‑D Spatial Understanding** – uses depth and camera pose information to
  recover object locations and bearings.
- **Gemini Driven Detection** – leverages Google Gemini models for accurate
  segmentation and classification.
- **Interactive Audio Guidance** – produces short descriptions or voice prompts
  that are easy to follow while moving.


## Structure of the SpatialGuidance Package

```plaintext
spatial_guidance
├── __init__.py
├── data_contracts
│   ├── __init__.py
│   ├── aabb_segmentation.py
│   ├── core.py
│   ├── dataset.py
│   └── obb_detection.py
├── data_handling
│   ├── __init__.py
│   └── stray_scanner
│       ├── __init__.py
│       ├── data_parser.py
│       ├── stray_dataset.py
│       └── stray_scanner_paths.py
├── live_agent
│   ├── __init__.py
│   ├── actor_protocols.py
│   ├── exec_api.py
│   ├── gemini_live_agent.py
│   ├── live_agent_config.py
│   ├── live_agent_enums.py
│   └── prompt_templates.py
├── response_generation
│   ├── __init__.py
│   └── response_generator.py
├── scene_understanding
│   ├── __init__.py
│   ├── gemini_aabb_detection.py
│   ├── gemini_obb_detection.py
│   └── gemini_scene_descriptor.py
├── ui
│   ├── __init__.py
│   └── streamlit_app_live.py
├── utils
│   ├── __init__.py
│   ├── base_config.py
│   ├── configs.py
│   └── console.py
└── visualization
    ├── __init__.py
    ├── detection_visualizer.py
    └── scene_visualizer.py
```

### Symbols by File

- `data_contracts/aabb_segmentation.py`: `RawAABBDetSeg`, `AABBDetection`, `AABBDetections`
- `data_contracts/core.py`: `DataModel`
- `data_contracts/dataset.py`: `PipelineIn`, `DatasetOut`
- `data_contracts/obb_detection.py`: `RawOBBDetection`, `OBBDetection`, `OBBDetections`
- `data_handling/stray_scanner/data_parser.py`: `StrayScannerDataParserConfig`, `StrayScannerDataParser`
- `data_handling/stray_scanner/stray_dataset.py`: `StrayDatasetConfig`, `StrayDataset`
- `data_handling/stray_scanner/stray_scanner_paths.py`: `StrayScannerPaths`
- `live_agent/actor_protocols.py`: `_Cmd`, `AskCmd`, `AudioCmd`, `SetFrameCmd`, `_Evt`, `TextEvt`, `AudioEvt`, `DetectionsEvt`, `ErrorEvt`
- `live_agent/exec_api.py`: `_ExecAPI`
- `live_agent/gemini_live_agent.py`: `GeminiLiveAgent`
- `live_agent/live_agent_config.py`: `GeminiLiveAgentConfig`
- `live_agent/live_agent_enums.py`: `DirectionalStyle`, `DistanceStyle`, `GenState`, `InteractionMode`, `OperationalMode`, `ModePromptTemplates`
- `response_generation/response_generator.py`: `DirectionalStyle`, `DistanceStyle`, `ResponseGenerator`
- `scene_understanding/gemini_aabb_detection.py`: `GeminiAABBDetSegConfig`, `GeminiAABBDetSeg`
- `scene_understanding/gemini_obb_detection.py`: `GeminiOBBDetConfig`, `GeminiOBBDet`
- `scene_understanding/gemini_scene_descriptor.py`: `GeminiSceneDescriptorConfig`, `GeminiSceneDescriptor`
- `utils/base_config.py`: `BaseConfig`, `NoTarget`, `SingletonConfig`
- `utils/configs.py`: `PathConfig`
- `utils/console.py`: `Console`
- `visualization/detection_visualizer.py`: `DetectionVisualizer`
- `visualization/scene_visualizer.py`: `VisualizationType`, `SceneVisualizerConfig`, `SceneVisualizer`

## Setup

For comprehensive setup instructions, please see **[SETUP.md](SETUP.md)**.
