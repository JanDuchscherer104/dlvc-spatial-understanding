# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and spatial audio rendering, the application seeks to enhance navigation and interaction in various environments.

- **3D Spatial Scene Understanding**: Utilizes depth estimation and object detection to create a comprehensive understanding of the user's surroundings.
- **Interactive Audio Guidance**: Generates spatial audio cues and spoken instructions to assist users in navigating their environment effectively.

## Strucuture of the SpatialGuidance Package

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

## Setup

```bash
conda create -n dlvc python=3.11
conda activate dlvc

# from root of the repo
pip install -e src/spatial_guidance
```

## Data

The stray scanner datasets should optimally be placed in `.data/SmartAIs-Recorded-Data`. Alternatively, you can modify the paths in `src/utils/configs.py` to point to the correct location of your datasets.

## Run

```bash
streamlit run src/spatial_guidance/spatial_guidance/ui/streamlit_app_live.py
```
