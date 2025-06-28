# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and live chat functions, the application seeks to enhance navigation and interaction in various environments.

# Spatial Guidance Setup Instructions

## Table of Contents

- [System Dependencies](#system-dependencies)
- [Installation](#installation)
- [API Configuration](#api-configuration)
- [Dataset Setup](#dataset-setup)
- [Repository Structure](#repository-structure)


## System Dependencies

### macOS

```bash
# Install system dependencies
brew install portaudio
brew install ffmpeg
```

### Ubuntu/Debian

```bash
sudo apt update

sudo apt install -y \
    portaudio19-dev \
    ffmpeg \

sudo apt install -y pulseaudio pulseaudio-utils
```

## Installation

Clone the dlvc-spatial-understanding repository:
- [GitHub Repository](https://github.com/JanDuchscherer104/dlvc-spatial-understanding)
- [GitLab Repository](https://gitlab.lrz.de/dlvc04-sose-25/dlvc-04-spatial-understanding)

```bash
git clone git@gitlab.lrz.de:dlvc04-sose-25/dlvc-04-spatial-understanding.git
# alternatively from GitHub
# git clone git@github.com:JanDuchscherer104/dlvc-spatial-understanding.git
cd dlvc-04-spatial-understanding

conda create -n dlvc python=3.11
conda activate dlvc
```

### Poetry Installation

```bash
conda install poetry
cd src/spatial_guidance

poetry install

# Verify installation
poetry run python -c "import spatial_guidance; print('✅ Installation successful')"
```

## API Configuration

[Create a Gemini API key](https://aistudio.google.com/app/apikey) and add it to `.env` file in the project root directory.


## Dataset Setup

### Default Dataset Structure

The application expects datasets in the following structure:

```
.data/SmartAIs-Recorded-Data/
├── scene1/
│   ├── rgb/
│   ├── depth/
│   ├── camera_matrix.csv
│   ├── imu.csv
|   └── odometry.csv
├── scene2/
│   ├── rgb/
│   ├── depth/
│   ├── camera_matrix.csv
│   ├── imu.csv
|   └── odometry.csv
└── ...
```


### Option 1: Default Location

```bash
# Create the default data directory
mkdir -p .data/SmartAIs-Recorded-Data

# Copy your datasets to this location
cp -r /path/to/your/datasets/* .data/SmartAIs-Recorded-Data/
```

### Option 2: Custom Location

Edit `src/spatial_guidance/spatial_guidance/utils/configs.py`:

```python
# Update the data path in configs.py
class PathConfig:
    def __init__(self):
        self.data = Path("/your/custom/path/to/datasets")

# (Optional) Update the scenario path in stray_scanner_paths.py
class StrayScannerPaths(BaseConfig):
    """Configuration for Stray Scanner dataset paths."""

    dataset_dir: Annotated[Path, Field(default="scenario")] # relative to path_config.data
```

### Step 3: Streamlit Application

```bash
streamlit run spatial_guidance/ui/streamlit_app_live.py
```


# Repository Structure

```
spatial_guidance
├── poetry.lock
├── pyproject.toml
└── spatial_guidance
    ├── __init__.py
    ├── data_contracts
    │   ├── __init__.py
    │   ├── aabb_segmentation.py
    │   │   ├── RawAABBDetSeg: Raw segmentation detection data
    │   │   ├── AABBDetection: Object detection with masks
    │   │   └── AABBDetections: Collection of AABB detections
    │   ├── core.py
    │   │   └── DataModel: Base data model class
    │   ├── dataset.py
    │   │   └── DatasetOut(DataModel): Complete dataset frame output
    │   └── obb_detection.py
    │       ├── OBBDetection: 3D oriented bounding boxes
    │       ├── OBBDetections: Collection of OBB detections
    │       └── RawOBBDetection: Raw OBB detection data
    ├── data_handling
    │   ├── __init__.py
    │   └── stray_scanner
    │       ├── __init__.py
    │       ├── data_parser.py
    │       │   ├── StrayScannerDataParser: Dataset file parser interface
    │       │   └── StrayScannerDataParserConfig: Parser configuration settings
    │       ├── stray_dataset.py
    │       │   ├── StrayDataset: Main dataset access interface
    │       │   └── StrayDatasetConfig: Dataset configuration parameters
    │       └── stray_scanner_paths.py
    │           └── StrayScannerPaths: File path configuration
    ├── live_agent
    │   ├── __init__.py
    │   ├── actor_protocols.py
    │   │   ├── AskCmd: Text query command
    │   │   ├── AudioCmd: Audio input command
    │   │   ├── AudioEvt: Audio response event
    │   │   ├── DetectionsEvt: Detection results event
    │   │   ├── ErrorEvt: Error notification event
    │   │   ├── SetFrameCmd: Frame selection command
    │   │   ├── TextEvt: Text response event
    │   │   ├── _Cmd: Base command protocol
    │   │   └── _Evt: Base event protocol
    │   ├── gemini_live_agent.py
    │   │   └── GeminiLiveAgent: Live audio interaction agent
    │   ├── live_agent_config.py
    │   │   └── GeminiLiveAgentConfig: Live agent configuration
    │   ├── live_agent_enums.py
    │   │   ├── DirectionalStyle: Direction description style enum
    │   │   ├── DistanceStyle: Distance description style enum
    │   │   ├── GenState: Generation state enum
    │   │   ├── InteractionMode: User interaction mode enum
    │   │   ├── ModePromptTemplates: Mode-specific prompt templates
    │   │   ├── OperationalMode: Operation mode enum
    │   │   └── ResponseStyle: Combined response style configuration
    │   ├── prompt_templates.py
    │   │   └── LiveAgentPromptTemplates: System prompt template manager
    │   └── tools.py
    │       └── LiveAgentTools: Available tool declarations
    ├── scene_understanding
    │   ├── __init__.py
    │   ├── gemini_aabb_detection.py
    │   │   ├── GeminiAABBDetSegConfig: AABB detection configuration
    │   │   └── GeminiAABBDetSeg: AABB detection and segmentation
    │   └── gemini_obb_detection.py
    │       ├── GeminiOBBDet: 3D OBB detection system
    │       └── GeminiOBBDetConfig: OBB detection configuration
    ├── ui
    │   ├── __init__.py
    │   └── streamlit_app_live.py
    ├── utils
    │   ├── __init__.py
    │   ├── base_config.py
    │   │   ├── BaseConfig: Base configuration class
    │   │   ├── NoTarget: No-operation target implementation
    │   │   └── SingletonConfig: Singleton configuration base
    │   ├── configs.py
    │   │   └── PathConfig: System path configuration
    │   └── console.py
    │       └── Console: Logging and console output
    └── visualization
        ├── __init__.py
        └── detection_visualizer.py
            └── DetectionVisualizer: Detection result visualization
```