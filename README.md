# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and live chat functions, the application seeks to enhance navigation and interaction in various environments.


## Table of Contents

- [3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users](#3d-spatial-scene-understanding-and-interactive-audio-guidance-for-blind-users)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Setup Instructions](#setup-instructions)
    - [System Dependencies](#system-dependencies)
      - [macOS](#macos)
      - [Ubuntu/Debian](#ubuntudebian)
    - [Installation](#installation)
    - [Poetry Installation](#poetry-installation)
    - [API Configuration](#api-configuration)
    - [Dataset Setup](#dataset-setup)
      - [Default Dataset Structure](#default-dataset-structure)
      - [Option 1: Default Location](#option-1-default-location)
      - [Option 2: Custom Location](#option-2-custom-location)
    - [Run the Application](#run-the-application)
  - [Repository Structure](#repository-structure)

## Setup Instructions

### System Dependencies

#### macOS

```bash
brew install portaudio
brew install ffmpeg
```

#### Ubuntu/Debian

```bash
sudo apt update

sudo apt install -y \
    portaudio19-dev \
    ffmpeg \

sudo apt install -y pulseaudio pulseaudio-utils
```

### Installation

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

### API Configuration

[Create a Gemini API key](https://aistudio.google.com/app/apikey) and add it to `.env` file in the project root directory.


### Dataset Setup

#### Default Dataset Structure

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


#### Option 1: Default Location

```bash
# Create the default data directory
mkdir -p .data/SmartAIs-Recorded-Data

# Copy your datasets to this location
cp -r /path/to/your/datasets/* .data/SmartAIs-Recorded-Data/
```

#### Option 2: Custom Location

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

### Run the Application

```bash
streamlit run spatial_guidance/ui/streamlit_app_live.py
```

## Repository Structure

```
spatial_guidance
├── poetry.lock
├── pyproject.toml
└── spatial_guidance
    ├── __init__.py
    ├── data_contracts
    │   ├── __init__.py
    │   ├── aabb_segmentation.py
    │   │   ├── RawAABBDetSeg(DataModel): Raw segmentation detection data
    │   │   ├── AABBDetection(DataModel): Object detection with masks
    │   │   └── AABBDetections(DataModel): Collection of AABB detections
    │   ├── core.py
    │   │   └── DataModel(BaseModel): Base data contract (interface)
    │   ├── dataset.py
    │   │   └── DatasetOut(DataModel): Dataset frame output
    │   └── obb_detection.py
    │       ├── RawOBBDetection(DataModel): Raw OBB detection data
    │       ├── OBBDetection(DataModel): 3D oriented bounding boxes
    │       └── OBBDetections(DataModel): Collection of OBB detections
    ├── data_handling
    │   ├── __init__.py
    │   └── stray_scanner
    │       ├── __init__.py
    │       ├── data_parser.py
    │       │   ├── StrayScannerDataParserConfig(BaseConfig): Parser configuration settings
    │       │   └── StrayScannerDataParser: Dataset parsing
    │       ├── stray_dataset.py
    │       │   ├── StrayDatasetConfig(BaseConfig): Dataset configuration parameters
    │       │   └── StrayDataset: Main dataset that yields DatasetOut frames
    │       └── stray_scanner_paths.py
    │           └── StrayScannerPaths(BaseConfig): Configuration of the file structure
    ├── live_agent
    │   ├── __init__.py
    │   ├── actor_protocols.py
    │   │   ├── _Cmd: Base command protocol
    │   │   ├── AskCmd(_Cmd): Text query command
    │   │   ├── AudioCmd(_Cmd): Audio input command
    │   │   ├── SetFrameCmd(_Cmd): Frame selection command
    │   │   ├── _Evt: Base event protocol
    │   │   ├── TextEvt(_Evt): Text response event
    │   │   ├── AudioEvt(_Evt): Audio response event
    │   │   ├── DetectionsEvt(_Evt): Detection results event
    │   │   └── ErrorEvt(_Evt): Error notification event
    │   ├── gemini_live_agent.py
    │   │   └── GeminiLiveAgent: Agent using Gemini Live API
    │   ├── live_agent_config.py
    │   │   └── GeminiLiveAgentConfig(BaseConfig): Live agent configuration
    │   ├── live_agent_enums.py
    │   │   ├── DirectionalStyle(Enum): Direction description style enum
    │   │   ├── DistanceStyle(Enum): Distance description style enum
    │   │   ├── ResponseStyle(NamedTuple): Combined response style configuration
    │   │   ├── GenState(Enum): Generation state enum
    │   │   ├── InteractionMode(Enum): User interaction mode enum
    │   │   ├── OperationalMode(Enum): Operation mode enum
    │   │   └── ModePromptTemplates: Mode-specific prompt templates
    │   ├── prompt_templates.py
    │   │   └── LiveAgentPromptTemplates(BaseConfig): System prompt
    │   └── tools.py
    │       └── LiveAgentTools(BaseConfig): Tool declarations
    ├── scene_understanding
    │   ├── __init__.py
    │   ├── gemini_aabb_detection.py
    │   │   ├── GeminiAABBDetSegConfig(BaseConfig): AABB detection configuration
    │   │   └── GeminiAABBDetSeg: AABB detection and segmentation
    │   └── gemini_obb_detection.py
    │       ├── GeminiOBBDetConfig(BaseConfig): OBB detection configuration
    │       └── GeminiOBBDet: 3D OBB detection system
    ├── ui
    │   ├── __init__.py
    │   └── streamlit_app_live.py
    ├── utils
    │   ├── __init__.py
    │   ├── base_config.py
    │   │   ├── NoTarget: Target Type for BaseConfig's w/o target
    │   │   ├── BaseConfig(BaseModel): Base configuration class for Config-as-Factory pattern
    │   │   └── SingletonConfig(BaseConfig): Singleton configuration base
    │   ├── configs.py
    │   │   └── PathConfig(SingletonConfig): Path handling
    │   └── console.py
    │       └── Console(RichConsole): Fancy logging
    └── visualization
        ├── __init__.py
        └── detection_visualizer.py
            └── DetectionVisualizer: Detection result visualization
```