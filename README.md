# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and live chat functions, the application seeks to enhance navigation and interaction in various environments.

# Spatial Guidance Setup Instructions

## Table of Contents

- [System Dependencies](#system-dependencies)
- [Installation](#installation)
- [API Configuration](#api-configuration)
- [Dataset Setup](#dataset-setup)


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

```bash
git clone https://github.com/your-org/dlvc-04-spatial-understanding.git
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
