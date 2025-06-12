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

### Option A: Poetry Installation (recommended)

```bash
conda install poetry
cd src/spatial_guidance

poetry install

# Verify installation
poetry run python -c "import spatial_guidance; print('✅ Installation successful')"
```

### Option B: Pip Installation

```bash
pip install -e src/spatial_guidance

# Verify installation
python -c "import spatial_guidance; print('✅ Installation successful')"
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
│   ├── poses.txt
│   └── intrinsics.txt
├── scene2/
│   ├── rgb/
│   ├── depth/
│   ├── poses.txt
│   └── intrinsics.txt
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


### Step 3: Test Streamlit Application

```bash
# Test Streamlit app startup (without running)
streamlit run src/spatial_guidance/spatial_guidance/ui/streamlit_app_live.py --help
```
