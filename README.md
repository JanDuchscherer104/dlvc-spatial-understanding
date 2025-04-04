# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and spatial audio rendering, the application seeks to enhance navigation and interaction in various environments.

- **3D Spatial Scene Understanding**: Utilizes depth estimation and object detection to create a comprehensive understanding of the user's surroundings.
- **Interactive Audio Guidance**: Generates spatial audio cues and spoken instructions to assist users in navigating their environment effectively.

## Installation

```bash
conda create -n dlvc python=3.11
conda activate dlvc

conda install -c conda-forge open3d

git clone <repository-url>
cd dlvc-04-spatial-understanding
pip install -r src/spatial_guidance/requirements.txt

pip install -e src/utils
pip install -e src/spatial_guidance
```

[Create a Gemini API key](https://aistudio.google.com/app/apikey) and add it to the `.env` file.
[Create an OpenAI API key](https://platform.openai.com/api-keys) and add it to the `.env` file.

```bash
GOOGLE_API_KEY=<your-api-key>
OPENAI_API_KEY=sk-<...>
```

## Folder Structure

## Usage
