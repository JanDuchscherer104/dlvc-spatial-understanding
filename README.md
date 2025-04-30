# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and spatial audio rendering, the application seeks to enhance navigation and interaction in various environments.

- **3D Spatial Scene Understanding**: Utilizes depth estimation and object detection to create a comprehensive understanding of the user's surroundings.
- **Interactive Audio Guidance**: Generates spatial audio cues and spoken instructions to assist users in navigating their environment effectively.

See [Setup](SETUP.md) for installation instructions.
Briefly:
```bash
conda create -n dlvc python=3.11
conda activate dlvc

# from root of the repo
pip install -e src/spatial_guidance

make registry-init
make zenml-init
```



## Usage

- Example usage of the `StrayScanner` dataset is provided in this [notebook](notebooks/test_stray.ipynb).

1. **Pipeline Definition**
   The core orchestration lives in [pipeline.py](src/spatial_guidance/spatial_guidance/pipeline/pipeline.py):
   - `PipelineConfig` gathers per‑stage configs and global settings.
   - `SpatialUnderstandingPipeline`’s `run(...)` method chains stages in order (`dataset → detection → visualization`) using `make_step()`.

2. **Data Models**
   All data exchanged between stages are Pydantic models in [data_contracts.py](src/spatial_guidance/spatial_guidance/pipeline/data_contracts.py):
   **Note**: This is not striclty necessary.
   - `PipelineIn`, `DatasetOut`, `DetectionStageOut`, `VisualizationIn`, `VisualizationOut`, etc.

3. **Example Stage**
   A concrete detection stage is implemented in [vlm_gemini_detector.py](src/spatial_guidance/spatial_guidance/scene_understanding/vlm_gemini_detector.py):
   - `GeminiVLMDetectionConfig` holds model hyperparameters, safety settings, retry and Docker options.
   - `GeminiVLMDetection` calls Google’s Gemini API and parses results into `DetectionStageOut`.

4. **Configurable Stages**
   Per‑stage execution settings (Docker, resources, caching, retries, callbacks) live in [step_configs.py](src/spatial_guidance/spatial_guidance/pipeline/step_configs.py) and [docker_config.py](src/spatial_guidance/spatial_guidance/pipeline/docker_config.py):
   ```py
   class StepConfig(BaseConfig):
       docker_config: DockerConfig = Field(default_factory=DockerConfig)
       resources: ResourceSettings = …
       enable_cache: bool = True
       retry: StepRetryConfig = …
       …
       def get_step_kwargs(self) -> Dict[str, Any]: …
   ```
   You instantiate a config, override fields, then call `config.setup_target()` to get your `BaseStep`.

5. **Simple Decorated Steps**
   For lightweight steps that don’t need full `StepConfig`, you can just use ZenML’s decorator:
   ```python
   from zenml import step
   @step
   def get_visualization_in(dataset_out, detection_output) -> VisualizationIn:
       return VisualizationIn(...)
   ```
   See visualization_in.py.

6. **Quickstart Example**
   A minimal run is shown in test_pipeline.py:
   ```python
   from spatial_guidance import PipelineConfig
   pipeline = PipelineConfig(stack="local_docker_stack").setup_target()
   out = pipeline(42, None)
   print(pipeline.get_output_of_stage("visualization"))
   out.visualization.show()
   ```