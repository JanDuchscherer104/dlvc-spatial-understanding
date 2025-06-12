Please help me restructuring our pipeline concept to make it less error prone and more aligned with ZenMLs best practices.

- We are currently using our Config-as-Factory pattern to create each target within the pipeline. A problem here is that the step specific configurations (like Docker Settings, Resource Settings, ...) are included in these configs together with the component specific settings. The former are needed to create the pipeline, and are causing issues within the Docker containers as for example the Dockerfiles are not available in the container. So we need to make sure that the step specific settings (basically all the settings that are configured in the `PipelineStageConfig`) are not included in the configs that are being used within the Docker containers! If possible I would like to stick with the current approach where we derive our Pipeline from `zenml.pipelines.pipeline_definition.Pipeline` and our PipelineStages `from zenml.steps.BaseStep` instead of using the `step` and `pipeline` decorators.
- Our local source code is currently distributed into two Python packages (spatial_guidance and utils). Utils defines our PathConfig, BaseConfig and Console. Both packages are currently being installed in the Dockerfile via
`RUN pip install --no-cache-dir -e src/utils -e src/spatial_guidance`. This is not ideal, it would be simpler if we could just use them via a correct Python path. The current approach often caused issues when running the pipeline as we would get import errors when trying to import objects from the utils package. This might have been due to a namespace issue (import via python path vs. import via the installed package).

For a general overview of our current repository structure, please consider the following tree:

❯ tree
├── spatial_guidance
│   ├── requirements.txt
│   ├── setup.py
│   ├── spatial_guidance
│   │   ├── __init__.py
│   │   ├── audio_guidance
│   │   │   ├── __init__.py
│   │   │   ├── audio_generator.py
│   │   │   └── stt_engine.py
│   │   ├── data_handling
│   │   │   ├── __init__.py
│   │   │   └── stray_scanner
│   │   │       ├── __init__.py
│   │   │       ├── data_parser.py
│   │   │       ├── stray_dataset.py
│   │   │       └── stray_scanner_paths.py
│   │   ├── pipeline
│   │   │   ├── __init__.py
│   │   │   ├── data_contracts.py
│   │   │   ├── docker_config.py
│   │   │   ├── materializer.py
│   │   │   ├── pipeline.py
│   │   │   └── pipeline_stage.py
│   │   ├── scene_understanding
│   │   │   ├── __init__.py
│   │   │   ├── depth_estimation.py
│   │   │   └── vlm_gemini_detector.py
│   │   └── visualization
│   │       ├── __init__.py
│   │       └── scene_visualizer.py
│   └── spatial_guidance.egg-info
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       └── top_level.txt
└── utils
    ├── setup.py
    └── utils
        ├── __init__.py
        ├── configs.py
        └── utils.py

Furthermore you are provided with the source code of the most relevant files in the pipeline:

- `pipeline.py`
- `pipeline_stage.py`
- `docker_config.py`
- `vlm_gemini_detector.py` (as an example of a concrete pipeline stage)

For further reference, please consider the following ZenML files:

- `docker_settings.py`
- `pipeline_definition.py`
- `base_step.py`

Start by reflecting about elegant ways to resolve our issues and the provide a solution that is aligned with ZenML best practices.
