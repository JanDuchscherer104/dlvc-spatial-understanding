import os
from typing import Callable, Dict, List, Literal, Optional, Self, Type

from pydantic import Field, model_validator
from zenml import step
from zenml.client import Client
from zenml.pipelines.pipeline_definition import Pipeline

from utils import CONSOLE

from ..data_handling.stray_scanner.stray_dataset import StrayDatasetConfig
from ..scene_understanding.segmentation_model import SegmentAnythingConfig
from ..scene_understanding.vlm_gemini_detector import GeminiVLMDetectionConfig
from ..visualization.scene_visualizer import SceneVisualizerConfig
from .data_contracts import (
    DataModel,
    DataSetOut,
    DetectionStageOut,
    PipelineIn,
    VisualizationIn,
    VisualizationOut,
)
from .docker_config import BaseDockerConfig


class PipelineConfig(BaseDockerConfig["SpatialUnderstandingPipeline"]):
    dataset_config: StrayDatasetConfig = Field(default_factory=StrayDatasetConfig)
    detection_config: GeminiVLMDetectionConfig = Field(
        default_factory=GeminiVLMDetectionConfig,
        description="Configuration for detection stage",
    )
    segmentation_config: SegmentAnythingConfig = Field(
        default_factory=SegmentAnythingConfig,
        description="Configuration for segmentation stage",
    )
    visualization_config: SceneVisualizerConfig = Field(
        default_factory=SceneVisualizerConfig
    )

    is_debug: bool = False
    verbose: bool = Field(default=True, description="Enable verbose logging")
    show_timestamps: bool = Field(default=False, description="Show timestamps in logs")

    # Global execution settings
    stack: Literal["default", "local_docker_stack"] = Field(
        default="default", description="Stack to use for pipeline execution"
    )
    enable_cache_global: bool = Field(
        default=False, description="Enable caching for pipeline steps"
    )
    enable_artifact_metadata_global: bool = Field(
        default=True, description="Enable artifact metadata"
    )
    on_failure_global: Optional[Callable] = Field(
        default=None, description="Failure callback"
    )
    on_success_global: Optional[Callable] = Field(
        default=None, description="Success callback"
    )

    target: Type["SpatialUnderstandingPipeline"] = Field(
        default_factory=lambda: SpatialUnderstandingPipeline,
        description="Pipeline target class",
    )

    @model_validator(mode="after")
    def validate_stack_name(self) -> Self:
        try:
            client = Client()
            client.activate_stack(stack_name_id_or_prefix=self.stack)
            CONSOLE.log(
                f"[green]Activated ZenML stack: [bold]{self.stack}[/bold][/green]"
            )
        except Exception as e:
            # Skip stack activation failures (e.g., during testing)
            CONSOLE.warn(f"Could not activate ZenML stack '{self.stack}': {e}")
        return self

    def setup_target(self) -> "SpatialUnderstandingPipeline":
        CONSOLE.set_verbose(self.verbose)
        CONSOLE.set_timestamp_display(self.show_timestamps)

        if self.is_debug:
            # Set zenml debug mode export ZENML_DEBUG=true
            CONSOLE.log("Debug mode is enabled")
            os.environ["ZENML_DEBUG"] = "true"
            os.environ["ZENML_DEBUG_LOG_LEVEL"] = "DEBUG"

        return self.target(self)

    def get_pipeline_kwargs(self) -> Dict:
        """
        Get the pipeline configuration as a dictionary.

        Returns:
            dict: The pipeline configuration.

        Args:
            name: str,
            entrypoint: F,
            enable_cache: Optional[bool] = None,
            enable_artifact_metadata: Optional[bool] = None,
            enable_artifact_visualization: Optional[bool] = None,
            enable_step_logs: Optional[bool] = None,
            settings: Optional[Mapping[str, "SettingsOrDict"]] = None,
            tags: Optional[List[Union[str, "Tag"]]] = None,
            extra: Optional[Dict[str, Any]] = None,
            on_failure: Optional["HookSpecification"] = None,
            on_success: Optional["HookSpecification"] = None,
            model: Optional["Model"] = None,
            substitutions: Optional[Dict[str, str]] = None,
        """
        return {
            "name": self.target.__name__,
            "enable_cache": self.enable_cache_global,
            "enable_artifact_metadata": self.enable_artifact_metadata_global,
            "on_failure": self.on_failure_global,
            "on_success": self.on_success_global,
            "settings": {
                "docker": self.docker_settings,
            },
        }


class SpatialUnderstandingPipeline(Pipeline):
    """
    Main pipeline for 3D scene understanding using modular, DataModel-based stages.

    This pipeline subclasses ZenML's Pipeline (from zenml.pipelines.pipeline_definition)
    so that it integrates with ZenML without using the @pipeline decorator. Instead,
    the pipeline's entrypoint() method is implemented by chaining our custom stages,
    each constructed via its PipelineStageConfig.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline using the top-level configuration.

        Global logging parameters are set, and each stage is instantiated by calling
        its respective configuration's setup_target() method. These stages are derived
        from our custom PipelineStage and enforce strong typing.
        """
        self.config = config

        super().__init__(**self.config.get_pipeline_kwargs(), entrypoint=self.run)

    def run(self, idx: int, user_prompt: Optional[str] = None) -> VisualizationOut:
        """
        Pipeline entrypoint function.

        This method is invoked when the pipeline is run. It creates the necessary input
        DataModels (via the dataset stage), then sequentially passes the data through the
        detection stage and the visualization stage (and optionally segmentation), ultimately
        returning the final output.

        Args:
            idx: Index of the data sample to process.
            user_prompt: Optional user query string.

        Returns:
            The final pipeline output.
        """

        self.dataset = self.config.dataset_config.setup_target()
        self.detection_stage = self.config.detection_config.setup_target()
        self.visualization_stage = self.config.visualization_config.setup_target()

        # Create initial pipeline input
        input_data = PipelineIn(idx=idx, user_prompt=user_prompt)
        dataset_out = self.dataset(input_data)
        detection_output = self.detection_stage(dataset_out)
        visualization_in = get_visualization_in(
            dataset_out=dataset_out, detection_output=detection_output
        )
        final_output = self.visualization_stage(visualization_in)

        return final_output

    @classmethod
    def get_latest_output(cls) -> VisualizationOut:
        """
        Loads the output of the latest run of this pipeline as a Python object.

        Returns:
            The VisualizationOutput object produced by the final step.
        """
        client = Client()
        pipeline_model = client.get_pipeline(cls.__name__)
        last_run = pipeline_model.last_run
        if last_run is None:
            raise RuntimeError("No runs found for this pipeline.")

        # Find the final step (usually the last in the steps dict)
        steps = last_run.steps
        if not steps:
            raise RuntimeError("No steps found in the latest run.")

        final_step = list(steps.values())[-1]
        # If your step has only one output, use .output
        output = final_step.output.load()
        return output

    @classmethod
    def get_names(cls, run_idx: int = -1) -> Dict[str, List[str]]:
        run = Client().get_pipeline(cls.__name__).runs[run_idx]
        if run is None:
            raise RuntimeError("No runs found for this pipeline.")

        return {run.name: list(run.steps.keys())}

    @classmethod
    def get_output_of_stage(cls, stage_name: str, run_idx: int = -1) -> DataModel:
        """
        Get the output of a specific stage in the pipeline.

        Args:
            stage_name: The name of the stage to retrieve output from.
            run_idx: The index of the run to retrieve output from.

        Returns:
            The output of the specified stage.
        """
        run = Client().get_pipeline(cls.__name__).runs[run_idx]
        if run is None:
            raise RuntimeError("No runs found for this pipeline.")

        step = run.steps.get(stage_name)
        if step is None:
            raise RuntimeError(f"No step found with name '{stage_name}'.")

        return step.output.load()


@step
def get_visualization_in(
    dataset_out: DataSetOut, detection_output: DetectionStageOut
) -> VisualizationIn:
    """
    Create the visualization input from dataset and detection outputs.

    Args:
        dataset_out: Output from the dataset stage.
        detection_output: Output from the detection stage.

    Returns:
        VisualizationIn: The combined input for the visualization stage.
    """
    return VisualizationIn(
        rgb_image=dataset_out.rgb_image,
        depth_image=dataset_out.depth_image,
        detection_output=detection_output,
    )
