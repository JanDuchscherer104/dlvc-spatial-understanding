from typing import Any, Callable, Optional, Self, Type

import numpy as np
from pydantic import Field, model_validator
from zenml import step
from zenml.client import Client
from zenml.pipelines.pipeline_definition import Pipeline

from utils import CONSOLE, BaseConfig

from ..data_handling.stray_scanner.stray_dataset import StrayDatasetConfig
from ..scene_understanding.segmentation_model import SegmentAnythingConfig
from ..scene_understanding.vlm_gemini_detector import GeminiVLMDetectionConfig
from ..visualization.scene_visualizer import SceneVisualizerConfig
from .data_contracts import PipelineIn, VisualizationIn, VisualizationOutput


class PipelineConfig(BaseConfig["SpatialUnderstandingPipeline"]):
    """
    Top-level configuration for the spatial understanding pipeline.

    This configuration wraps the configurations of all individual pipeline stages
    (e.g. dataset, detection, segmentation, visualization) as well as global options
    (stack, caching, logging, etc.).
    """

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
    # Add additional stage configs as needed

    # Global execution settings
    stack: str = Field(
        default="default", description="Stack to use for pipeline execution"
    )
    enable_cache_global: bool = Field(
        default=True, description="Enable caching for pipeline steps"
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
    verbose: bool = Field(default=True, description="Enable verbose logging")
    show_timestamps: bool = Field(default=False, description="Show timestamps in logs")
    target: Type["SpatialUnderstandingPipeline"] = Field(
        default_factory=lambda: SpatialUnderstandingPipeline,
        description="Pipeline target class",
    )

    @model_validator(mode="after")
    def validate_stack_name(self) -> Self:
        client = Client()
        try:
            client.activate_stack(stack_name_id_or_prefix=self.stack)
            CONSOLE.log(
                f"[green]Activated ZenML stack: [bold]{self.stack}[/bold][/green]"
            )
        except KeyError:
            CONSOLE.rule("[bold red]Invalid Stack Selection[/bold red]")
            CONSOLE.error(f"The specified stack '{self.stack}' does not exist.")
            raise ValueError(f"ZenML stack '{self.stack}' not found.")
        return self

    def setup_target(self) -> "SpatialUnderstandingPipeline":
        CONSOLE.set_verbose(self.verbose)
        CONSOLE.set_timestamp_display(self.show_timestamps)
        return self.target(self)

    def get_pipeline_kwargs(self) -> dict:
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

        # Instantiate stages via their configurations.
        self.dataset = self.config.dataset_config.setup_target()
        self.detection_stage = self.config.detection_config.setup_target()
        self.visualization_stage = self.config.visualization_config.setup_target()

        super().__init__(**self.config.get_pipeline_kwargs(), entrypoint=self.run)

    def run(self, idx: int, user_prompt: Optional[str] = None) -> VisualizationOutput:
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

        # Create initial pipeline input
        input_data = PipelineIn(idx=idx, user_prompt=user_prompt)
        dataset_out = self.dataset(input_data)
        detection_output = self.detection_stage(dataset_out)
        visualization_in = self.get_visualization_in(
            dataset_out=dataset_out, detection_output=detection_output
        )

        final_output = self.visualization_stage(visualization_in)

        return final_output

    @staticmethod
    @step
    def get_visualization_in(
        dataset_out: Any, detection_output: Any
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

    @classmethod
    def get_latest_output(cls) -> VisualizationOutput:
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
