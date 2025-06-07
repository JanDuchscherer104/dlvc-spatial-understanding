import os
from typing import Callable, Dict, List, Literal, Optional, Self, Type, Union

from pydantic import Field, model_validator
from zenml.client import Client
from zenml.materializers.materializer_registry import materializer_registry
from zenml.pipelines.pipeline_definition import Pipeline
from zenml.steps import BaseStep

from ..data_contracts import DataModel
from ..data_contracts.aabb_segmentation import AABBDetections
from ..data_contracts.dataset import DatasetOut, PipelineIn
from ..data_handling.stray_scanner.stray_dataset import StrayDatasetConfig
from ..pipeline.docker_config import DockerConfig
from ..scene_understanding.gemini_aabb_detection import GeminiAABBDetSeg
from ..scene_understanding.gemini_scene_descriptor import (
    GeminiSceneDescriptor,
    GeminiSceneDescriptorConfig,
)
from ..utils.base_config import BaseConfig
from ..utils.console import Console
from .materializer import PydanticNumpyMaterializer
from .step_configs import GeminiStepConfig, StepConfig


class PipelineStepSpec(BaseConfig):
    target: Union[BaseConfig, Callable]
    step_config: Optional[StepConfig] = None


class PipelineConfig(BaseConfig["SpatialUnderstandingPipeline"]):
    steps: Dict[str, PipelineStepSpec] = Field(
        default_factory=lambda: {
            "dataset": PipelineStepSpec(
                target=StrayDatasetConfig(),
            ),
            "detection": PipelineStepSpec(
                target=GeminiAABBDetSeg(),
                step_config=GeminiStepConfig(enable_cache=True),
            ),
            "scene_description": PipelineStepSpec(
                target=GeminiSceneDescriptorConfig(),
                step_config=GeminiStepConfig(enable_cache=False),
            ),
        }
    )

    global_docker_config: DockerConfig = Field(default_factory=DockerConfig)

    is_debug: bool = False
    verbose: bool = Field(default=True, description="Enable verbose logging")
    show_timestamps: bool = Field(default=False, description="Show timestamps in logs")

    # Global execution settings
    stack: Literal["default", "local_docker_stack"] = Field(
        default="local_docker_stack", description="Stack to use for pipeline execution"
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

    target: Type["SpatialUnderstandingPipeline"] = Field(
        default_factory=lambda: SpatialUnderstandingPipeline,
        description="Pipeline target class",
    )

    @model_validator(mode="after")
    def validate_stack_name(self) -> Self:
        CONSOLE = Console.with_prefix(self.__class__.__name__, "validate_stack_name")
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
        CONSOLE = Console.with_prefix(self.__class__.__name__, "setup_target")
        CONSOLE.set_verbose(self.verbose)
        CONSOLE.set_timestamp_display(self.show_timestamps)

        materializer_registry.register_and_overwrite_type(
            key=DataModel, type_=PydanticNumpyMaterializer
        )

        if self.is_debug:
            # Set zenml debug mode export ZENML_DEBUG=true
            CONSOLE.log("Debug mode is enabled")
            os.environ["ZENML_DEBUG"] = "true"
            os.environ["ZENML_DEBUG_LOG_LEVEL"] = "DEBUG"

        os.environ["ZENML_STORE_BACKUP_STRATEGY"] = "disabled"
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

        settings = {}

        if self.global_docker_config.docker_settings is not None:
            settings["docker"] = self.global_docker_config.docker_settings
        if self.global_docker_config.orchestrator_settings is not None:
            settings["orchestrator"] = self.global_docker_config.orchestrator_settings

        params = {
            "name": self.target.__name__,
            "enable_cache": self.enable_cache_global,
            "enable_artifact_metadata": self.enable_artifact_metadata_global,
            "on_failure": self.on_failure_global,
            "on_success": self.on_success_global,
            "settings": settings,
        }

        return {k: v for k, v in params.items() if v is not None}


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

        self.dataset = self.make_step(self.config.steps["dataset"])
        self.detection_stage = self.make_step(self.config.steps["detection"])
        # self.obb_detection_stage = self.make_step(self.config.steps["obb_detection"])
        # self.visualization_stage = self.make_step(self.config.steps["visualization"])
        # self.get_visualization_in = self.make_step(
        #     self.config.steps["get_visualization_in"]
        # )
        self.scene_description_stage = self.make_step(
            self.config.steps["scene_description"]
        )

        super().__init__(**self.config.get_pipeline_kwargs(), entrypoint=self.run)

    def run(self, idx: int, user_prompt: Optional[str] = None) -> AABBDetections:
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
        CONSOLE = Console.with_prefix(self.__class__.__name__, "run")
        # Create initial pipeline input
        input_data = PipelineIn(idx=idx, user_prompt=user_prompt)
        dataset_out = self.dataset(input_data)  # type: DatasetOut

        # Run AABB and OBB detection.
        # ZenML can run these in parallel if their inputs are ready and they don't depend on each other.
        aabb_detection_output = self.detection_stage(
            dataset_out
        )  # type: AABBDetections
        # obb_detection_output = self.obb_detection_stage(
        #     dataset_out
        # )  # type: OBBDetections

        scene_description = self.scene_description_stage(
            dataset_out, aabb_detection_output
        )

        CONSOLE.log(f"[green]Scene description generated: {scene_description}[/green]")

        # visualization_in = self.get_visualization_in(
        #     dataset_out=dataset_out,
        #     aabb_detection_output=aabb_detection_output,
        #     obb_detection_output=obb_detection_output,
        # )  # type: VisualizationIn
        # final_output = self.visualization_stage(
        #     visualization_in
        # )  # type: VisualizationOut

        return aabb_detection_output

    @staticmethod
    def make_step(
        spec: "PipelineStepSpec",
    ) -> BaseStep:
        kwargs = spec.step_config.get_step_kwargs() if spec.step_config else {}
        if isinstance(spec.target, BaseConfig):
            target = spec.target.setup_target(**kwargs)
        else:
            target = spec.target
        return target

    @staticmethod
    def get_latest_output() -> AABBDetections:
        """
        Loads the output of the latest run of this pipeline as a Python object.

        Returns:
            The VisualizationOutput object produced by the final step.
        """
        client = Client()
        pipeline_model = client.get_pipeline(SpatialUnderstandingPipeline.__name__)
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

    @staticmethod
    def get_stage_names(run_idx: int = 0) -> List[str]:
        run = Client().get_pipeline(SpatialUnderstandingPipeline.__name__).runs[run_idx]
        if run is None:
            raise RuntimeError("No runs found for this pipeline.")

        return list(run.steps.keys())

    @staticmethod
    def get_output_of_stage(stage_name: str, run_idx: int = 0) -> DataModel:
        """
        Get the output of a specific stage in the pipeline.

        Args:
            stage_name: The name of the stage to retrieve output from.
            run_idx: The index of the run to retrieve output from.

        Returns:
            The output of the specified stage.
        """
        run = Client().get_pipeline(SpatialUnderstandingPipeline.__name__).runs[run_idx]
        if run is None:
            raise RuntimeError("No runs found for this pipeline.")

        step = run.steps.get(stage_name)
        if step is None:
            raise RuntimeError(f"No step found with name '{stage_name}'.")

        return step.output.load()

    @staticmethod
    def get_all_outputs(run_idx: int = 0) -> Dict[str, DataModel]:
        """
        Get all outputs of the pipeline run.

        Args:
            run_idx: The index of the run to retrieve outputs from.

        Returns:
            A dictionary mapping stage names to their outputs.
        """
        run = Client().get_pipeline(SpatialUnderstandingPipeline.__name__).runs[run_idx]
        if run is None:
            raise RuntimeError("No runs found for this pipeline.")

        return {step.name: step.output.load() for step in run.steps.values()}
