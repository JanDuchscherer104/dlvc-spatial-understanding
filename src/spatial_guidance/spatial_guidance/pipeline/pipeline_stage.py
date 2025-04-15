"""Base implementation of pipeline stages."""

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Type, TypeVar

from pydantic import Field
from zenml.config import DockerSettings, ResourceSettings, StepRetryConfig
from zenml.steps import BaseStep

from utils import CONSOLE, BaseConfig

from .data_contracts import DataModel
from .materializer import PydanticNumpyMaterializer

I = TypeVar("I", bound=DataModel)
O = TypeVar("O", bound=DataModel)
T = TypeVar("T", bound="PipelineStage")


class PipelineStageConfig(BaseConfig[T]):
    """Configuration for pipeline stages.

    This configuration class defines settings for pipeline stages
    including execution parameters and resources requirements.
    """

    # Resource configuration
    resources: Optional[ResourceSettings] = Field(
        None,
        description="Compute resources required for this stage",
    )

    # Environment configuration
    docker_settings: Optional[DockerSettings] = Field(
        default=None, description="Docker configuration for containerized execution"
    )
    step_operator: Optional[str] = Field(
        default=None, description="Step operator to use, if any"
    )

    # Execution behavior
    enable_cache: bool = Field(
        default=False, description="Enable caching for this stage"
    )
    retry: StepRetryConfig = Field(
        default_factory=lambda: StepRetryConfig(max_retries=2, delay=5),
        description="Retry configuration for this stage",
    )

    # Callbacks
    on_failure: Optional[Callable] = Field(default=None, description="Failure callback")
    on_success: Optional[Callable] = Field(default=None, description="Success callback")

    # Logging
    verbose: bool = Field(default=True, description="Enable verbose logging")

    def get_step_kwargs(self) -> Dict[str, Any]:
        step_params = {
            "enable_cache": self.enable_cache,
            "retry": self.retry,
            "on_failure": self.on_failure,
            "on_success": self.on_success,
            "step_operator": self.step_operator,
            "settings": {
                k: v
                for k, v in {
                    "docker": self.docker_settings,
                    "resources": self.resources,
                }.items()
                if v is not None
            },
        }

        # Filter out None values
        return {k: v for k, v in step_params.items() if v is not None}


class PipelineStage(BaseStep, Generic[I, O]):
    """
    Abstract base class for all pipeline stages.

    Each stage receives an input of type I and produces an output of type O,
    where both I and O are DataModel subclasses. By inheriting from ZenML's BaseStep,
    the stage automatically integrates with ZenML's runtime without an explicit decorator.

    Derived classes must implement the entrypoint() method.
    """

    # Store a class-specific configuration - needed for ZenML compatibility
    _default_config: ClassVar[Optional[PipelineStageConfig]] = None

    def __init__(self, config: Optional[PipelineStageConfig] = None) -> None:
        """
        Initialize the stage with the given configuration.

        Args:
            config: A PipelineStageConfig instance carrying caching, docker, retry, and other settings.
            **kwargs: Optional keyword arguments passed to ZenML's BaseStep.
        """
        # Use provided config or try to find a default config for this class
        self.config = config or self._get_default_config()

        # Pass through to ZenML's BaseStep with appropriate step kwargs
        super().__init__(
            output_materializers=PydanticNumpyMaterializer,
            **self.config.get_step_kwargs(),
        )

    @classmethod
    def _get_default_config(cls) -> PipelineStageConfig:
        """
        Get default config for this class.
        Ensures that even if ZenML instantiates us directly, we have a valid config.
        """
        if cls._default_config is None:
            # Create a minimal default config for this specific class
            cls._default_config = PipelineStageConfig(target=cls)
        return cls._default_config

    @classmethod
    def register_config(cls, config: PipelineStageConfig) -> None:
        cls._default_config = config

    @abstractmethod
    def entrypoint(self, input_data: I) -> O:
        pass

    def __call__(self, input_data: I) -> O:
        # if self.config.verbose:
        #     CONSOLE.log(f"Input to [green]{self.__class__.__name__}[/green]:")
        #     CONSOLE.plog(input_data)

        output = super().__call__(input_data)

        # if self.config.verbose:
        #     CONSOLE.log(f"Output from [orange]{self.__class__.__name__}[/orange]:")
        #     CONSOLE.plog(output.model_dump())

        return output
