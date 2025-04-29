"""Base implementation of pipeline stages."""

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Type, TypeVar

from pydantic import Field, ValidationInfo, field_validator
from zenml.config import DockerSettings, ResourceSettings, StepRetryConfig
from zenml.config.docker_settings import DockerBuildConfig
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)
from zenml.steps import BaseStep
from zipp import Path

from utils import CONSOLE, BaseConfig, PathConfig

from .data_contracts import DataModel
from .docker_config import BaseDockerConfig
from .materializer import PydanticNumpyMaterializer

I = TypeVar("I", bound=DataModel)
O = TypeVar("O", bound=DataModel)
T = TypeVar("T", bound="PipelineStage")


class PipelineStageConfig(BaseDockerConfig[T]):
    """Configuration for pipeline stages.

    This configuration class defines settings for pipeline stages
    including execution parameters and resources requirements.
    """

    # Resource configuration
    resources: Optional[ResourceSettings] = Field(
        ResourceSettings(),
        description="Compute resources required for this stage",
    )

    # Environment configuration
    # docker_file: Optional[Path] = Field(default=".step-requirements/Dockerfile")

    # Execution behavior
    enable_cache: bool = Field(
        default=True, description="Enable caching for this stage"
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
            "settings": {
                k: v
                for k, v in {
                    "docker": self.docker_settings,
                    "resources": self.resources,
                    "orchestrator": self.orchestrator_settings,
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
        config = config or self._get_default_config()

        # Pass through to ZenML's BaseStep with appropriate step kwargs
        super().__init__(
            output_materializers=PydanticNumpyMaterializer,
            **config.get_step_kwargs(),
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

    @abstractmethod
    def entrypoint(self, input_data: I) -> O:
        pass
