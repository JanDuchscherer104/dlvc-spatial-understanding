"""Base implementation of pipeline stages."""

from typing import Any, Callable, Dict, Optional

from pydantic import Field
from zenml.config import ResourceSettings, StepRetryConfig

from ..utils import BaseConfig
from .docker_config import DockerConfig, GeminiDockerConfig


class StepConfig(BaseConfig):
    """Configuration for pipeline stages.

    This configuration class defines settings for pipeline stages
    including execution parameters and resources requirements.
    """

    # Resource configuration
    resources: ResourceSettings = Field(
        default_factory=ResourceSettings,
        description="Compute resources required for this stage",
    )

    # Docker configuration
    docker_config: DockerConfig = Field(default_factory=DockerConfig)

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
        settings = {}
        if self.docker_config.docker_settings is not None:
            settings["docker"] = self.docker_config.docker_settings
        if self.resources is not None:
            settings["resources"] = self.resources
        if self.docker_config.orchestrator_settings is not None:
            settings["orchestrator"] = self.docker_config.orchestrator_settings

        step_params = {
            "enable_cache": self.enable_cache,
            # "output_materializers": self.output_materializers,
            "retry": self.retry,
            "on_failure": self.on_failure,
            "on_success": self.on_success,
            "settings": settings,
        }

        # Filter out None values at top level
        return {k: v for k, v in step_params.items() if v is not None}


class GeminiStepConfig(StepConfig):
    """Configuration for Gemini pipeline stages.

    This configuration class extends the base StepConfig to include
    specific settings for the Gemini model.
    """

    docker_config: GeminiDockerConfig = Field(default_factory=GeminiDockerConfig)
    retry: StepRetryConfig = Field(
        default_factory=lambda: StepRetryConfig(max_retries=4, delay=10),
        description="Retry configuration for this stage",
    )
