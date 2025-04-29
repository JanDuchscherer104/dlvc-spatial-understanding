from pathlib import Path
from typing import Any, TypeVar

from pydantic import Field, ValidationInfo, field_validator
from zenml.config.docker_settings import DockerSettings
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

from utils import BaseConfig, PathConfig

T = TypeVar("T")


class BaseDockerConfig(BaseConfig[T]):
    """Common Docker + orchestrator settings."""

    docker_file: Path = Field(
        default=PathConfig().root / ".step-requirements" / "Dockerfile",
        description="Path to the Dockerfile",
    )
    docker_settings: DockerSettings = Field(
        default_factory=lambda: DockerSettings(
            build_context_root=PathConfig().root.as_posix(),
            install_stack_requirements=False,
            environment={"PYTHONPATH": "/app/src"},
        ),
    )
    orchestrator_settings: LocalDockerOrchestratorSettings = Field(
        default_factory=lambda: LocalDockerOrchestratorSettings(
            run_args={
                "volumes": {
                    f"{(PathConfig().root / '.data').as_posix()}": {
                        "bind": "/app/.data",
                        "mode": "rw",
                    },
                    f"{(PathConfig().root / '.env').as_posix()}": {
                        "bind": "/app/.env",
                        "mode": "ro",
                    },
                }
            }
        ),
        description="Local orchestrator settings with mounted volumes",
    )

    @field_validator("docker_settings", mode="before")
    @classmethod
    def _validate_and_inject_docker(
        cls, v: Any, info: ValidationInfo
    ) -> DockerSettings:
        """Ensure the Dockerfile exists and build a matching DockerSettings."""
        df = info.data.get("docker_file")
        assert isinstance(df, Path), "docker_file must be a Path"
        assert df.exists(), f"Dockerfile not found at {df}"
        return DockerSettings(
            dockerfile=df.as_posix(),
            build_context_root=PathConfig().root.as_posix(),
            install_stack_requirements=False,
            environment={"PYTHONPATH": "/app/src"},
        )
