from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field, ValidationInfo, field_validator
from zenml.config.docker_settings import DockerBuildConfig, DockerSettings
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

from ..utils import BaseConfig, PathConfig


class DockerConfig(BaseConfig):
    """Common Docker + orchestrator settings."""

    docker_file: Optional[Annotated[Path, str]] = Field(
        # If building from a Dockerfile, set this to the path of the Dockerfile!
        # default=".step-requirements/Dockerfile",
        None,
        description="Path to the Dockerfile",
    )
    docker_settings: DockerSettings = Field(
        default_factory=lambda: DockerSettings(
            parent_image="zenmldocker/zenml:0.81.0-py3.11",
            apt_packages=["ffmpeg", "libsm6", "libxext6", "libgl1-mesa-glx"],
            requirements=(
                PathConfig().root / ".step-requirements" / "requirements.txt"
            ).as_posix(),
            environment={"PYTHONPATH": "/app/src"},
            build_context_root=PathConfig().root.as_posix(),
            # install_stack_requirements=False,
            # parent_image_build_config=DockerBuildConfig(build_options={"pull": False}),
            # build_config=DockerBuildConfig(build_options={"pull": False}),
            dockerignore=(PathConfig().root / ".dockerignore").as_posix(),
        )
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

    @field_validator("docker_file", mode="before")
    @classmethod
    def _validate_docker_file(
        cls, v: Optional[Annotated[Path, str]], info: ValidationInfo
    ) -> Optional[Annotated[Path, str]]:
        """Ensure the Dockerfile exists."""
        if v is None:
            return v
        path = PathConfig().root / v
        assert path.exists(), f"Dockerfile {path} does not exist"
        return path

    @field_validator("docker_settings", mode="before")
    @classmethod
    def _validate_and_inject_docker(
        cls, v: DockerSettings, info: ValidationInfo
    ) -> DockerSettings:
        """Ensure the Dockerfile exists and build a matching DockerSettings."""
        df = info.data.get("docker_file")
        if df is None or not df.exists():
            return v
        kwargs = v.model_dump()
        kwargs["dockerfile"] = df.as_posix()

        return DockerSettings(**kwargs)


class GeminiDockerConfig(DockerConfig):
    """Same as DockerConfig, but injects extra Gemini requirements."""

    docker_settings: DockerSettings = Field(
        default_factory=lambda: (
            # Grab the base DockerSettings instance
            lambda base: DockerSettings(
                **{
                    **base.model_dump(),  # all existing settings
                    "requirements": (
                        PathConfig().root
                        / ".step-requirements"
                        / "requirements-gemini.txt"
                    ).as_posix(),
                }
            )
        )(
            DockerConfig().docker_settings
        )  # instantiate a default DockerConfig
    )
