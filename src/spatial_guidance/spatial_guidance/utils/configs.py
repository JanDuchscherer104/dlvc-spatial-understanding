import os
from pathlib import Path
from typing import Annotated, Literal, Optional, Type, Union

from dotenv import load_dotenv
from pydantic import Field, ValidationInfo, field_validator

from .base_config import SingletonConfig
from .console import Console


class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: PathConfig._detect_root())
    data: Annotated[Path, Field(default=".data/SmartAIs-Recorded-Data")]

    # data: Annotated[Path, Field(default=".data")]
    env_file: Annotated[Path, Field(default=".env")]

    @staticmethod
    def _detect_root() -> Path:
        # Use /app if running in Docker, else use repo root
        if (
            os.environ.get("IN_DOCKER") == "1"
            or Path("/.dockerenv").exists()
            or (
                Path("/proc/1/cgroup").exists()
                and "docker" in Path("/proc/1/cgroup").read_text()
            )
        ):
            return Path("/app").resolve()
        # Default: repo root (4 parents up from this file)
        return Path(__file__).parents[4].resolve()

    @field_validator("env_file", "data", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path, info: ValidationInfo) -> Path:
        CONSOLE = Console.with_prefix(
            cls.__name__, f"convert_to_path ({info.field_name})"
        )
        if isinstance(v, str):
            root = info.data.get("root", Path.cwd())
            v = root / v if not Path(v).is_absolute() else Path(v)
        v = v.resolve()

        # Check if the field name ends with "_file"
        field_name = info.field_name
        if field_name and field_name.endswith("_file"):
            if not v.exists():
                CONSOLE.warn(f"File {v} does not exist")
        else:
            v.mkdir(parents=True, exist_ok=True)

        return v

    def get_api_key(
        self, key_name: Union[Literal["GOOGLE_API_KEY"], str]
    ) -> Optional[str]:
        """Load API key from .env file

        Args:
            key_name: Name of the environment variable to retrieve

        Returns:
            The API key value or None if not found
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "get_api_key")
        env_path = self.root / self.env_file
        if not env_path.exists():
            CONSOLE.warn(f"Environment file {env_path} does not exist")
            return None
        load_dotenv(env_path)
        api_key = os.getenv(key_name)
        if api_key is None:
            CONSOLE.warn(f"API key '{key_name}' not found in {env_path}")
        else:
            CONSOLE.log(f"API key '{key_name}' loaded successfully")
        return api_key
