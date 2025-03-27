import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Literal, Optional, Type, Union

from dotenv import load_dotenv
from pydantic import Field, ValidationInfo, field_validator

from .utils import CONSOLE, SingletonConfig


class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[3].resolve())
    target: Type["PathConfig"] = Field(default_factory=lambda: PathConfig)

    data: Annotated[Path, Field(default=".data")]
    env_file: Annotated[Path, Field(default=".env")]

    @field_validator("data", "env_file", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path, info: ValidationInfo) -> Path:
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
