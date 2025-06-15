import dis
from typing import Annotated, Any, List, Literal, Self, Type
from urllib import response

import pyaudio
from google.genai import types
from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..data_handling.stray_scanner.stray_dataset import StrayDatasetConfig
from ..scene_understanding.gemini_aabb_detection import GeminiAABBDetSegConfig
from ..utils import BaseConfig, Console, PathConfig
from .gemini_live_agent import GeminiLiveAgent
from .live_agent_enums import (
    DirectionalStyle,
    DistanceStyle,
    InteractionMode,
    ResponseStyle,
)
from .prompt_templates import LiveAgentPromptTemplates
from .tools import LiveAgentTools

MODEL_OPTIONS: dict[str, str] = {
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview(05-20) - adaptive thinking, cost-efficient",
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro Preview (05-06) - enhanced reasoning, multimodal",
}


class GeminiLiveAgentConfig(BaseConfig["GeminiLiveAgent"]):
    target: Type["GeminiLiveAgent"] = Field(default_factory=lambda: GeminiLiveAgent)

    is_debug: bool = True
    """Verbose debug logging."""

    # Dataset configuration
    dataset: StrayDatasetConfig = Field(default_factory=StrayDatasetConfig)

    # Expert Models
    gemini_aabb_detseg: GeminiAABBDetSegConfig = GeminiAABBDetSegConfig(
        visualize_rgb=True
    )
    num_detseg_attempts: int = 3

    # Tools
    enable_code_execution: bool = True
    """Enable code execution tool via _ExecAPI."""
    tools: Annotated[List[types.Tool], Field(default_factory=lambda: LiveAgentTools())]

    # Live API model configuration
    interaction_mode: InteractionMode = InteractionMode.TEXT
    response_style: ResponseStyle = ResponseStyle(
        dir_style=DirectionalStyle.CLOCK_FACE, dist_style=DistanceStyle.PRECISE
    )
    live_model: Literal["gemini-2.0-flash-live-001"] = "gemini-2.0-flash-live-001"
    live_model_config: types.LiveConnectConfig = Field(
        default_factory=lambda: types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
            # Low temperature minimises drift; modest nucleus & k keep phrasing varied without schema errors.
            temperature=0.1,
            top_p=0.9,
            top_k=64,
            system_instruction=None,
            tools=None,
        )
    )
    http_options: types.HttpOptions = Field(
        default_factory=lambda: types.HttpOptions(api_version="v1beta")
    )

    # Audio configuration
    format: int = pyaudio.paInt16
    channels: int = 1
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    chunk_size: int = 1024

    system_instruction_template: Type[LiveAgentPromptTemplates] = Field(
        default_factory=lambda: LiveAgentPromptTemplates
    )
    system_instruction: Annotated[str, Field(default=LiveAgentPromptTemplates)]

    @field_validator("tools", mode="before")
    @classmethod
    def make_tools(
        cls, tools: LiveAgentTools, info: ValidationInfo
    ) -> list[types.Tool]:
        """Create a default live model configuration."""
        # Use the new tools - will be updated later with actual styles
        return tools.setup_target(has_code_execution=info.data["enable_code_execution"])  # type: ignore

    @field_validator("system_instruction", mode="before")
    @classmethod
    def make_system_instruction(
        cls, prompt_cls: LiveAgentPromptTemplates, info: ValidationInfo
    ) -> str:
        """Create a default live model configuration."""
        # Use the new prompt template - will be updated later with actual styles
        return prompt_cls.make_prompt(response_style=info.data["response_style"], enable_code_execution=info.data.get("enable_code_execution", False))  # type: ignore

    @model_validator(mode="after")
    def validate_live_model_config(self) -> "GeminiLiveAgentConfig":
        """Add tools to the live model configuration and set system instruction."""

        self.live_model_config.tools = self.tools
        self.live_model_config.system_instruction = self.system_instruction

        params_to_log = {
            "live_model": self.live_model,
            "interaction_mode": self.interaction_mode,
            "response_modalities": self.live_model_config.response_modalities,
            "tools": LiveAgentTools.make_info(self.tools),
            "system_instruction": self.live_model_config.system_instruction,
        }
        console = Console.with_prefix(self.__class__.__name__)
        console.log(f"Live model configuration:")
        console.plog(params_to_log, title="Live Model Parameters")

        return self

    @model_validator(mode="after")
    def share_fields(self) -> Self:
        self.gemini_aabb_detseg.is_debug = self.is_debug

        return self
