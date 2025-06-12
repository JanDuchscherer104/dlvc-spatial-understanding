from typing import Annotated, List, Literal, Type

import pyaudio
from google.genai import types
from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..data_handling.stray_scanner.stray_dataset import StrayDatasetConfig
from ..scene_understanding.gemini_aabb_detection import GeminiAABBDetSegConfig
from ..utils import BaseConfig, Console, PathConfig
from .gemini_live_agent import GeminiLiveAgent
from .live_agent_enums import InteractionMode
from .prompt_templates import SYS_PROMPT_TEMPLATE

MODEL_OPTIONS: dict[str, str] = {
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview(05-20) - adaptive thinking, cost-efficient",
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro Preview (05-06) - enhanced reasoning, multimodal",
}


class GeminiLiveAgentConfig(BaseConfig["GeminiLiveAgent"]):
    SYSTEM_PROMPT_TEMPLATE: str = Field(default=SYS_PROMPT_TEMPLATE)

    target: Type["GeminiLiveAgent"] = Field(default_factory=lambda: GeminiLiveAgent)

    is_debug: bool = True
    """Verbose debug logging."""

    # Dataset configuration
    dataset: StrayDatasetConfig = Field(default_factory=StrayDatasetConfig)

    # Expert Models
    aabb_detseg: GeminiAABBDetSegConfig = GeminiAABBDetSegConfig()
    num_detseg_attemts: int = 3

    # Tools
    tools: Annotated[List[types.Tool], Field(None)]

    # Live API model configuration
    interaction_mode: InteractionMode = InteractionMode.TEXT
    live_model: Literal["gemini-2.0-flash-live-001"] = "gemini-2.0-flash-live-001"
    live_model_config: types.LiveConnectConfig = Field(
        default_factory=lambda: types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
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

    @field_validator("tools", mode="before")
    @classmethod
    def make_tools(cls, _, info: ValidationInfo) -> List[types.Tool]:
        tools = []
        aabb_detseg_config = info.data.get("aabb_detseg")
        assert isinstance(aabb_detseg_config, GeminiAABBDetSegConfig)

        # 1) cache lookup – should be called first
        tools.append(
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="get_last_detections",
                        description="Return cached detections for this frame so the model can avoid re‑running detection.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "frame_idx": {"type": "integer"},
                                "labels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["frame_idx", "labels"],
                        },
                    )
                ]
            )
        )
        # 2) expensive detector
        tools.append(aabb_detseg_config.make_tool())
        # 3) optional code execution (fallback only)
        tools.append(types.Tool(code_execution=types.ToolCodeExecution))

        return tools

    @model_validator(mode="after")
    def validate_live_model_config(self) -> "GeminiLiveAgentConfig":
        """Add tools to the live model configuration and set system instruction."""
        self.live_model_config.tools = self.tools

        # Use the new prompt template - will be updated later with actual styles
        self.live_model_config.system_instruction = SYS_PROMPT_TEMPLATE.format(
            DIR_STYLE="Express direction as relative positioning.",
            DIST_STYLE="Give distances in approximate metres.",
        )

        params_to_log = {
            "live_model": self.live_model,
            "interaction_mode": self.interaction_mode,
            "response_modalities": self.live_model_config.response_modalities,
            "tools": self.tools,
        }
        console = Console.with_prefix(self.__class__.__name__)
        console.log(f"Live model configuration:")
        console.plog(params_to_log)

        return self
