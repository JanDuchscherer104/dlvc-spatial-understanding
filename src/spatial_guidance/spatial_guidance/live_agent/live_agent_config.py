import dis
from typing import Annotated, Any, List, Literal, Type
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
    aabb_detseg: GeminiAABBDetSegConfig = GeminiAABBDetSegConfig()
    num_detseg_attemts: int = 3

    # Tools
    enable_code_execution: bool = True
    """Enable code execution tool via _ExecAPI."""
    tools: Annotated[List[types.Tool], Field(None)]

    # Live API model configuration
    interaction_mode: InteractionMode = InteractionMode.TEXT
    response_style: ResponseStyle = ResponseStyle(
        dir_style=DirectionalStyle.CLOCK_FACE, dist_style=DistanceStyle.PRECISE
    )
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

    system_instruction_template: Type[LiveAgentPromptTemplates] = Field(
        default_factory=lambda: LiveAgentPromptTemplates
    )
    system_instruction: Annotated[str, Field(default=LiveAgentPromptTemplates)]

    @field_validator("tools", mode="before")
    @classmethod
    def make_tools(cls, _, info: ValidationInfo) -> List[types.Tool]:
        tools = []
        aabb_detseg_config = info.data.get("aabb_detseg")
        assert isinstance(aabb_detseg_config, GeminiAABBDetSegConfig)
        run_aabb_detection_tool = aabb_detseg_config.make_tool()
        # cache lookup – should be called first
        tools.append(
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="get_last_detections",
                        description="Return cached detections for this frame so the model can avoid re-running detection.",
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
                        response=run_aabb_detection_tool.function_declarations[
                            0
                        ].response,
                    )
                ]
            )
        )
        #         TOOLS (in order of preference)
        # - `list_all_detections` - get a map from frame index to the labels of the detected objects. Use ths tool to get an overview of all chached detections in case of a long conversation.
        # - `get_last_detections` - cache lookup. If you can recall that the object was already detected in the current frame, use this tool to avoid re-running detection.
        # - `run_aabb_detection` - detect objects if cache misses.
        # overview of all cached detections
        tools.append(
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="list_all_detections",
                        description="Return an overview of all cached detections: mapping from frame index to list of detected labels.",
                        parameters=types.Schema(
                            type="object",
                            properties={},
                            required=[],
                        ),
                        response=types.Schema(
                            type="object",
                            properties={"overview": types.Schema(type="object")},
                            required=["overview"],
                        ),
                    )
                ]
            )
        )
        # expensive detector
        tools.append(run_aabb_detection_tool)
        # optional code execution (fallback only)
        if info.data.get("enable_code_execution", False):
            tools.append(types.Tool(code_execution=types.ToolCodeExecution))

        return tools

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

        # Use the new prompt template - will be updated later with actual styles
        self.live_model_config.system_instruction = self.system_instruction

        # Build tool overview: function name -> declarations
        tools_info: dict[str, Any] = {}
        for tool in self.tools:
            for decl in getattr(tool, "function_declarations", []) or []:
                # Dump only non-None schema fields
                params_dict = (
                    decl.parameters.model_dump(exclude_none=True)
                    if decl.parameters
                    else None
                )
                resp_dict = (
                    decl.response.model_dump(exclude_none=True)
                    if decl.response
                    else None
                )
                entry: dict[str, Any] = {"description": decl.description}
                if params_dict is not None:
                    entry["parameters"] = params_dict
                if resp_dict is not None:
                    entry["response"] = resp_dict
                tools_info[decl.name] = entry

        params_to_log = {
            "live_model": self.live_model,
            "interaction_mode": self.interaction_mode,
            "response_modalities": self.live_model_config.response_modalities,
            "tools": tools_info,
            "system_instruction": self.live_model_config.system_instruction,
        }
        console = Console.with_prefix(self.__class__.__name__)
        console.log(f"Live model configuration:")
        console.plog(params_to_log)

        return self
