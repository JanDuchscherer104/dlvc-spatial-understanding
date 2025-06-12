from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from ..data_contracts.dataset import DatasetOut
from ..utils import BaseConfig, Console, PathConfig


class OperationalMode(Enum):
    """Different operational modes for the Spatial Understanding Agent."""

    GENERAL_SCENE = "general_scene_understanding"
    OBJECT_DETECTION = "specific_object_detection"
    COOKING_ASSISTANCE = "cooking_assistance"
    NAVIGATION_GUIDANCE = "navigation_guidance"
    ACCESSIBILITY_SUPPORT = "accessibility_support"


class ChatMessage(BaseModel):
    """Structured representation of a chat message."""

    role: Literal["user", "assistant"]  # "user" or "assistant"
    content: str
    timestamp: float
    mode: OperationalMode
    frame_idx: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    """Optional tags describing the message content."""


class ModePromptTemplates:
    """Centralized prompt templates for different operational modes."""

    TEMPLATES = {
        OperationalMode.GENERAL_SCENE: {
            "system": (
                "You are an AI assistant helping visually impaired users understand their environment. "
                "Provide clear, concise descriptions of scenes, focusing on spatial relationships, "
                "important objects, and potential navigation hazards. Use directional language like "
                "'at 2 o'clock' or 'to your left' when describing object positions."
            ),
            "context_filter": [
                "scene",
                "overview",
                "general",
                "describe",
                "what do you see",
            ],
        },
        OperationalMode.OBJECT_DETECTION: {
            "system": (
                "You are specialized in object detection for accessibility. Focus on identifying "
                "specific objects the user requests. Provide precise locations using clock directions "
                "(12 o'clock = straight ahead), distances when available, and actionable guidance "
                "for reaching or avoiding objects."
            ),
            "context_filter": [
                "find",
                "locate",
                "where",
                "detect",
                "show",
                "identify",
                "objects",
            ],
        },
        OperationalMode.COOKING_ASSISTANCE: {
            "system": (
                "You are a cooking assistant for visually impaired users. Help identify kitchen "
                "tools, ingredients, cooking surfaces, and safety hazards. Provide step-by-step "
                "guidance with precise spatial references. Alert users to hot surfaces, sharp "
                "objects, and spill hazards."
            ),
            "context_filter": [
                "cooking",
                "kitchen",
                "recipe",
                "ingredient",
                "stove",
                "knife",
                "hot",
            ],
        },
        OperationalMode.NAVIGATION_GUIDANCE: {
            "system": (
                "You are a navigation guide for visually impaired users. Focus on identifying "
                "pathways, obstacles, stairs, doors, and mobility hazards. Provide clear directional "
                "guidance and warn about trip hazards, overhead obstacles, and surface changes."
            ),
            "context_filter": [
                "path",
                "walk",
                "go",
                "navigate",
                "stairs",
                "door",
                "obstacle",
                "hazard",
            ],
        },
        OperationalMode.ACCESSIBILITY_SUPPORT: {
            "system": (
                "You are an accessibility support assistant. Help users interact with their "
                "environment by identifying buttons, switches, signs, text, and interactive elements. "
                "Describe how to operate devices and navigate spaces independently."
            ),
            "context_filter": [
                "button",
                "switch",
                "sign",
                "text",
                "operate",
                "use",
                "access",
            ],
        },
    }


class GeminiClientConfig(BaseConfig["GeminiClient"]):
    """Configuration for :class:`GeminiClient`."""

    target: Type["GeminiClient"] = Field(default_factory=lambda: GeminiClient)
    model_name: str = "gemini-2.5-pro-preview-05-06"
    max_history_length: int = 20
    context_window_size: int = 10
    temperature: float = 0.7
    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "BLOCK_ONLY_HIGH"),
        ]
    )


class GeminiClient:
    """Central Gemini interface used by all scene understanding models."""

    def __init__(self, config: Optional[GeminiClientConfig] = None) -> None:
        self.config = config or GeminiClientConfig()
        self.console = Console.with_prefix(self.__class__.__name__)
        self.client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))
        self.chat_history: List[ChatMessage] = []
        self.current_mode = OperationalMode.GENERAL_SCENE
        self.mode_templates = ModePromptTemplates()

        self.console.log(
            f"Initialized Gemini client with model: {self.config.model_name}"
        )

    # ------------------------------------------------------------------
    # Conversation utilities
    # ------------------------------------------------------------------
    def _detect_mode(self, user_input: str) -> OperationalMode:
        """Simple keyword based mode detection."""
        # TODO: should be replaced with gemini-live function calling
        user_lower = user_input.lower()
        for mode, template in self.mode_templates.TEMPLATES.items():
            if any(k in user_lower for k in template["context_filter"]):
                if mode != self.current_mode:
                    self.console.log(
                        f"Mode switch detected: {self.current_mode.value} → {mode.value}"
                    )
                self.current_mode = mode
                return mode
        return self.current_mode

    def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        *,
        tags: Optional[List[str]] = None,
    ) -> None:
        import time

        self.chat_history.append(
            ChatMessage(
                role=role,
                content=content,
                timestamp=time.time(),
                mode=self.current_mode,
                frame_idx=None,
                metadata={},
                tags=tags or [],
            )
        )

        if len(self.chat_history) > self.config.max_history_length:
            self.chat_history = self.chat_history[-self.config.max_history_length :]

    def get_context(self, tags: Optional[List[str]] = None) -> List[ChatMessage]:
        tags = set(tags or [])
        history = self.chat_history[-self.config.context_window_size :]
        if not tags:
            return history
        return [m for m in history if tags.intersection(m.tags)]

    # ------------------------------------------------------------------
    # Gemini interaction
    # ------------------------------------------------------------------
    def generate_response(
        self,
        user_input: str,
        frame: DatasetOut,
        *,
        parts: Optional[List[types.Part]] = None,
        tags: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, OperationalMode]:
        mode = self._detect_mode(user_input)
        system_prompt = system_prompt or self.mode_templates.TEMPLATES[mode]["system"]

        context = self.get_context(tags)
        contents: List[types.Content] = []
        contents.extend(parts or [frame.rgb_image])
        contents.append(types.Part.from_text(text=f"System: {system_prompt}"))
        for msg in context:
            contents.append(
                types.Part.from_text(text=f"{msg.role.title()}: {msg.content}")
            )
        contents.append(types.Part.from_text(text=f"User: {user_input}"))

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                generation_config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    safety_settings=[
                        types.SafetySetting(category=c, threshold=t)
                        for c, t in self.config.safety_settings
                    ],
                ),
            )
            text = response.text or ""
        except Exception as e:  # pragma: no cover - network errors
            self.console.warn(f"Gemini error: {e}")
            text = ""

        self.add_message("user", user_input, tags=tags)
        self.add_message("assistant", text, tags=tags)
        return text, mode

    def generate_content(
        self,
        *,
        parts: List[types.Part],
        generation_config: types.GenerateContentConfig,
        tags: Optional[List[str]] = None,
    ) -> types.GenerateContentResponse:
        """Low level wrapper to call the Gemini API and record the response."""
        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=parts,
                generation_config=generation_config,
            )
            text = response.text or ""
        except Exception as e:  # pragma: no cover - network errors
            self.console.warn(f"Gemini error: {e}")
            raise

        self.add_message("assistant", text, tags=tags)
        return response

    def clear_history(self) -> None:
        self.chat_history.clear()
