"""
Dedicated Gemini Client module for the Spatial Understanding Agent.
Handles client initialization, chat history management, and operational modes.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    mode: OperationalMode
    frame_idx: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    """Configuration for the Gemini Client."""

    model_name: str = "gemini-2.5-pro-preview-05-06"
    """Gemini model to use for conversations."""

    max_history_length: int = 20
    """Maximum number of messages to keep in chat history."""

    context_window_size: int = 10
    """Number of recent messages to include in context for each request."""

    auto_mode_detection: bool = True
    """Whether to automatically detect operational mode from user input."""

    temperature: float = 0.7
    """Controls randomness in responses."""

    safety_settings: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"),
            ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "BLOCK_ONLY_HIGH"),
        ]
    )


class GeminiClient:
    """
    Dedicated Gemini client for the Spatial Understanding Agent.
    Handles client initialization, chat history management, and operational modes.
    """

    def __init__(self, config: Optional[GeminiClientConfig] = None):
        """Initialize the Gemini client with configuration."""
        self.config = config or GeminiClientConfig()
        self.console = Console.with_prefix(self.__class__.__name__)

        # Initialize Gemini client
        self.client = genai.Client(api_key=PathConfig().get_api_key("GOOGLE_API_KEY"))

        # Chat history and state management
        self.chat_history: List[ChatMessage] = []
        self.current_mode = OperationalMode.GENERAL_SCENE
        self.mode_templates = ModePromptTemplates()

        self.console.log(
            f"Initialized Gemini client with model: {self.config.model_name}"
        )

    def detect_operational_mode(self, user_input: str) -> OperationalMode:
        """
        Automatically detect the appropriate operational mode based on user input.
        """
        if not self.config.auto_mode_detection:
            return self.current_mode

        user_lower = user_input.lower()

        # Check each mode's context filters
        mode_scores = {}
        for mode, template in self.mode_templates.TEMPLATES.items():
            score = sum(
                1 for keyword in template["context_filter"] if keyword in user_lower
            )
            if score > 0:
                mode_scores[mode] = score

        if mode_scores:
            # Return mode with highest score
            detected_mode = max(mode_scores.keys(), key=lambda k: mode_scores[k])
            if detected_mode != self.current_mode:
                self.console.log(
                    f"Mode switch detected: {self.current_mode.value} → {detected_mode.value}"
                )
                self.current_mode = detected_mode
            return detected_mode

        return self.current_mode

    def filter_chat_history(
        self, mode: OperationalMode, current_input: str = ""
    ) -> List[ChatMessage]:
        """
        Filter chat history based on operational mode and relevance to current input.
        """
        if not self.chat_history:
            return []

        # Get relevant keywords for the current mode
        mode_keywords = self.mode_templates.TEMPLATES[mode]["context_filter"]
        input_lower = current_input.lower()

        # Filter messages by relevance
        relevant_messages = []
        for message in self.chat_history[-self.config.context_window_size :]:
            message_lower = message.content.lower()

            # Include if same mode or contains relevant keywords
            if (
                message.mode == mode
                or any(keyword in message_lower for keyword in mode_keywords)
                or any(
                    keyword in input_lower and keyword in message_lower
                    for keyword in mode_keywords
                )
            ):
                relevant_messages.append(message)

        return relevant_messages[-self.config.context_window_size :]

    def add_message(
        self,
        role: str,
        content: str,
        frame_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to chat history with current mode context."""
        import time

        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            mode=self.current_mode,
            frame_idx=frame_idx,
            metadata=metadata or {},
        )

        self.chat_history.append(message)

        # Trim history if too long
        if len(self.chat_history) > self.config.max_history_length:
            self.chat_history = self.chat_history[-self.config.max_history_length :]

    def generate_response(
        self, user_input: str, frame: DatasetOut, frame_idx: int
    ) -> Tuple[str, OperationalMode]:
        """
        Generate a response using the current operational mode and filtered history.
        """
        # Detect and set operational mode
        mode = self.detect_operational_mode(user_input)

        # Get mode-specific system prompt
        system_prompt = self.mode_templates.TEMPLATES[mode]["system"]

        # Filter relevant chat history
        relevant_history = self.filter_chat_history(mode, user_input)

        # Build contents for Gemini
        contents: List[types.Content] = [frame.rgb_image]

        # Add system prompt
        contents.append(types.Part.from_text(text=f"System: {system_prompt}"))

        # Add relevant chat history
        for msg in relevant_history:
            contents.append(
                types.Part.from_text(text=f"{msg.role.title()}: {msg.content}")
            )

        # Add current user input
        contents.append(types.Part.from_text(text=f"User: {user_input}"))

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                generation_config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    safety_settings=[
                        types.SafetySetting(category=cat, threshold=thresh)
                        for cat, thresh in self.config.safety_settings
                    ],
                ),
            )

            response_text = (
                response.text or "I'm sorry, I couldn't process that request."
            )

            # Add messages to history
            self.add_message("user", user_input, frame_idx)
            self.add_message("assistant", response_text, frame_idx)

            return response_text, mode

        except Exception as e:
            self.console.log(f"Error generating response: {e}")
            error_msg = "I'm sorry, I encountered an error processing your request."
            self.add_message("user", user_input, frame_idx)
            self.add_message("assistant", error_msg, frame_idx)
            return error_msg, mode

    def clear_history(self) -> None:
        """Clear all chat history."""
        self.chat_history.clear()
        self.console.log("Chat history cleared")

    def set_mode(self, mode: OperationalMode) -> None:
        """Manually set the operational mode."""
        old_mode = self.current_mode
        self.current_mode = mode
        self.console.log(f"Mode manually set: {old_mode.value} → {mode.value}")

    def get_mode_description(self, mode: OperationalMode) -> str:
        """Get a human-readable description of an operational mode."""
        descriptions = {
            OperationalMode.GENERAL_SCENE: "General scene understanding and description",
            OperationalMode.OBJECT_DETECTION: "Specific object detection and location",
            OperationalMode.COOKING_ASSISTANCE: "Kitchen and cooking assistance",
            OperationalMode.NAVIGATION_GUIDANCE: "Navigation and mobility guidance",
            OperationalMode.ACCESSIBILITY_SUPPORT: "Accessibility and interaction support",
        }
        return descriptions.get(mode, mode.value)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        mode_counts: Dict[str, int] = {}
        for msg in self.chat_history:
            mode_counts[msg.mode.value] = mode_counts.get(msg.mode.value, 0) + 1

        return {
            "current_mode": self.current_mode.value,
            "total_messages": len(self.chat_history),
            "mode_distribution": mode_counts,
            "recent_messages": len([m for m in self.chat_history[-5:]]),
        }
