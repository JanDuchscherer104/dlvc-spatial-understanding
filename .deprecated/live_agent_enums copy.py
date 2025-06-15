from enum import Enum, auto
from typing import NamedTuple


class DirectionalStyle(Enum):
    """Different styles for expressing directions."""

    CLOCK_FACE = "clock"  # "at 2 o'clock"
    RELATIVE = "relative"  # 3 meters to your right and 2.5 meters in front of you
    CARTESIAN = "cartesian"
    DEGREES = "degrees"  # "30 degrees right"

    def prompt(self) -> str:
        """Return detailed instructions for each directional style."""
        return {
            "clock": (
                "Express bearing as a clock-face position where 12 o'clock is straight ahead, "
                "3 o'clock is directly to your right, 6 o'clock is behind you, and 9 o'clock is to your left. "
                "Use the rotation_clock value from the 'run_aabb_detection' tool result. "
                "Examples: 'at 2 o'clock', 'at 10 o'clock', 'at 6 o'clock'"
            ),
            "relative": (
                "Use natural spatial language relative to the user's position. "
                "Describe objects as 'to your left/right', 'in front of you', 'behind you', "
                "or combinations like 'slightly to your right' or 'far to your left'. "
                "Include forward/backward positioning when helpful."
            ),
            "cartesian": (
                "Use precise Cartesian coordinates based on the center_point_3d values from detection results. "
                "X-axis: positive is to your right, negative to your left (in meters). "
                "Y-axis: positive is down, negative is up. "
                "Z-axis: positive is in front of you, negative is behind you (in meters). "
                "Format: 'X.X meters to your right/left and X.X meters in front of you'"
            ),
            "degrees": (
                "Express bearing in degrees clockwise from straight ahead (0°). "
                "0° = straight ahead, 90° = directly right, 180° = behind you, 270° = directly left. "
                "Use whole numbers and include 'clockwise' for clarity. "
                "Examples: '45 degrees clockwise', '270 degrees clockwise'"
            ),
        }[self.value]

    def examples(self) -> list[str]:
        """Multiple example phrases for each directional style with variety."""
        return {
            DirectionalStyle.CLOCK_FACE: [
                "at 3 o'clock",
                "at 11 o'clock",
                "at 6 o'clock",
                "at 1:30",
                "between 2 and 3 o'clock",
            ],
            DirectionalStyle.RELATIVE: [
                "to your right",
                "slightly to your left",
                "directly in front of you",
                "far to your right",
                "to your left and slightly forward",
                "behind you and to the right",
            ],
            DirectionalStyle.CARTESIAN: [
                "5.2 meters to your right and 3.1 meters in front of you",
                "2.1 meters to your left and 1.8 meters in front of you",
                "0.9 meters to your right and 4.2 meters behind you",
                "directly in front of you at 2.5 meters",
                "1.3 meters to your left",
            ],
            DirectionalStyle.DEGREES: [
                "30 degrees clockwise",
                "120 degrees clockwise",
                "270 degrees clockwise",
                "45 degrees clockwise",
                "180 degrees clockwise",
            ],
        }[self]


class DistanceStyle(Enum):
    """Different styles for expressing distances."""

    PRECISE = "precise"  # "2.3 meters"
    APPROXIMATE = "approximate"  # "about 2 meters"
    RELATIVE = "relative"  # "arm's reach", "across the room"

    def examples(self) -> list[str]:
        """Multiple example phrases for each distance style with variety."""
        return {
            DistanceStyle.PRECISE: [
                "8.4 meters",
                "2.1 meters",
                "0.6 meters",
                "12.3 meters",
                "1.0 meter",
            ],
            DistanceStyle.APPROXIMATE: [
                "about 8 meters",
                "roughly 2 meters",
                "around 5 meters",
                "nearly 1 meter",
                "just over 3 meters",
            ],
            DistanceStyle.RELATIVE: [
                "arm's reach",
                "across the room",
                "a few steps away",
                "within reach",
                "far across the space",
                "just beyond your reach",
                "very close",
                "at the other end of the room",
            ],
        }[self]

    def prompt(self) -> str:
        """Return detailed instructions for each distance style."""
        return {
            "precise": (
                "Always give distances to exactly one decimal place in meters (e.g. '2.3 m', '0.8 m', '15.6 m'). "
                "Use the exact distance measurements from the detection results. "
                "For distances under 1 meter, still use one decimal place (e.g. '0.3 m', not '30 cm')."
            ),
            "approximate": (
                "Round distances to the nearest whole meter and use approximating language. "
                "Use phrases like 'about X meters', 'roughly X meters', 'around X meters', "
                "'nearly X meters', or 'just over X meters'. Make it conversational and natural."
            ),
            "relative": (
                "Use intuitive, qualitative distance descriptions that relate to human experience. "
                "Examples: 'arm's reach' (≤1m), 'a step away' (~1m), 'a few steps' (2-3m), "
                "'across the room' (4-8m), 'far across the space' (>8m). "
                "Choose phrases that help users understand reachability and navigation."
            ),
        }[self.value]


# Combined direction and distance style for prompts
class ResponseStyle(NamedTuple):
    """Combined direction and distance style for prompts."""

    dir_style: DirectionalStyle
    dist_style: DistanceStyle

    def prompt(self) -> str:
        """Return combined style constraints."""
        return f"{self.dir_style.prompt()}\n{self.dist_style.prompt()}"


class GenState(Enum):
    COLLECT_FIRST = auto()
    WAIT_TOOL = auto()
    COLLECT_FINAL = auto()


class InteractionMode(Enum):
    TEXT = auto()
    VOICE = auto()


class OperationalMode(Enum):
    """Different operational modes for the Spatial Understanding Agent."""

    GENERAL_SCENE = "general_scene_understanding"
    OBJECT_DETECTION = "specific_object_detection"
    COOKING_ASSISTANCE = "cooking_assistance"
    NAVIGATION_GUIDANCE = "navigation_guidance"
    ACCESSIBILITY_SUPPORT = "accessibility_support"


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
