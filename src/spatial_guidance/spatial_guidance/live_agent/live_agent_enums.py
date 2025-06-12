from enum import Enum, auto


class DirectionalStyle(Enum):
    """Different styles for expressing directions."""

    CLOCK_FACE = "clock"  # "at 2 o'clock"
    RELATIVE = "relative"  # 3 meters to your right and 2.5 meters in front of you
    COMPASS = "compass"  # "northeast"
    DEGREES = "degrees"  # "30 degrees right"

    def prompt(self) -> str:
        return {
            "clock": "Express bearing as a clock-face (e.g. 'at 2 o'clock'). Use the rotation_clock from the result of the 'run_aabb_detection' tool.",
            "relative": "Say 'to your right / left' and, if useful, include the lateral offset in metres.",
            "compass": "Use compass points such as 'north-east'.",
            "degrees": "State the bearing in degrees clockwise from straight ahead.",
        }[self.value]


class DistanceStyle(Enum):
    """Different styles for expressing distances."""

    PRECISE = "precise"  # "2.3 meters"
    APPROXIMATE = "approximate"  # "about 2 meters"
    RELATIVE = "relative"  # "arm's reach", "across the room"

    def prompt(self) -> str:
        return {
            "precise": "Give the distance to one decimal place (e.g. '2.3 m').",
            "approximate": "Round to whole metres (e.g. 'about 5 m').",
            "relative": "Use qualitative phrases (e.g. 'arm's reach', 'half the room').",
        }[self.value]


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
