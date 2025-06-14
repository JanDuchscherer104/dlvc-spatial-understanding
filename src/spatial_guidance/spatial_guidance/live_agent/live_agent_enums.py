from enum import Enum, auto
from typing import NamedTuple


class DirectionalStyle(Enum):
    CLOCK_FACE = "clock"
    CARTESIAN = "cartesian"
    DEGREES = "degrees"

    def prompt(self):
        return {
            DirectionalStyle.CLOCK_FACE: "I express orientations using a clock-face bearing (e.g. “at 2 o'clock”, 12 = ahead). Read `rotation_clock` and `depth`.",
            DirectionalStyle.CARTESIAN: "I report coordinates using the metric coordinates from `center_point_3d`: <x> meters to your right/left, <z> meters ahead/behind” (x right +, z forward +). Depending on the context, I may also use y (y down +).",
            DirectionalStyle.DEGREES: "I report bearings in degrees clockwise from straight ahead (0 degrees = front), using `rotation_clock` and `depth`.",
        }[self]

    def examples(self):
        q1 = "Where is the trash can?"
        q2 = "Were is the next traffic light?"
        q3 = "How do I get to the door?"
        return {
            DirectionalStyle.CLOCK_FACE: [
                f'Q: "{q1}"\nA: "The trash can is 2.5 meters away at 3 o\'clock."',
                f'Q: "{q2}"\nA: "There are two traffic lights, one at 1 o\'clock, 3.2 meters away, and another at 5 o\'clock, 1.5 meters away."',
                f"Q: \"{q3}\"\nA: \"To reach the door at 11 o'clock, 4.2 meters away, walk straight ahead. There's a chair at 10 o'clock, 2.1 meters away that you'll need to walk around to your right.\"",
            ],
            DirectionalStyle.CARTESIAN: [
                f'Q: "{q1}"\nA: "The trash can is 2.5 meters to your right and 1.0 meters ahead."',
                f'Q: "{q2}"\nA: "There are two traffic lights, one 6.2 meters to your right and 8.5 meters ahead, and another 4.5 meters to your left and 2.0 meters ahead."',
                f'Q: "{q3}"\nA: "To reach the door 1.5 meters to your left and 4.0 meters ahead, walk forward and slightly left. Watch out for a chair 2.0 meters to your left and 2.1 meters ahead that may block your path."',
            ],
            DirectionalStyle.DEGREES: [
                f'Q: "{q1}"\nA: "The trash can is 20 degrees left at 2.5 m."',
                f'Q: "{q2}"\nA: "There are two traffic lights, one at 30 degrees right and 5 meters ahead, and another at 65 degrees left at a distance of 3.2 meters."',
                f'Q: "{q3}"\nA: "To reach the door at 330 degrees (30 degrees left), 4.2 meters away, walk forward and turn slightly left. Avoid the chair at 45 degrees, 2.1 meters away."',
            ],
        }[self]


class DistanceStyle(Enum):
    PRECISE = "precise"
    APPROXIMATE = "approximate"

    def prompt(self):
        return {
            DistanceStyle.PRECISE: "I will report distance with one decimal place (e.g. 2.3 m).",
            DistanceStyle.APPROXIMATE: "I will round to whole metres prefaced by “about/roughly”.",
        }[self]

    def examples(self):
        return {
            DistanceStyle.PRECISE: ["2.3 meters", "0.8 meters"],
            DistanceStyle.APPROXIMATE: ["about 8 meters", "roughly 1 meter"],
        }[self]


# Combined direction and distance style for prompts
class ResponseStyle(NamedTuple):
    """Combined direction and distance style for prompts."""

    dir_style: DirectionalStyle
    dist_style: DistanceStyle

    def prompt(self) -> str:
        """Return combined style constraints."""
        return f"I will express directions and distances in the following manner:\n{self.dir_style.prompt()}\n{self.dist_style.prompt()}"

    def examples(self) -> str:
        """Return example Q&A pairs using the combined style."""
        return (
            f"EXAMPLES on how I will express directions:\n{self.dir_style.examples()}"
        )


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
