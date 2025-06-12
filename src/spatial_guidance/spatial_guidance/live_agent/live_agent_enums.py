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
