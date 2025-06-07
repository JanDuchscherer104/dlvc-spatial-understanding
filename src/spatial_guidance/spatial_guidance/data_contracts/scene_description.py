from typing import Dict, List

from pydantic import Field

from . import DataModel


class SceneDescription(DataModel):
    """Data model for scene description output for visually impaired navigation assistance."""

    immediate_safety_hazards: str = Field(
        ...,
        description="Brief description of the most critical safety hazards in the scene that pose immediate risks, including their approximate distance (in meters) and orientation (using clock positions 1-12 o'clock). Focus on moving objects, obstacles in the path, and head-level hazards. If no hazards are present, state 'No immediate safety hazards detected.'",
    )
    scene_description: str = Field(
        ...,
        description="Concise but comprehensive description of all relevant detections in the scene, organized by their spatial relationship using clock positions and distances. Include objects that are important for navigation, orientation, and environmental understanding. Prioritize objects closest to the user and those that affect safe movement.",
    )
    navigation_guidance: str = Field(
        ...,
        description="Specific, actionable guidance for safe navigation and movement through the environment. Include recommended paths, areas to avoid, and any special considerations. When a user destination or goal is specified, provide direction-specific guidance to help reach that location safely.",
    )
