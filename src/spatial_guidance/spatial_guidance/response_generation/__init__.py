"""
Response Generation Module

This module provides natural language response generation capabilities for the
Spatial Understanding Agent. It translates detection results and spatial data
into contextually appropriate, mode-specific natural language responses.

Key Components:
    - ResponseGenerator: Main class for generating natural language responses
    - DirectionalStyle: Enum for different directional description styles
    - DistanceStyle: Enum for different distance expression formats
"""

from .response_generator import DirectionalStyle, DistanceStyle, ResponseGenerator

__all__ = [
    "ResponseGenerator",
    "DirectionalStyle",
    "DistanceStyle",
]
