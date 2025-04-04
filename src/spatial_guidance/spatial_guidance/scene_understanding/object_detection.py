import json
from enum import Enum, auto
from typing import Dict, List, Optional, Type, Union

from google import genai
from google.genai import types
from langchain.output_parsers import PydanticOutputParser

# from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema.messages import HumanMessage, SystemMessage

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image, ImageColor, ImageDraw
from pydantic import BaseModel, Field, field_validator

from utils import BaseConfig, PathConfig


# Pydantic model for object detection results
class DetectedObject(BaseModel):
    label: str
    box_2d: List[int]  # [y1, x1, y2, x2] in normalized coordinates (0-1000)
    description: Optional[str] = None
    distance: Optional[str] = None  # near, middle, far
    height: Optional[str] = None  # low, eye-level, high
    is_hazard: Optional[bool] = None

    @field_validator("box_2d", mode="before")
    def validate_box(cls, box):
        if len(box) != 4:
            raise ValueError(
                "Bounding box must contain exactly 4 values [y1, x1, y2, x2]"
            )
        return box


class DetectionResults(BaseModel):
    objects: List[DetectedObject]


class ObjectDetectionConfig(BaseConfig["ObjectDetector3D"]):
    target: Type["ObjectDetector3D"] = Field(default_factory=lambda: ObjectDetector3D)

    # Model configuration
    model: str = "gemini-2.0-flash"
    temperature: float = 0.5
    system_instructions: str = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    Focus on details that would be helpful for navigation and spatial understanding for a blind person.
    Include distances (far, near, middle), heights (low, eye-level, high), and potential hazards.

    Each object should have the following properties:
    - label: name of the object
    - box_2d: [y1, x1, y2, x2] with normalized coordinates from 0-1000
    - description: detailed description including position in the scene
    - distance: "near", "middle", or "far" if identifiable
    - height: "low", "eye-level", or "high" if identifiable
    - is_hazard: boolean indicating if the object could be a hazard for navigation
    """

    api_key: str = "GOOGLE_API_KEY"

    # Prompt configuration
    prompt: str = (
        "Detect all objects in this scene. For each object, describe its position in the scene, its approximate distance, and whether it could be a hazard for navigation. Provide detailed spatial information that would be helpful for a blind person."
    )


class ObjectDetector3D:
    """
    Detects objects in images using Google's Gemini model for spatial understanding.
    Designed to provide detailed descriptions for blind navigation assistance.
    """

    def __init__(self, config: ObjectDetectionConfig):
        self.config = config
        self.client = None
        self.langchain_client = None
        self.parser = SimpleJsonOutputParser()
        self.pydantic_parser = PydanticOutputParser(pydantic_object=DetectionResults)

        self.setup_client()

    def setup_client(self):
        """Setup the Gemini client with an API key."""
        api_key = PathConfig().get_api_key(self.config.api_key)
        self.client = genai.Client(api_key=api_key)

        # Setup LangChain client
        self.langchain_client = ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=api_key,
            temperature=self.config.temperature,
            convert_system_message_to_human=True,
        )

    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """
        Detect objects in the provided image and return bounding boxes with descriptions.

        Args:
            image: PIL Image to analyze

        Returns:
            List of dictionaries containing bounding boxes and object descriptions
        """
        if not self.langchain_client:
            raise ValueError(
                "Client not initialized. Please call setup_client with your API key."
            )

        # Resize image if too large (Gemini has input size limits)
        image_copy = image.copy()
        if max(image_copy.size) > 1024:
            image_copy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # Run the model inference using LangChain
        messages = [
            SystemMessage(content=self.config.system_instructions),
            HumanMessage(
                content=[
                    {"type": "text", "text": self.config.prompt},
                    {
                        "type": "image_url",
                        "image_url": self._get_image_data_url(image_copy),
                    },
                ]
            ),
        ]

        response = self.langchain_client.invoke(messages)

        # Parse the JSON response
        result = self._parse_bounding_boxes(response.content)
        return result

    def _get_image_data_url(self, image: Image.Image) -> str:
        """Convert PIL image to data URL for LangChain."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def _parse_bounding_boxes(self, json_output: str) -> List[Dict]:
        """Parse the output from Gemini into a structured format using Pydantic."""
        # Handle markdown fencing if present
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1 :])
                json_output = json_output.split("```")[0]
                break

        # Parse JSON
        try:
            # Try to parse and validate with Pydantic
            json_data = json.loads(json_output)

            # If the result is a direct list of objects
            if isinstance(json_data, list):
                try:
                    validated_data = DetectionResults(objects=json_data).dict()[
                        "objects"
                    ]
                    return validated_data
                except Exception:
                    # Fall back to returning the raw list if validation fails
                    return json_data

            # If there's a wrapper with an "objects" key
            elif isinstance(json_data, dict) and "objects" in json_data:
                try:
                    validated_data = DetectionResults(**json_data).dict()["objects"]
                    return validated_data
                except Exception:
                    # Fall back to returning objects directly
                    return json_data["objects"]

            return json_data

        except json.JSONDecodeError:
            # If failed to parse directly, try to extract JSON from the text
            import re

            json_match = re.search(r"\[.*\]", json_output, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    return []
            return []

    def visualize_detections(
        self, image: Image.Image, detections: List[Dict]
    ) -> Image.Image:
        """
        Draw bounding boxes on the image for visualization.

        Args:
            image: Original PIL image
            detections: List of detection dictionaries with bounding boxes

        Returns:
            PIL Image with bounding boxes drawn
        """
        img = image.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Define colors for bounding boxes
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "brown",
            "gray",
            "beige",
            "turquoise",
            "cyan",
            "magenta",
            "lime",
            "navy",
            "maroon",
            "teal",
            "olive",
            "coral",
            "lavender",
        ] + list(ImageColor.colormap.keys())

        # Draw each bounding box
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]

            # Convert normalized coordinates to absolute coordinates
            box = detection.get("box_2d", [0, 0, 0, 0])
            abs_y1 = int(box[0] / 1000 * height)
            abs_x1 = int(box[1] / 1000 * width)
            abs_y2 = int(box[2] / 1000 * height)
            abs_x2 = int(box[3] / 1000 * width)

            # Fix coordinates if they're out of order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            # Draw the bounding box
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

            # Draw the label
            if "label" in detection:
                draw.text((abs_x1 + 8, abs_y1 + 6), detection["label"], fill=color)

        return img


# Helper function for easier visualization
def plot_bounding_boxes(image: Image.Image, bounding_boxes_json: str) -> Image.Image:
    """
    Utility function to plot bounding boxes directly from JSON output.

    Args:
        image: PIL Image to annotate
        bounding_boxes_json: JSON string of bounding boxes from the model

    Returns:
        Annotated PIL Image with bounding boxes
    """
    # Create temporary detector to parse JSON
    config = ObjectDetectionConfig()
    detector = ObjectDetector3D(config)

    # Parse the boxes and visualize
    boxes = detector._parse_bounding_boxes(bounding_boxes_json)
    return detector.visualize_detections(image, boxes)
