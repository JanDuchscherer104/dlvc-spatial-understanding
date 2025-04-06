import requests

from spatial_guidance.pipeline import Data, PipelineStage
from utils import CONSOLE
from pydantic import Field
from typing import Type
from utils import BaseConfig, PathConfig

class VlmQueryStage(PipelineStage):
    def __init__(self, config: "VlmQueryConfig"):
        self.api_url = f"{config.api_url}?key={config.api_key_env}"

    def process(self, data: Data) -> Data:
        if not data.prompt or not data.bounding_boxes or not data.labels:
            CONSOLE.warn("Missing prompt, bounding boxes, or labels for Gemini API call.")
            return data

        payload = {
            "contents": [{
                "parts": [{"text": data.prompt + " " + "labels: " + data.labels + " " + "bounding boxes: " + data.bounding_boxes}]
            }]
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            CONSOLE.log(f"Gemini API response: {result}")
            data.gemini_response = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:
            CONSOLE.warn(f"Error calling Gemini API: {e}")
            data.gemini_response = "Error calling Gemini API."

        return data


class VlmQueryConfig(BaseConfig["VlmQueryStage"]):
    target: Type["VlmQueryStage"] = Field(default_factory=lambda: VlmQueryStage)

    api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    api_key_env: str = PathConfig().get_api_key("GOOGLE_API_KEY")

