from typing import Any

from attr import field
from google.genai.types import FunctionDeclaration, Schema, Tool, ToolCodeExecution
from pydantic import Field, ValidationInfo, field_validator

from ..utils import BaseConfig


class LiveAgentTools(BaseConfig):
    """Configuration for live agent tools."""

    run_aabb_detection: FunctionDeclaration = Field(
        default_factory=lambda: FunctionDeclaration(
            name="run_aabb_detection",
            description=(
                "Detects objects in RGB-D images and computes their 3D spatial properties "
                "for navigation assistance."
            ),
            parameters=Schema(
                type="OBJECT",
                properties={
                    "user_prompt": Schema(
                        type="STRING",
                        description=(
                            "Objects to detect or navigation goal. Provide a concise query. Examples:"
                            " 'blue street sign saying 'Dacheuer Str.', 'silver car in front of the house'.\n"
                            "For path descriptions, clearly state the destination: "
                            "'Destination: <unique and descriptive label of the destination>'."
                        ),
                    ),
                    "detection_mode": Schema(
                        type="STRING",
                        enum=[
                            "hazards",
                            "navigation_landmarks",
                            "subset",
                            "path_description",
                        ],
                        description=(
                            "REQUIRED. Always specify detection_mode based on the query: "
                            "Detection mode: "
                            "'subset': detect specific objects mentioned in user_prompt (use for queries like 'where is X?'), "
                            "'hazards': find dangerous objects and obstacles for safety assessment, "
                            "'navigation_landmarks': find doors, stairs, signs, crossings, directional markers for wayfinding, "
                            "path_description: "
                            """
When the user asks for path descriptions to reach a destination (e.g., "How do I get to the door?", "What's the route to the stairs?", "Navigate to the exit"):
1. Use the `run_aabb_detection` tool with `detection_mode="path_description"` to perform comprehensive path analysis. Provide a concise and unique destination label in the user prompt. The tool will detect the destination + obstacles and landmarks along the path.
2. Based on the  detection results, provide clear step-by-step navigation instructions:
   - Start with the destination location and distance
   - Describe the recommended path with specific directions, mentioning waypoints in order of encounter (increasing detpth order)
   - Highlight obstacles to avoid and how to navigate around them
   - Include helpful landmarks for orientation
"""
                        ),
                    ),
                },
                required=["detection_mode"],
            ),
            response=Schema(
                type="OBJECT",
                properties={
                    "detections": Schema(
                        type="ARRAY",
                        description="List of detected objects with spatial information",
                        items=Schema(
                            type="OBJECT",
                            properties={
                                "label": Schema(
                                    type="STRING",
                                    description="Unique human-readable object name (e.g., 'red parked car', 'wooden stairs')",
                                ),
                                "frame_idx": Schema(
                                    type="INTEGER",
                                    description="Frame index of the detection",
                                ),
                                "bbox": Schema(
                                    type="ARRAY",
                                    description="Bounding box [y0, x0, y1, x1], normalized to [0,1000]",
                                    items=Schema(type="NUMBER"),
                                    min_items=4,
                                    max_items=4,
                                ),
                                "center_point_3d": Schema(
                                    type="ARRAY",
                                    description="3D position [x, y, z] in meters: x=right, y=down, z=forward",
                                    items=Schema(type="NUMBER"),
                                    min_items=3,
                                    max_items=3,
                                ),
                                "depth": Schema(
                                    type="NUMBER",
                                    description="Distance to object in meters",
                                ),
                                "rotation_clock": Schema(
                                    type="INTEGER",
                                    description="Direction in clock hours (12=straight ahead, 3=right, 9=left)",
                                ),
                                "rotation_deg": Schema(
                                    type="NUMBER",
                                    description="Angle in degrees from camera's optical axis (0=straight ahead, 90=left, -90=right)",
                                ),
                                "center_height_3d": Schema(
                                    type="NUMBER",
                                    description="Height of 'center_point_3d' above ground in meters. If the user asks for the height of an object (e.g. 'What's the height of the blue sign?') ",
                                    title="Height",
                                ),
                            },
                            required=[
                                "label",
                                "bbox",
                                "center_point_3d",
                                "depth",
                                "rotation_clock",
                                "rotation_deg",
                            ],
                        ),
                    )
                },
                required=["detections"],
            ),
        )
    )

    get_last_detections: FunctionDeclaration = Field(
        default_factory=lambda: FunctionDeclaration(
            name="get_last_detections",
            # description="Return cached detections for this frame so the model can avoid re-running detection.",
            description=(
                "Return cached detections from *frame_idx* transformed into the current ego frame."
                "*Always* use this tool, when you have alreay detected the relevant objects. This will work even if the objects are not visible anymore! If you have not detected the objects yet, use the `run_aabb_detection` tool instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "frame_idx": {"type": "integer"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["frame_idx", "labels"],
            },
            response=None,  # Same as run_aabb_detection response
        )
    )

    # get_last_detection_keys: FunctionDeclaration = Field(
    #     default_factory=lambda: FunctionDeclaration(
    #         name="get_last_detection_keys",
    #         description="Return all cached detection keys (frame index and label) from previous detections. Use this when you are not sure which objects are cached.",
    #         parameters=Schema(type="OBJECT", properties={}, required=[]),
    #         response=Schema(
    #             type="OBJECT",
    #             properties={
    #                 "keys": Schema(
    #                     type="ARRAY",
    #                     description="List of cached detection keys.",
    #                     items=Schema(
    #                         type="OBJECT",
    #                         properties={
    #                             "frame_idx": Schema(type="INTEGER"),
    #                             "label": Schema(type="STRING"),
    #                         },
    #                         required=["frame_idx", "label"],
    #                     ),
    #                 )
    #             },
    #             required=["keys"],
    #         ),
    #     )
    # )

    @field_validator("get_last_detections", mode="before")
    @classmethod
    def _copy_response_schema(
        cls, v: FunctionDeclaration, info: ValidationInfo
    ) -> FunctionDeclaration:
        run_aabb_detection = info.data.get("run_aabb_detection", None)
        assert isinstance(run_aabb_detection, FunctionDeclaration)

        v.response = run_aabb_detection.response.model_copy()
        assert (
            v.response is not None
        ), "Response schema must be set for get_last_detections"
        return v

    def setup_target(self, has_code_execution: bool = True) -> list[Tool]:
        tools: list[Tool] = [
            Tool(
                function_declarations=[
                    self.run_aabb_detection,
                    self.get_last_detections,
                ],
            )
        ]

        if has_code_execution:
            tools.append(ToolCodeExecution)

        return tools

    @classmethod
    def make_info(cls, tools: list[Tool]) -> dict[str, Any]:
        """Create a dictionary with tool information for logging."""
        # Build tool overview: function name -> declarations
        tools_info: dict[str, Any] = {}
        for tool in tools:
            for decl in tool.function_declarations or []:
                # Dump only non-None schema fields
                params_dict = (
                    decl.parameters.model_dump(exclude_none=True)
                    if decl.parameters
                    else None
                )
                resp_dict = (
                    decl.response.model_dump(exclude_none=True)
                    if decl.response
                    else None
                )
                entry: dict[str, Any] = {"description": decl.description}
                if params_dict is not None:
                    entry["parameters"] = params_dict
                if resp_dict is not None:
                    entry["response"] = resp_dict
                tools_info[decl.name] = entry

        return tools_info


### Deprecated tools ###
# tools.append(
#     Tool(
#         function_declarations=[
#             FunctionDeclaration(
#                 name="list_all_detections",
#                 description="Return an overview of all cached detections: mapping from frame index to list of detected labels.",
#                 parameters=Schema(
#                     type="object",
#                     properties={},
#                     required=[],
#                 ),
#                 response=Schema(
#                     type="object",
#                     properties={"overview": Schema(type="object")},
#                     required=["overview"],
#                 ),
#             )
#         ]
#     )
# )
