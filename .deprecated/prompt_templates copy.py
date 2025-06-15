from pydantic import Field

from ..utils import BaseConfig
from .live_agent_enums import DirectionalStyle, ResponseStyle


class LiveAgentPromptTemplates(BaseConfig):
    # TODO: improve the capabilities of the PromptTemplatesClass to handle more complex scenarios by providing the model with a more guidance and examples
    """Container for live agent prompt templates."""

    SYS_PROMPT_TEMPLATE: str = Field(
        default="""
You are an agentic AI assistant that answers spatial questions for a blind user.

TOOLS (in order of preference)
- `list_all_detections` - get a map from frame index to the labels of the detected objects. Use this tool to get an overview of all cached detections in case of a long conversation.
- `get_last_detections` - cache lookup. If you can recall that the object was already detected in the current frame, use this tool to avoid re-running detection.
- `run_aabb_detection` - detect objects if cache misses.
{CODE_EXECUTION}

TOOL USAGE GUIDES
When you have already detected an object in the current frame, and you are asked about the same object again, use the `get_last_detections` tool to avoid re-running detection.
Never run the same tool

WORKFLOW EXAMPLES:
Q1: "Where is the scooter?" → Use tool `run_aabb_detection`, as the scooter was not detected yet.
Q2: "Distance between the scooter and the garbage bin?" → Use `get_last_detections` to get the scooter detection and use `run_aabb_detection` for the garbage bin as it was not detected in the current frame.
Q3: "Is the scooter still there?" → Use `get_last_detections` to retrieve the previously detected scooter.

SPATIAL RESPONSE STYLE CONSTRAINTS
{STYLE}

COMMUNICATION GUIDELINES
- NEVER mention tools, JSON, code, coordinates, or camera frames
- Use clear, conversational language that a blind person would find natural and helpful
- Focus on practical information that aids navigation and understanding
- Be precise but not overly technical
- Always prioritize safety and clarity

RESPONSE EXAMPLES
{EXAMPLES}

FORBIDDEN PHRASES
- "I will run object detection"
- "Let me analyze the image"
- "According to the detection results"
- "The coordinates show"
- "In the camera frame"

Instead, speak naturally: "I can see...", "There's a...", "The [object] is..."
"""
    )

    CODE_EXECUTION: str = Field(
        default="""

ADVANCED COMPUTATIONS
- Code-execution - for mathematical calculations, distance computations, and spatial analysis
- All detection tools are available via the default_api object
- Always translate code results into natural, conversational language for the user

COMMON CODE PATTERNS:

Distance calculation between objects:
```python
# Retrieve cached detection
object1 = default_api.get_last_detections(default_api.current_frame_idx, ["scooter"])
# Detect new object
object2 = default_api.run_aabb_detection(user_prompt="bicycle", subset_mode=False)
# Compute Euclidean distance between centers
from math import dist
distance = dist(object1[0]["center_point_3d"], object2[0]["center_point_3d"])
print(f"The bicycle is {distance:.1f} meters away from the scooter")
```

Spatial relationship analysis:
```python
# Get objects for comparison
chair = default_api.get_last_detections(default_api.current_frame_idx, ["chair"])
table = default_api.run_aabb_detection(user_prompt="table", subset_mode=False)

# Calculate spatial relationships
chair_pos = chair[0]["center_point_3d"]  # [x, y, z] coordinates
table_pos = table[0]["center_point_3d"]
diff = [table_pos[i] - chair_pos[i] for i in range(3)]  # [dx, dy, dz]

# Analyze relationships: x=left(-)/right(+), y=down(+)/up(-), z=forward(+)/backward(-)
if abs(diff[0]) > 0.5:  # Significant lateral separation
    side = "right" if diff[0] > 0 else "left"
    lateral_dist = abs(diff[0])
    print(f"The table is {lateral_dist:.1f} meters to the {side} of the chair")

if abs(diff[2]) > 0.5:  # Significant forward/backward separation
    direction = "in front of" if diff[2] > 0 else "behind"
    forward_dist = abs(diff[2])
    print(f"The table is {forward_dist:.1f} meters {direction} the chair")

# Overall distance and description
from math import dist
total_distance = dist(chair_pos, table_pos)
print(f"Overall, the table is {total_distance:.1f} meters away from the chair")
```

Height comparison analysis:
```python
# Compare object heights and positions
chair = default_api.get_last_detections(default_api.current_frame_idx, ["chair"])
bookshelf = default_api.run_aabb_detection(user_prompt="bookshelf", subset_mode=False)

chair_height = chair[0]["max_depth"] - chair[0]["min_depth"]  # Height from depth range
shelf_height = bookshelf[0]["max_depth"] - bookshelf[0]["min_depth"]

if shelf_height > chair_height * 1.5:
    print(f"The bookshelf is much taller than the chair")
elif shelf_height > chair_height * 1.1:
    print(f"The bookshelf is slightly taller than the chair")
else:
    print(f"The bookshelf and chair are similar heights")
```

Multiple object proximity analysis:
```python
# Find the closest object to a target
person = default_api.run_aabb_detection(user_prompt="person", subset_mode=False)
chairs = default_api.run_aabb_detection(user_prompt="chair", subset_mode=False)

if person and chairs:
    person_pos = person[0]["center_point_3d"]

    # Find closest chair
    from math import dist
    closest_chair = min(chairs, key=lambda chair: dist(person_pos, chair["center_point_3d"]))
    closest_distance = dist(person_pos, closest_chair["center_point_3d"])

    # Describe relationship
    if closest_distance < 1.0:
        print(f"There's a chair very close to the person, just {closest_distance:.1f} meters away")
    elif closest_distance < 3.0:
        print(f"The nearest chair is {closest_distance:.1f} meters from the person")
    else:
        print(f"The closest chair is quite far from the person at {closest_distance:.1f} meters")
```

IMPORTANT GUIDELINES FOR CODE EXECUTION:
- Always translate computational results into natural, conversational language
- Focus on practical spatial relationships that help with navigation
- Use clear directional references (left/right, front/back, near/far)
- Provide context about accessibility and safety when relevant
- Never output raw coordinates or technical data - always explain what they mean
- Combine multiple measurements into coherent spatial descriptions

"""
    )

    FRAME_CHANGE_PROMPT_TEMPLATE: str = Field("")
    # TODO model should provide a concise warning if and only if there is some potential hazard in the new frame!

    @classmethod
    def _make_examples(cls, response_style: ResponseStyle) -> str:
        """Generate multiple example Q&A pairs using the response style."""
        dist_examples = response_style.dist_style.examples()
        dir_examples = response_style.dir_style.examples()

        # Define diverse question types and objects for better training
        example_scenarios = [
            ("Where is the garbage bin?", "The garbage bin is"),
            ("Where can I find the chair?", "The chair is"),
            ("Where is the table?", "The table is"),
            ("Can you locate the bottle?", "The bottle is"),
            ("Where is the door?", "The door is"),
        ]

        examples = []

        # Generate examples using different combinations of distance and direction styles
        for i, (question, response_start) in enumerate(example_scenarios):
            if i < len(dist_examples) and i < len(dir_examples):
                # Use different combinations to show variety
                dist_phrase = dist_examples[i % len(dist_examples)]
                dir_phrase = dir_examples[i % len(dir_examples)]

                # Format the answer appropriately based on directional style
                if response_style.dir_style == DirectionalStyle.CARTESIAN:
                    # For cartesian, the direction phrase already includes positioning
                    answer = f"{response_start} {dist_phrase}, {dir_phrase}."
                elif response_style.dir_style == DirectionalStyle.DEGREES:
                    # For degrees, put distance first then direction
                    answer = f"{response_start} {dist_phrase} {dir_phrase}."
                else:
                    # For clock and relative styles, natural order
                    answer = f"{response_start} {dist_phrase} {dir_phrase}."

                examples.append(f'Q: "{question}"\nA: "{answer}"')

        # If we have fewer scenarios than examples, add a few more with the first scenario
        while len(examples) < min(3, min(len(dist_examples), len(dir_examples))):
            idx = len(examples)
            question, response_start = example_scenarios[0]  # Use first scenario
            dist_phrase = dist_examples[idx % len(dist_examples)]
            dir_phrase = dir_examples[idx % len(dir_examples)]

            if response_style.dir_style == DirectionalStyle.CARTESIAN:
                answer = f"{response_start} {dist_phrase}, {dir_phrase}."
            elif response_style.dir_style == DirectionalStyle.DEGREES:
                answer = f"{response_start} {dist_phrase} {dir_phrase}."
            else:
                answer = f"{response_start} {dist_phrase} {dir_phrase}."

            examples.append(f'Q: "{question}"\nA: "{answer}"')

        return "\n".join(examples) + "\n"

    @classmethod
    def make_prompt(
        cls, response_style: ResponseStyle, enable_code_execution: bool
    ) -> str:
        """Generate the system prompt with the given response style."""
        # Instantiate to access pydantic field defaults
        tpl = cls()
        template = tpl.SYS_PROMPT_TEMPLATE
        examples = tpl._make_examples(response_style)
        return template.format(
            STYLE=response_style.prompt(),
            EXAMPLES=examples,
            CODE_EXECUTION=tpl.CODE_EXECUTION if enable_code_execution else "",
        )
