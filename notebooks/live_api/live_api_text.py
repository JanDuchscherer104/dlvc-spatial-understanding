import asyncio
import traceback
import warnings
from json import tool

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

from spatial_guidance.data_handling.stray_scanner.stray_dataset import (
    StrayDatasetConfig,
)

ds = StrayDatasetConfig().setup_target()

MODEL = "models/gemini-2.0-flash-live-001"
client = genai.Client(http_options={"api_version": "v1beta"})


def get_scene_description():
    """Returns a static scene description."""
    print("\n[FUNCTION CALL] Executing get_scene_description()...")
    return {"scene": "This is a test scene description from a Python function."}


def get_object_description(object_name: str):
    """Returns a dummy description for a named object."""
    print(f"\n[FUNCTION CALL] Executing get_object_description('{object_name}')...")
    return {"description": f"The object '{object_name}' is located at two o'clock."}


def get_directions(destination: str):
    """Returns dummy directions to a destination."""
    print(f"\n[FUNCTION CALL] Executing get_directions('{destination}')...")
    return {
        "route": f"To reach the destination '{destination}', go straight for 200 meters and then turn left at the large oak tree."
    }


tools = {
    "function_declarations": [
        {
            "name": "get_scene_description",
            "description": "Returns a general description of the current scene.",
        },
        {
            "name": "get_object_description",
            "description": "Returns a detailed description of a single object in the scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "Object to be described.",
                    }
                },
                "required": ["object_name"],
            },
        },
        {
            "name": "get_directions",
            "description": "Creates directions to a specific destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The location to which directions should be provided.",
                    }
                },
                "required": ["destination"],
            },
        },
    ]
}

CONFIG = {"response_modalities": ["TEXT"], "tools": [tools]}


class TextChatLoop:
    def __init__(self):
        self.session = None

    async def handle_tool_call(self, tool_call):
        """Processes function calls from the model."""
        print(type(tool_call))
        print(tool_call)
        function_responses = []
        for fc in tool_call.function_calls:
            result = None
            if fc.name == "get_scene_description":
                result = get_scene_description()
            elif fc.name == "get_object_description":
                result = get_object_description(
                    fc.args.get("object_name", "unknown object")
                )
            elif fc.name == "get_directions":
                result = get_directions(
                    fc.args.get("destination", "unknown destination")
                )

            if result:
                function_responses.append(
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"result": result},
                    )
                )

        if function_responses:
            await self.session.send_tool_response(function_responses=function_responses)

    async def handle_model_responses(self):
        """Receives and processes responses (text, function calls) from the API."""
        print("Model: ", end="")

        turn = self.session.receive()
        async for response in turn:
            if tool_call := response.tool_call:
                await self.handle_tool_call(tool_call)
                continue
            if text := response.text:
                print(text, end="", flush=True)
        print("\n")

    async def run(self):
        """Main loop for text-to-text conversation."""
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("--- Text chat started (exit with 'q' or 'exit') ---")

                frame_idx = 320
                # await self.session.send_realtime_input(
                #     activity_start=types.ActivityStart
                # )
                await self.session.send_realtime_input(media=ds[frame_idx].rgb_image)
                await self.session.send_realtime_input(
                    text=f"I'm now looking at frame {frame_idx} from the dataset. "
                    "I'm seeing what the visually imparied user is seeing. ",
                )

                while True:
                    user_input = await asyncio.to_thread(input, "You: ")

                    if user_input.lower() in ["q", "exit", "quit"]:
                        print("\nProgram is exiting.")
                        break

                    if not user_input:
                        continue

                    # await self.session.send(input=user_input, end_of_turn=True)
                    # self.session.send_realtime_input(activity_start=types.ActivityStart)
                    await self.session.send_realtime_input(text=user_input)
                    # await self.session.send_realtime_input(
                    #     activity_end=types.ActivityEnd
                    # )

                    await self.handle_model_responses()

        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main = TextChatLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nProgram is exiting.")
