import asyncio
import traceback
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

MODEL = "models/gemini-2.0-flash-live-001"
client = genai.Client(http_options={"api_version": "v1beta"})


def get_scene_description():
    """Gibt eine statische Szenenbeschreibung zurück."""
    print("\n[FUNCTION CALL] Führe get_scene_description() aus...")
    return {"scene": "Dies ist eine Test-Szenenbeschreibung aus einer Python-Funktion."}


def get_object_description(object_name: str):
    """Gibt eine Dummy-Beschreibung für ein benanntes Objekt zurück."""
    print(f"\n[FUNCTION CALL] Führe get_object_description('{object_name}') aus...")
    return {"description": f"Das Objekt '{object_name}' befindet sich auf zwei Uhr."}


def get_directions(destination: str):
    """Gibt eine Dummy-Wegbeschreibung zu einem Ziel zurück."""
    print(f"\n[FUNCTION CALL] Führe get_directions('{destination}') aus...")
    return {
        "route": f"Um zum Ziel '{destination}' zu gelangen, gehe 200 Meter geradeaus und biege dann an der großen Eiche links ab."}


tools = {
    'function_declarations': [
        {'name': 'get_scene_description',
         'description': "Gibt eine allgemeine Beschreibung der aktuellen Szene zurück."},
        {'name': 'get_object_description',
         'description': "Gibt eine detaillierte Beschreibung eines einzelnen Objekts in der Szene.",
         'parameters': {'type': 'object', 'properties': {
             'object_name': {'type': 'string', 'description': 'Objekt, das beschrieben werden soll.'}},
                        'required': ['object_name']}},
        {'name': 'get_directions', 'description': "Erstellt eine Wegbeschreibung zu einem bestimmten Ziel.",
         'parameters': {'type': 'object', 'properties': {
             'destination': {'type': 'string', 'description': 'Der Ort, zu dem die Wegbeschreibung führen soll.'}},
                        'required': ['destination']}}
    ]
}

CONFIG = {
    "response_modalities": ["TEXT"],
    "tools": [tools]
}


class TextChatLoop:
    def __init__(self):
        self.session = None

    async def handle_tool_call(self, tool_call):
        """Verarbeitet Funktionsaufrufe vom Modell."""
        function_responses = []
        for fc in tool_call.function_calls:
            result = None
            if fc.name == 'get_scene_description':
                result = get_scene_description()
            elif fc.name == 'get_object_description':
                result = get_object_description(fc.args.get('object_name', 'unbekanntes Objekt'))
            elif fc.name == 'get_directions':
                result = get_directions(fc.args.get('destination', 'unbekanntes Ziel'))

            if result:
                function_responses.append(types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": result},
                ))

        if function_responses:
            await self.session.send_tool_response(function_responses=function_responses)

    async def handle_model_responses(self):
        """Empfängt und verarbeitet Antworten (Text, Funktionsaufrufe) von der API."""
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
        """Haupt-Schleife für die Text-zu-Text-Konversation."""
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("--- Text-Chat gestartet (beenden mit 'q' oder 'exit') ---")

                while True:
                    user_input = await asyncio.to_thread(input, "Du: ")

                    if user_input.lower() in ["q", "exit", "quit"]:
                        print("\nProgramm wird beendet.")
                        break

                    if not user_input:
                        continue

                    await self.session.send(input=user_input, end_of_turn=True)

                    await self.handle_model_responses()

        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main = TextChatLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nProgramm wird beendet.")