import asyncio
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

import pyaudio
from google import genai
from google.genai import types

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

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
        {
            'name': 'get_object_description',
            'description': "Gibt eine detaillierte Beschreibung eines einzelnen Objekts in der Szene.",
            'parameters': {
                'type': 'object',
                'properties': {'object_name': {'type': 'string',
                                               'description': 'Der Name des Objekts, das beschrieben werden soll.'}},
                'required': ['object_name']
            }
        },
        {
            'name': 'get_directions',
            'description': "Erstellt eine Wegbeschreibung zu einem bestimmten Ziel.",
            'parameters': {
                'type': 'object',
                'properties': {'destination': {'type': 'string',
                                               'description': 'Der Ort, zu dem die Wegbeschreibung führen soll.'}},
                'required': ['destination']
            }
        }
    ]
}

CONFIG = {
    "response_modalities": ["AUDIO"],
    "tools": [tools]
}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.audio_stream = None

    async def send_text(self):
        """Ermöglicht das Senden von Textnachrichten an das Modell."""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            print(f"[INFO] Sende Text an API: '{text}'")
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        """Sendet die Audiodaten aus der Warteschlange an die API."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """Nimmt Audio vom Mikrofon auf und stellt es in die Warteschlange."""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

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

    async def receive_audio(self):
        """Empfängt Antworten (Audio, Text, Funktionsaufrufe) von der API."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if tool_call := response.tool_call:
                    await self.handle_tool_call(tool_call)
                    continue
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(f"\n[MODEL OUTPUT] {text}", end="")

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Spielt das empfangene Audio ab."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """Haupt-Schleife, die alle asynchronen Aufgaben startet und verwaltet."""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("Benutzer hat das Programm beendet.")
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(eg)


if __name__ == "__main__":
    try:
        main = AudioLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nProgramm wird beendet.")
    finally:
        pya.terminate()