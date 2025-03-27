from enum import Enum, auto
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Sequence, Set, Type, Union

import torch
from pydantic import Field

from utils import BaseConfig, PathConfig


class WhisperTranscriber(Protocol):
    """Protocol for Whisper transcriber."""

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file."""
        ...


class OpenAIClient(Protocol):
    """Protocol for OpenAI client with audio transcription capability."""

    class audio:
        class transcriptions:
            @staticmethod
            def create(file: Any, **kwargs) -> Any:
                """Transcribe audio file with the OpenAI API v1.0+."""
                ...


class STTProvider(Enum):
    """Speech-to-Text providers."""

    OPENAI_WHISPER_LOCAL = auto()
    OPENAI_WHISPER_API = auto()

    @classmethod
    def setup_target(
        cls, config: "SpeechToTextConfig"
    ) -> Union[WhisperTranscriber, OpenAIClient]:
        """Set up the appropriate transcription engine based on provider."""
        match config.provider:
            case cls.OPENAI_WHISPER_LOCAL:
                try:
                    import whisper

                    model = whisper.load_model(
                        config.whisper_config.model_name,
                        device=config.whisper_config.device,
                    )
                    return model
                except ImportError:
                    raise ImportError(
                        "OpenAI Whisper is required. Install with: `pip install git+https://github.com/openai/whisper.git` "
                    )

            case cls.OPENAI_WHISPER_API:
                try:
                    from openai import OpenAI

                    client = OpenAI(api_key=config.whisper_api_config.api_key)
                    return client
                except ImportError:
                    raise ImportError(
                        "OpenAI Python client is required. Install with: `pip install openai`"
                    )

            case _:
                raise ValueError(f"Unsupported STT provider: {config.provider}")

    def transcribe(
        self,
        transcriber: Union[WhisperTranscriber, OpenAIClient],
        config: "SpeechToTextConfig",
        audio_path: Path,
    ) -> str:
        """Transcribe audio based on provider type."""
        match self:
            case self.OPENAI_WHISPER_LOCAL:
                provider_config = config.whisper_config

                result = transcriber.transcribe(
                    str(audio_path),
                    language=provider_config.language,
                    fp16=(provider_config.precision == "float16"),
                )
                return str(result["text"])

            case self.OPENAI_WHISPER_API:
                provider_config = config.whisper_api_config

                with open(audio_path, "rb") as audio_file:
                    # Using the new OpenAI API structure (v1.0+)
                    response = transcriber.audio.transcriptions.create(
                        model=provider_config.model,
                        file=audio_file,
                        response_format=provider_config.response_format,
                        temperature=provider_config.temperature,
                        language=provider_config.language,
                        prompt=provider_config.prompt,
                    )

                # Handle different response formats based on API behavior
                if hasattr(response, "text"):
                    # Object with text attribute
                    return str(response.text)
                elif isinstance(response, str):
                    # Direct string response
                    return response
                elif isinstance(response, dict) and "text" in response:
                    # Dictionary with text key
                    return str(response["text"])
                else:
                    # Fallback for unexpected response format
                    return str(response)

            case _:
                # This case should never be reached due to validation when creating the enum
                # but we need it for type checking
                raise ValueError(f"Unsupported STT provider: {self}")


class WhisperConfig(BaseConfig):
    """Configuration for local Whisper model."""

    model_name: Literal["tiny", "base", "small", "medium", "large"] = "medium"
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Literal["float16", "float32"] = (
        "float16" if torch.cuda.is_available() else "float32"
    )
    language: Optional[str] = None  # None for auto-detection


class WhisperAPIConfig(BaseConfig):
    """Configuration for OpenAI Whisper API."""

    api_key: str = ""
    model: Literal["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"] = (
        "gpt-4o-transcribe"
    )
    response_format: Literal["json", "text"] = "text"
    temperature: float = 0.0
    language: Optional[str] = None  # None for auto-detection
    prompt: str = ""  # Optional prompt to guide transcription

    def setup_target(self) -> "WhisperAPIConfig":
        """Set up the API key from PathConfig."""
        path_config = PathConfig()
        api_key = path_config.get_api_key("OPENAI_API_KEY")
        if api_key:
            self.api_key = api_key
        return self


class AudioConfig(BaseConfig):
    """Combined configuration for audio files and voice messages."""

    # Voice messages config
    voice_dir_name: str = "voice_messages"
    create_voice_dir: bool = True
    voice_dir: Optional[Path] = None

    # File handling config
    audio_formats: Set[str] = {
        "m4a",
        "mp3",
        "wav",
        "ogg",
        "flac",
        "m4a",
        "mp4",
        "mpeg",
        "mpga",
        "webm",
    }
    output_dir_name: Optional[str] = None  # Relative to PathConfig().data
    output_format: Literal["txt", "json", "srt"] = "txt"
    output_dir: Optional[Path] = None

    def setup_target(self) -> "AudioConfig":
        """Set up all directories using PathConfig."""
        path_config = PathConfig()

        # Set up voice directory
        self.voice_dir = path_config.data / self.voice_dir_name
        if self.create_voice_dir and self.voice_dir is not None:
            self.voice_dir.mkdir(parents=True, exist_ok=True)

        # Set up output directory if specified
        if self.output_dir_name:
            self.output_dir = path_config.data / self.output_dir_name
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)

        return self

    def validate_directory(self, directory: Union[str, Path]) -> Path:
        """Validate that a directory exists."""
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        return directory_path

    def find_audio_files(self, directory: Path) -> list[Path]:
        """Find all supported audio files in directory."""
        return list(
            chain.from_iterable(
                directory.glob(f"*.{fmt}") for fmt in self.audio_formats
            )
        )


class SpeechToTextConfig(BaseConfig["SpeechToTextEngine"]):
    """Configuration for Speech-to-Text (STT) engine with multilingual support."""

    target: Type["SpeechToTextEngine"] = Field(
        default_factory=lambda: SpeechToTextEngine
    )
    provider: STTProvider = Field(default=STTProvider.OPENAI_WHISPER_API)

    # Provider-specific configs
    whisper_config: WhisperConfig = Field(default_factory=WhisperConfig)
    whisper_api_config: WhisperAPIConfig = Field(default_factory=WhisperAPIConfig)

    # Combined audio config
    audio_config: AudioConfig = Field(default_factory=AudioConfig)

    def setup_target(self) -> "SpeechToTextEngine":
        """Set up the engine with proper configuration."""
        # Ensure API keys are set
        if self.provider == STTProvider.OPENAI_WHISPER_API:
            self.whisper_api_config = self.whisper_api_config.setup_target()
        else:
            raise NotImplementedError("Only OpenAI Whisper API is supported for now")

        # Setup audio config
        self.audio_config = self.audio_config.setup_target()

        return self.target(self)


class SpeechToTextEngine:
    """Speech-to-Text engine with multilingual transcription capabilities."""

    def __init__(self, config: SpeechToTextConfig):
        self.config = config
        self.transcriber: Union[WhisperTranscriber, OpenAIClient] = (
            STTProvider.setup_target(config)
        )

    def transcribe_file(self, audio_path: Optional[Union[str, Path]] = None) -> str:
        """Transcribe a single audio file with language auto-detection.

        If no path is provided, uses the first matching audio file in the voice_dir.
        """
        # If no path provided, use default from config
        if audio_path is None:
            voice_dir = self.config.audio_config.voice_dir
            if voice_dir is None:
                raise ValueError(
                    "No voice directory configured and no audio path provided"
                )

            # Find first matching audio file
            audio_files = self.config.audio_config.find_audio_files(voice_dir)
            if not audio_files:
                raise FileNotFoundError(f"No audio files found in {voice_dir}")

            audio_path = audio_files[0]

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use the provider's transcribe method
        return self.config.provider.transcribe(
            self.transcriber, self.config, audio_path
        )

    def transcribe_directory(
        self, directory: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """Transcribe all supported audio files in a directory.

        If no directory is provided, uses the voice_dir from config.
        """
        # If no directory provided, use default from config
        if directory is None:
            voice_dir = self.config.audio_config.voice_dir
            if voice_dir is None:
                raise ValueError(
                    "No voice directory configured and no directory path provided"
                )
            directory = voice_dir

        # Use config to validate directory
        directory_path = self.config.audio_config.validate_directory(directory)

        # Find all supported audio files using itertools.chain
        audio_files = self.config.audio_config.find_audio_files(directory_path)

        return self.batch_transcribe(audio_files)

    def batch_transcribe(
        self, audio_paths: Optional[Sequence[Union[str, Path]]] = None
    ) -> Dict[str, str]:
        """Transcribe multiple audio files with language auto-detection.

        If no paths are provided, transcribes all files in the voice_dir.
        """
        # If no paths provided, use all files from voice_dir
        if audio_paths is None:
            voice_dir = self.config.audio_config.voice_dir
            if voice_dir is None:
                raise ValueError(
                    "No voice directory configured and no audio paths provided"
                )

            audio_paths = self.config.audio_config.find_audio_files(voice_dir)
            if not audio_paths:
                raise FileNotFoundError(f"No audio files found in {voice_dir}")

        # Use map for conversion
        path_list = list(map(Path, audio_paths))

        # Validate all files exist using functional approach
        invalid_paths = list(filter(lambda p: not p.exists(), path_list))
        if invalid_paths:
            raise FileNotFoundError(f"Audio file(s) not found: {invalid_paths}")

        results: Dict[str, str] = {}

        # Process files one by one
        for audio_path in path_list:
            text = self.transcribe_file(audio_path)
            results[str(audio_path)] = text

            # Save output if configured
            self._save_transcription(audio_path, text)

        return results

    def _save_transcription(self, audio_path: Path, text: str) -> None:
        """Save transcription text to a file if output directory is configured."""
        output_dir = self.config.audio_config.output_dir
        if output_dir is None:
            return

        output_file = (
            output_dir / f"{audio_path.stem}.{self.config.audio_config.output_format}"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
