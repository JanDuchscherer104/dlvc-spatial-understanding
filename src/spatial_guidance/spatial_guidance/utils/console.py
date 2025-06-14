import json
import re
import traceback
from pathlib import Path
from typing import Any, Optional

from devtools import pformat
from rich.console import Console as RichConsole
from rich.theme import Theme


class Console(RichConsole):
    # TODO: get prefix automatically from caller (via caller stack?)
    is_debug: bool
    prefix: Optional[str] = None

    default_sttings = {
        "theme": Theme(
            {
                "config.name": "bold blue",  # Config class names
                "config.field": "green",  # Regular fields
                "config.propagated": "yellow",  # Propagated fields
                "config.value": "white",  # Field values
                "config.type": "dim",  # Type annotations
                "config.doc": "italic dim",  # Documentation
            }
        ),
        "width": 120,
        "force_terminal": True,
        "color_system": "auto",
        "markup": True,
        "highlight": True,
    }

    def __init__(self, **kwargs):
        settings = self.default_sttings.copy()
        settings.update(kwargs)
        super().__init__(**settings)
        self.is_debug = False
        self.verbose = True
        self.show_timestamps = False
        self.prefix = None

    @classmethod
    def with_prefix(cls, *parts: str) -> "Console":
        """
        Create a new Console instance with a custom prefix for all log messages.
        Enables builder-style chaining.

        Usage:
        ```python
        CONSOLE = Console.with_prefix(
            self.__class__.__name__,
            <name_of_the_current_method>
            <further_parts>, # eg. stage, worker_idx...
        )
        ```
        """
        instance = cls()
        instance.set_prefix(*parts)
        return instance

    def set_prefix(self, *parts: str) -> "Console":
        """
        Set a custom prefix for all log messages (e.g., class name + stage).
        Enables builder-style chaining.
        """
        if not parts:
            self.prefix = None
        else:
            self.prefix = "[/bold cyan][grey]::[/grey][bold cyan]".join(
                filter(None, parts)
            )

        return self

    def unset_prefix(self) -> "Console":
        """Unset the prefix for all log messages."""
        self.prefix = None
        return self

    def log(self, message: str) -> None:
        if self.verbose:
            self.print(self._format_message(message))

    def warn(self, message: str) -> None:
        if self.verbose:
            self.print(
                f"[bright_yellow]Warning:[/bright_yellow] {self._format_message(message)}\n"
                f"[dim]{self._get_caller_stack()}[/dim]"
            )

    def error(self, exception: Exception, message: Optional[str] = None) -> None:
        """Print error information including exception details and stack trace."""
        error_msg = message if message else str(exception)
        exception_type = type(exception).__name__

        # Format the error message with exception type and message
        formatted_error = f"[bright_red]Error ({exception_type}):[/bright_red] {self._format_message(error_msg)}"

        # Add the full exception details
        if str(exception) != error_msg:
            formatted_error += f"\n[dim]Exception: {str(exception)}[/dim]"

        # Add the full traceback if available
        if hasattr(exception, "__traceback__") and exception.__traceback__:
            tb_lines = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            formatted_error += f"\n[dim]Full traceback:\n{''.join(tb_lines)}[/dim]"
        else:
            # Fall back to caller stack if no traceback available
            formatted_error += f"\n[dim]{self._get_caller_stack()}[/dim]"

        self.print(formatted_error)

    def plog(self, obj: Any, title: Optional[str] = None, **kwargs) -> None:
        """Pretty print an object using rich."""
        if self.verbose:
            if title:
                self.log(f"[bold]{title}[/bold]")
            self.print(pformat(obj, **kwargs))

    def dbg(self, message: str) -> None:
        if self.is_debug:
            self.print(
                f"[bold magenta]Debug:[/bold magenta] {self._format_message(message)}"
            )

    def set_verbose(self, verbose: bool) -> "Console":
        self.verbose = verbose
        return self

    def set_debug(self, is_debug: bool) -> "Console":
        self.is_debug = is_debug
        self.is_verbose = self.verbose or is_debug
        return self

    def set_timestamp_display(self, show_timestamps: bool) -> "Console":
        self.show_timestamps = show_timestamps
        return self

    def _format_message(self, message: str) -> str:
        """Format message with optional timestamp and prefix."""
        prefix = f"\[[bold cyan]{self.prefix}[/bold cyan]]: " if self.prefix else ""
        if self.show_timestamps:
            return f"[{self._get_timestamp()}] {prefix}{message}"
        return f"{prefix}{message}"

    def _get_caller_stack(self) -> str:
        """Get formatted stack trace excluding Console internals"""
        stack = traceback.extract_stack()
        # Filter out frames from this file
        current_file = Path(__file__).resolve()
        relevant_frames = [
            frame
            for frame in stack[:-1]  # Exclude current frame
            if Path(frame.filename).resolve() != current_file
        ]
        # Format remaining frames
        return "".join(
            traceback.format_list(relevant_frames[-2:])
        )  # Show last 2 relevant frames

    def print_instructions(self, instructions: str) -> None:
        """
        Pretty print Pydantic format instructions, including the explanatory text and schema.

        Args:
            instructions: The full format instructions from PydanticOutputParser.get_format_instructions()
            title: Optional title to display above the output
        """
        try:
            # Extract the schema part
            schema_match = re.search(
                r"(The output should be formatted.*?)Here is the output schema:\s*```\s*(.*?)\s*```",
                instructions,
                re.DOTALL,
            )

            if schema_match:
                intro_text = schema_match.group(1).strip()
                schema_str = schema_match.group(2)

                # Print the explanatory text
                self.print(f"[yellow]{intro_text}[/yellow]\n")
                self.print("[bold]Here is the output schema:[/bold]")

                # Parse and format the schema
                schema = json.loads(schema_str)
                formatted = json.dumps(schema, indent=2)

                # Replace property patterns with colored highlights
                formatted = re.sub(
                    r'"(\w+)":\s*{', r'"[green]\1[/green]": {', formatted
                )
                formatted = re.sub(
                    r'"description":\s*"([^"]*)"',
                    r'"description": "[italic dim]\1[/italic dim]"',
                    formatted,
                )
                formatted = re.sub(
                    r'"type":\s*"([^"]*)"', r'"type": "[cyan]\1[/cyan]"', formatted
                )
                formatted = re.sub(
                    r'"title":\s*"([^"]*)"',
                    r'"title": "[yellow]\1[/yellow]"',
                    formatted,
                )

                self.print(formatted)
            else:
                self.print_schema(instructions)

        except json.JSONDecodeError as e:
            self.error(e, f"Failed to parse schema JSON: {instructions[:100]}...")
        except Exception as e:
            self.error(e, f"Error formatting instructions")

    def print_schema(self, schema_json: str) -> None:
        """
        Pretty print a JSON schema (like from PydanticOutputParser.get_format_instructions()).

        Args:
            schema_json: The JSON schema string to format
            title: Optional title to display above the schema
        """
        try:
            # Extract just the schema part from format instructions
            schema_match = re.search(
                r"Here is the output schema:\s*```\s*(.*?)\s*```",
                schema_json,
                re.DOTALL,
            )
            if schema_match:
                schema_str = schema_match.group(1)
            else:
                schema_str = schema_json

            # Parse the schema
            schema = json.loads(schema_str)

            # Format with indentation and color
            formatted = json.dumps(schema, indent=2)

            # Replace property patterns with colored highlights
            formatted = re.sub(r'"(\w+)":\s*{', r'"[green]\1[/green]": {', formatted)

            formatted = re.sub(
                r'"description":\s*"([^"]*)"',
                r'"description": "[italic dim]\1[/italic dim]"',
                formatted,
            )

            formatted = re.sub(
                r'"type":\s*"([^"]*)"', r'"type": "[cyan]\1[/cyan]"', formatted
            )

            formatted = re.sub(
                r'"title":\s*"([^"]*)"', r'"title": "[yellow]\1[/yellow]"', formatted
            )

            self.print(formatted)
        except json.JSONDecodeError as e:
            self.error(e, f"Failed to parse schema JSON: {schema_json[:100]}...")
        except Exception as e:
            self.error(e, f"Error formatting schema")
