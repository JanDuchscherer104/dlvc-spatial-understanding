import abc
import json
from typing import Annotated, Any, Dict, List, Optional, Self

from pydantic import BaseModel, ConfigDict, Field
from pydantic.type_adapter import TypeAdapter

try:
    import orjson as _fast_json

    json_loads = _fast_json.loads
except ImportError:
    json_loads = json.loads

from ..utils.console import Console


class DataModel(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    @classmethod
    def get_json_schema(
        cls,
        with_examples: bool = False,
        as_list: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON schema for the model, optionally as a list,
        and with or without examples and titles.

        Args:
            with_examples (bool): If True, include example values for fields in the schema.
            as_list (bool): If True, generate schema for a list of instances of the class.
            min_length (int, optional): Minimum length of the output list if as_list is True.
            max_length (int, optional): Maximum length of the output list if as_list is True.
        Returns:
            Dict[str, Any]: JSON schema for the model.
        """
        if as_list:
            field_args = {}
            if min_length is not None:
                field_args["min_length"] = min_length
            if max_length is not None:
                field_args["max_length"] = max_length

            if field_args:
                constrained_list_type = Annotated[List[cls], Field(**field_args)]
                schema = TypeAdapter(constrained_list_type).json_schema()
            else:
                schema = TypeAdapter(List[cls]).json_schema()
        else:
            schema = cls.model_json_schema()

        # Helper to recursively remove "title" fields
        def remove_titles(node: Any) -> None:
            if isinstance(node, dict):
                node.pop("title", None)
                for value in node.values():
                    remove_titles(value)
            elif isinstance(node, list):
                for item in node:
                    remove_titles(item)

        remove_titles(schema)

        # Helper to conditionally remove "examples" fields
        if not with_examples:

            def remove_examples(node: Any) -> None:
                if isinstance(node, dict):
                    node.pop("examples", None)
                    for key, value in list(
                        node.items()
                    ):  # Iterate over a copy for safe modification
                        if key == "properties":
                            for prop_value in value.values():
                                if isinstance(prop_value, dict):
                                    prop_value.pop("examples", None)
                                remove_examples(prop_value)
                        elif key == "items":
                            if isinstance(value, dict):
                                value.pop("examples", None)
                            remove_examples(value)
                        elif isinstance(value, (dict, list)):
                            remove_examples(value)
                elif isinstance(node, list):
                    for item in node:
                        remove_examples(item)

            remove_examples(schema)

        return schema

    @classmethod
    def get_output_instructions(
        cls,
        with_examples: bool = False,
        as_list: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        readable: bool = False,
    ) -> str:
        """
        Generate formatted instructions for model output: schema (potentially with examples).
        Args:
            with_examples (bool): If True, include example values for fields in the schema.
            as_list (bool): If True, format the output as a list of JSON objects.
            min_length (int, optional): Minimum length of the output list if as_list is True.
            max_length (int, optional): Maximum length of the output list if as_list is True.
            readable (bool): If True, format the JSON schema for better human readability.
        Returns:
            str: Formatted instructions for the model output.
        """
        schema_dict = cls.get_json_schema(
            with_examples=with_examples,
            as_list=as_list,
            min_length=min_length,
            max_length=max_length,
        )

        schema_str = json.dumps(schema_dict, indent=2 if readable else None)

        if as_list:
            instruction_intro = (
                "The output should be formatted as a JSON list. "
                "This list must conform to the JSON schema provided below. "
                "The schema defines the structure of the list and the items it contains.\n"
            )
        else:
            instruction_intro = "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n"

        # Adjust the hint about examples based on whether they are actually in schema_str
        # This is implicitly handled by how schema_str is generated by get_json_schema
        examples_hint_text = (
            "includes examples " if with_examples else "does NOT include examples "
        )

        instruction_text_template = (
            instruction_intro
            + f"The schema {examples_hint_text}for fields where provided.\n\n"
        )

        if readable:
            instruction = instruction_text_template + schema_str
        else:
            instruction = (
                instruction_text_template + "```json\n" + schema_str + "\n```\n"
            )

        if (
            with_examples and not readable
        ):  # Only add specific note if examples are included and it's for LLM
            instruction += "\nPlease pay close attention to the example formats provided within the schema for each field."

        return instruction

    @classmethod
    def parse_json_list(
        cls, json_output: str, console: Optional[Console] = None
    ) -> list[Self]:
        console = console or Console()

        if not json_output or json_output.isspace():
            console.warn("Received empty or whitespace-only JSON output.")
            return []

        # Extract JSON array by locating the first '[' and last ']'
        s = json_output
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
        else:
            s = s.strip()

        if not s:
            console.warn("JSON output became empty after trimming.")
            return []

        try:
            parsed_data = json_loads(s)
        except Exception as e:
            console.error(f"Failed to load JSON: {e}")
            return []

        # Handle wrapped objects list or unexpected types
        if (
            isinstance(parsed_data, dict)
            and "objects" in parsed_data
            and isinstance(parsed_data["objects"], list)
        ):
            parsed_data = parsed_data["objects"]
        elif not isinstance(parsed_data, list):
            console.error(f"Expected a list, but got {type(parsed_data)}.")
            return []

        # Validate and collect items
        validated = []
        for item in parsed_data:
            try:
                validated.append(cls.model_validate(item))
            except Exception as e_item:
                console.error(f"Validation error for item {item}: {e_item}")

        if not validated and parsed_data:
            console.warn("All items failed validation, returning empty list.")

        return validated


import abc
import json
from typing import Annotated, Any, Dict, List, Optional, Self

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic.type_adapter import TypeAdapter

try:
    import orjson as _fast_json

    json_loads = _fast_json.loads
except ImportError:
    json_loads = json.loads

from ..utils.console import Console


class DataModel(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    @classmethod
    def get_json_schema(
        cls,
        with_examples: bool = False,
        as_list: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON schema for the model, optionally as a list,
        and with or without examples and titles.

        Args:
            with_examples (bool): If True, include example values for fields in the schema.
            as_list (bool): If True, generate schema for a list of instances of the class.
            min_length (int, optional): Minimum length of the output list if as_list is True.
            max_length (int, optional): Maximum length of the output list if as_list is True.
        Returns:
            Dict[str, Any]: JSON schema for the model.
        """
        if as_list:
            field_args = {}
            if min_length is not None:
                field_args["min_length"] = min_length
            if max_length is not None:
                field_args["max_length"] = max_length

            if field_args:
                constrained_list_type = Annotated[List[cls], Field(**field_args)]
                schema = TypeAdapter(constrained_list_type).json_schema()
            else:
                schema = TypeAdapter(List[cls]).json_schema()
        else:
            schema = cls.model_json_schema()

        # Helper to recursively remove "title" fields
        def remove_titles(node: Any) -> None:
            if isinstance(node, dict):
                node.pop("title", None)
                for value in node.values():
                    remove_titles(value)
            elif isinstance(node, list):
                for item in node:
                    remove_titles(item)

        remove_titles(schema)

        # Helper to conditionally remove "examples" fields
        if not with_examples:

            def remove_examples(node: Any) -> None:
                if isinstance(node, dict):
                    node.pop("examples", None)
                    for key, value in list(
                        node.items()
                    ):  # Iterate over a copy for safe modification
                        if key == "properties":
                            for prop_value in value.values():
                                if isinstance(prop_value, dict):
                                    prop_value.pop("examples", None)
                                remove_examples(prop_value)
                        elif key == "items":
                            if isinstance(value, dict):
                                value.pop("examples", None)
                            remove_examples(value)
                        elif isinstance(value, (dict, list)):
                            remove_examples(value)
                elif isinstance(node, list):
                    for item in node:
                        remove_examples(item)

            remove_examples(schema)

        return schema

    @classmethod
    def get_output_instructions(
        cls,
        with_examples: bool = False,
        as_list: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        readable: bool = False,
    ) -> str:
        """
        Generate formatted instructions for model output: schema (potentially with examples).
        Args:
            with_examples (bool): If True, include example values for fields in the schema.
            as_list (bool): If True, format the output as a list of JSON objects.
            min_length (int, optional): Minimum length of the output list if as_list is True.
            max_length (int, optional): Maximum length of the output list if as_list is True.
            readable (bool): If True, format the JSON schema for better human readability.
        Returns:
            str: Formatted instructions for the model output.
        """
        schema_dict = cls.get_json_schema(
            with_examples=with_examples,
            as_list=as_list,
            min_length=min_length,
            max_length=max_length,
        )

        schema_str = json.dumps(schema_dict, indent=2 if readable else None)

        if as_list:
            instruction_intro = (
                "The output should be formatted as a JSON list. "
                "This list must conform to the JSON schema provided below. "
                "The schema defines the structure of the list and the items it contains.\n"
            )
        else:
            instruction_intro = "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n"

        # Adjust the hint about examples based on whether they are actually in schema_str
        # This is implicitly handled by how schema_str is generated by get_json_schema
        examples_hint_text = (
            "includes examples " if with_examples else "does NOT include examples "
        )

        instruction_text_template = (
            instruction_intro
            + f"The schema {examples_hint_text}for fields where provided.\n\n"
        )

        if readable:
            instruction = instruction_text_template + schema_str
        else:
            instruction = (
                instruction_text_template + "```json\n" + schema_str + "\n```\n"
            )

        if (
            with_examples and not readable
        ):  # Only add specific note if examples are included and it's for LLM
            instruction += "\nPlease pay close attention to the example formats provided within the schema for each field."

        return instruction

    @classmethod
    def parse_json_list(
        cls, json_output: str, console: Optional[Console] = None
    ) -> list[Self]:
        console = console or Console()

        if not json_output or json_output.isspace():
            console.warn("Received empty or whitespace-only JSON output.")
            return []

        # Extract JSON array by locating the first '[' and last ']'
        s = json_output
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
        else:
            s = s.strip()

        if not s:
            console.warn("JSON output became empty after trimming.")
            return []

        try:
            parsed_data = json_loads(s)
        except Exception as e:
            console.error(f"Failed to load JSON: {e}")
            return []

        # Handle wrapped objects list or unexpected types
        if (
            isinstance(parsed_data, dict)
            and "objects" in parsed_data
            and isinstance(parsed_data["objects"], list)
        ):
            parsed_data = parsed_data["objects"]
        elif not isinstance(parsed_data, list):
            console.error(f"Expected a list, but got {type(parsed_data)}.")
            return []

        # Validate and collect items
        validated = []
        for item in parsed_data:
            try:
                validated.append(cls.model_validate(item))
            except Exception as e_item:
                console.error(f"Validation error for item {item}: {e_item}")

        if not validated and parsed_data:
            console.warn("All items failed validation, returning empty list.")

        return validated
