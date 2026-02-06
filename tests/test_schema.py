from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from simpleai.schema import openai_response_schema


class OpenAIStrictExample(BaseModel):
    required_value: int
    optional_text: str | None = None
    optional_number: int | None = None


def _is_nullable(node: dict[str, Any]) -> bool:
    node_type = node.get("type")
    if node_type == "null":
        return True
    if isinstance(node_type, list) and "null" in node_type:
        return True
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        return any(isinstance(item, dict) and item.get("type") == "null" for item in any_of)
    return False


def test_openai_schema_requires_all_object_properties_and_nullable_optionals() -> None:
    schema = openai_response_schema(OpenAIStrictExample)
    props = schema["properties"]

    assert set(schema["required"]) == set(props.keys())
    assert _is_nullable(props["optional_text"])
    assert _is_nullable(props["optional_number"])
