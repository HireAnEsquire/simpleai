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


class TagsExample(BaseModel):
    name: str
    tags: list[str] | None = None


def test_openai_schema_nullable_array_keeps_items_in_variant() -> None:
    """Regression: array schema made nullable must keep 'items' inside the
    anyOf variant, not as a sibling of anyOf.  OpenAI rejects the latter with
    'array schema missing items'."""
    schema = openai_response_schema(TagsExample)
    tags_schema = schema["properties"]["tags"]

    assert _is_nullable(tags_schema)

    # Find the array variant inside anyOf
    any_of = tags_schema.get("anyOf")
    assert any_of is not None

    array_variants = [v for v in any_of if v.get("type") == "array"]
    assert len(array_variants) == 1
    assert "items" in array_variants[0], (
        "items must be inside the array anyOf variant, not at the top level"
    )

    # Ensure 'items' is NOT a top-level sibling of anyOf
    assert "items" not in tags_schema, (
        "items must not be a sibling of anyOf"
    )
