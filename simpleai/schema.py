"""Provider-safe JSON schema helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from pydantic import BaseModel

# Anthropic output schemas currently reject these JSON Schema keywords.
# Source: docs.anthropic.com (Structured outputs limitations), accessed 2026-02-06.
ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def output_model_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Return the Pydantic-generated JSON schema for an output model."""

    return output_format.model_json_schema()


def enforce_closed_objects(schema: dict[str, Any]) -> dict[str, Any]:
    """Set additionalProperties=false on all object-like schema nodes."""

    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            looks_objectish = any(
                key in node
                for key in ("properties", "required", "patternProperties", "additionalProperties")
            )
            if is_object or looks_objectish:
                node["additionalProperties"] = False

            for value in node.values():
                walk(value)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(normalized)
    return normalized


def strip_schema_keywords(schema: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Remove unsupported JSON Schema keywords recursively."""

    keys_set = set(keys)
    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in list(node.keys()):
                if key in keys_set:
                    node.pop(key, None)
                    continue
                walk(node[key])
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(normalized)
    return normalized


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema variant that accepts null as a value."""

    node = deepcopy(schema)

    # Already nullable.
    node_type = node.get("type")
    if node_type == "null":
        return node
    if isinstance(node_type, list) and "null" in node_type:
        return node

    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        if not any(isinstance(item, dict) and item.get("type") == "null" for item in any_of):
            any_of.append({"type": "null"})
        return node

    one_of = node.get("oneOf")
    if isinstance(one_of, list):
        if not any(isinstance(item, dict) and item.get("type") == "null" for item in one_of):
            # Prefer anyOf once nullability is introduced.
            node["anyOf"] = one_of + [{"type": "null"}]
            node.pop("oneOf", None)
        return node

    # OpenAI Structured Outputs do not support type: [type, "null"].
    # They require anyOf: [{type: type}, {type: "null"}].
    if isinstance(node_type, str):
        new_node = deepcopy(node)
        new_node.pop("type", None)
        return {"anyOf": [{"type": node_type}, {"type": "null"}], **new_node}

    if isinstance(node_type, list):
        new_node = deepcopy(node)
        new_node.pop("type", None)
        types = [{"type": t} for t in node_type if t != "null"] + [{"type": "null"}]
        return {"anyOf": types, **new_node}

    # No explicit type/union: wrap in anyOf.
    return {"anyOf": [node, {"type": "null"}]}


def enforce_openai_required_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode requires all object properties to be listed in required."""

    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            looks_objectish = any(
                key in node
                for key in ("properties", "required", "patternProperties", "additionalProperties")
            )

            if is_object or looks_objectish:
                if "properties" in node and isinstance(node["properties"], dict):
                    properties = node["properties"]
                    required = set(node.get("required") or [])
                    all_keys = list(properties.keys())
                    for key in all_keys:
                        if key not in required and isinstance(properties.get(key), dict):
                            properties[key] = _make_nullable(properties[key])
                    node["required"] = all_keys
                else:
                    # Even if no properties are present, OpenAI wants 'required' array.
                    if "required" not in node:
                        node["required"] = []

            for value in node.values():
                walk(value)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(normalized)
    return normalized


def openai_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build strict-schema payload for OpenAI Responses API."""

    schema = output_model_schema(output_format)
    schema = enforce_closed_objects(schema)
    schema = enforce_openai_required_all_properties(schema)
    # OpenAI Structured Outputs (strict mode) does not support 'default' or 'title'.
    # We strip these LAST to ensure we catch everything.
    return strip_schema_keywords(schema, ["default", "title"])


def anthropic_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build output schema compatible with Anthropic output_config constraints."""

    schema = enforce_closed_objects(output_model_schema(output_format))
    return strip_schema_keywords(schema, ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS)


def perplexity_response_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """Build JSON schema payload for Perplexity responses."""

    return enforce_closed_objects(output_model_schema(output_format))
