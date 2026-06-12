"""ontonym-core — turn free text into a typed object graph.

Quickstart:

    import asyncio
    from ontonym_core import extract

    result = asyncio.run(extract(
        "Sarah deployed PaymentService to production at 14:02. "
        "An outage at 14:05 affected the payments flow.",
        backend="ollama",
    ))
    print(result.model_dump_json(indent=2))

See README.md for the full API.
"""
from __future__ import annotations

from .core import extract, extract_classes, extract_objects
from .coref import CorefResolver, get_coref_resolver
from .llm import (
    AnthropicBackend,
    Backend,
    DeepSeekBackend,
    OllamaBackend,
    parse_class_json,
    parse_object_json,
    render_class_schema,
    render_known_classes,
    render_known_objects,
    render_previous_context,
)
from .schema import (
    Action,
    Class,
    ClassExtraction,
    Event,
    Extraction,
    ObjectAction,
    ObjectExtraction,
    ObjectInstance,
    ObjectProperty,
    ObjectRelationship,
    Property,
    Relationship,
    Rule,
)

__version__ = "0.4.0"

__all__ = [
    # High-level API
    "extract",
    "extract_classes",
    "extract_objects",
    # Coreference pre-pass
    "CorefResolver",
    "get_coref_resolver",
    # Backends
    "Backend",
    "OllamaBackend",
    "AnthropicBackend",
    "DeepSeekBackend",
    # Parsers (for testing / custom flows)
    "parse_class_json",
    "parse_object_json",
    # Prompt rendering helpers (for callers building their own prompts on top)
    "render_known_classes",
    "render_previous_context",
    "render_class_schema",
    "render_known_objects",
    # Schema
    "Class",
    "Property",
    "Action",
    "Relationship",
    "Rule",
    "ClassExtraction",
    "ObjectInstance",
    "Event",
    "ObjectProperty",
    "ObjectAction",
    "ObjectRelationship",
    "ObjectExtraction",
    "Extraction",
]
