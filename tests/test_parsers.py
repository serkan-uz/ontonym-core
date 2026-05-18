"""Smoke tests for the JSON parsers — no LLM required.

These freeze the parser's behaviour against frozen LLM-shaped JSON so a
refactor can't silently change the output structure. The parsers are the
"hardened against malformed output" layer between the LLM and the user's
Pydantic models, so they need their own tests independent of network calls.
"""
from __future__ import annotations

import json

from ontonym_core import (
    ClassExtraction,
    parse_class_json,
    parse_object_json,
)

# ----------------------------------------------------------------------------
# Class pass
# ----------------------------------------------------------------------------


def test_class_parser_basic():
    raw = json.dumps(
        {
            "classes": [
                {
                    "name": "person",
                    "description": "A human",
                    "inherited_from": None,
                    "properties": [{"name": "role", "data_type": "string"}],
                },
                {
                    "name": "engineer",
                    "description": "An engineer",
                    "inherited_from": "person",
                    "properties": [],
                },
            ],
            "actions": [
                {
                    "name": "deploy",
                    "actor": "engineer",
                    "target": "service",
                    "description": "Ship to prod",
                }
            ],
            "relationships": [
                {
                    "source": "engineer",
                    "target": "team",
                    "type": "works_in",
                    "description": None,
                }
            ],
            "rules": [
                {
                    "name": "engineer_has_role",
                    "description": "Every engineer has a role.",
                    "classes": ["engineer"],
                }
            ],
        }
    )
    result = parse_class_json(raw)
    assert len(result.classes) == 2
    assert result.classes[1].inherited_from == "person"
    assert len(result.properties) == 1
    assert result.properties[0].class_name == "person"
    assert result.properties[0].name == "role"
    assert len(result.actions) == 1
    assert result.actions[0].actor == "engineer"
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "works_in"
    assert len(result.rules) == 1
    assert result.rules[0].classes == ["engineer"]


def test_class_parser_handles_markdown_fences():
    raw = '```json\n{"classes": [{"name": "x", "description": null, "properties": []}], "actions": [], "relationships": [], "rules": []}\n```'
    result = parse_class_json(raw)
    assert result.classes[0].name == "x"


def test_class_parser_rejects_invalid_json():
    import pytest

    with pytest.raises(ValueError):
        parse_class_json("not json")


def test_class_parser_dedups_properties_within_class():
    raw = json.dumps(
        {
            "classes": [
                {
                    "name": "person",
                    "description": None,
                    "inherited_from": None,
                    "properties": [
                        {"name": "role", "data_type": "string"},
                        {"name": "role", "data_type": "string"},  # duplicate
                    ],
                }
            ],
            "actions": [],
            "relationships": [],
            "rules": [],
        }
    )
    result = parse_class_json(raw)
    assert len(result.properties) == 1


def test_class_parser_drops_rules_without_classes():
    raw = json.dumps(
        {
            "classes": [],
            "actions": [],
            "relationships": [],
            "rules": [
                {"name": "orphan", "description": "no targets", "classes": []},
                {"name": "valid", "description": "ok", "classes": ["x"]},
            ],
        }
    )
    result = parse_class_json(raw)
    assert len(result.rules) == 1
    assert result.rules[0].name == "valid"


# ----------------------------------------------------------------------------
# Object pass
# ----------------------------------------------------------------------------


def _schema_for_object_tests() -> ClassExtraction:
    return parse_class_json(
        json.dumps(
            {
                "classes": [
                    {
                        "name": "event",
                        "description": "actor-less happening",
                        "inherited_from": None,
                        "properties": [
                            {"name": "occurred_at", "data_type": "date"},
                            {"name": "summary", "data_type": "string"},
                        ],
                    },
                    {
                        "name": "incident",
                        "description": "an incident",
                        "inherited_from": "event",
                        "properties": [{"name": "severity", "data_type": "string"}],
                    },
                    {
                        "name": "person",
                        "description": "human",
                        "inherited_from": None,
                        "properties": [{"name": "role", "data_type": "string"}],
                    },
                ],
                "actions": [
                    {"name": "triage", "actor": "person", "target": "incident",
                     "description": None}
                ],
                "relationships": [
                    {"source": "incident", "target": "person", "type": "affected",
                     "description": None}
                ],
                "rules": [],
            }
        )
    )


def test_object_parser_routes_objects_and_events_correctly():
    schema = _schema_for_object_tests()
    raw = json.dumps(
        {
            "objects": [
                {"class_name": "person", "name": "sarah", "display_name": "Sarah",
                 "description": "On-call"},
            ],
            "events": [
                {"class_name": "incident", "name": "pd_9981", "display_name": "PD-9981",
                 "description": None},
            ],
            "properties": [
                {"class_name": "incident", "object_name": "pd_9981",
                 "name": "occurred_at", "value": "2026-05-05T14:02:00",
                 "data_type": "date"},
            ],
            "actions": [
                {"action_name": "triage", "actor": "sarah", "target": "pd_9981",
                 "description": None, "occurred_at": "2026-05-05T14:12:00"},
            ],
            "relationships": [
                {"source": "pd_9981", "target": "sarah", "type": "affected",
                 "description": None},
            ],
        }
    )
    result = parse_object_json(raw, schema)
    assert [o.name for o in result.objects] == ["sarah"]
    assert [e.name for e in result.events] == ["pd_9981"]
    assert len(result.properties) == 1
    assert result.properties[0].name == "occurred_at"
    assert len(result.actions) == 1
    assert len(result.relationships) == 1


def test_object_parser_drops_unknown_class_actions_rels():
    schema = _schema_for_object_tests()
    raw = json.dumps(
        {
            "objects": [
                {"class_name": "alien", "name": "x"},  # unknown class
                {"class_name": "person", "name": "ok"},
            ],
            "events": [
                {"class_name": "person", "name": "wrong"},  # not an event-class
            ],
            "properties": [
                {"class_name": "person", "object_name": "ok",
                 "name": "unknown_prop", "value": "x"},  # unknown property
            ],
            "actions": [
                {"action_name": "fly", "actor": "ok", "target": "ok"},  # unknown action
            ],
            "relationships": [
                {"source": "ok", "target": "ok", "type": "hates"},  # unknown type
            ],
        }
    )
    result = parse_object_json(raw, schema)
    assert [o.name for o in result.objects] == ["ok"]
    assert result.events == []
    assert result.properties == []
    assert result.actions == []
    assert result.relationships == []


def test_object_parser_routes_event_class_in_objects_to_events_pile():
    """If the LLM mis-files an event-class instance under objects[], the
    parser drops it rather than mis-classifying. The events[] array is the
    only legitimate home for an event-class instance."""
    schema = _schema_for_object_tests()
    raw = json.dumps(
        {
            "objects": [
                # incident inherits event — should not be in objects[].
                {"class_name": "incident", "name": "should_be_in_events"},
            ],
            "events": [],
            "properties": [],
            "actions": [],
            "relationships": [],
        }
    )
    result = parse_object_json(raw, schema)
    assert result.objects == []
    assert result.events == []


def test_object_parser_inherited_properties_allowed():
    """A property defined on `event` should be valid on instances of `incident`
    (which inherits event)."""
    schema = _schema_for_object_tests()
    raw = json.dumps(
        {
            "objects": [],
            "events": [
                {"class_name": "incident", "name": "pd_1"},
            ],
            "properties": [
                # `summary` is defined on event; incident inherits it.
                {"class_name": "incident", "object_name": "pd_1",
                 "name": "summary", "value": "boom", "data_type": "string"},
            ],
            "actions": [],
            "relationships": [],
        }
    )
    result = parse_object_json(raw, schema)
    assert len(result.properties) == 1
    assert result.properties[0].value == "boom"
