"""Pydantic models for ontonym-core extraction output.

Stateless — no surrogate ids, no FK ints, no approval status, no tenant scoping.
Cross-references are by NAME (snake_case string). This matches the LLM's raw
output shape and lets callers consume the result without any persistence layer.

Two top-level containers:
  - `ClassExtraction`: schema-level — classes, their properties, actions,
    relationships, rules.
  - `ObjectExtraction`: instance-level — objects, events (actor-less
    happenings), object_properties, object_actions, object_relationships.

Both are fully serialisable to JSON via `.model_dump_json()`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Class layer
# ----------------------------------------------------------------------------


class Class(BaseModel):
    """A kind of thing — `person`, `database`, `incident`. Never an instance."""

    name: str = Field(..., description="snake_case English name.")
    description: str | None = None
    inherited_from: str | None = Field(
        None, description="Parent class name when this class IS-A specialisation; single-parent."
    )


class Property(BaseModel):
    """A property a class has — `(class_name=person, name=role, data_type=string)`."""

    class_name: str
    name: str
    data_type: str | None = None


class Action(BaseModel):
    """An action one class performs — `(actor=engineer, name=deploy, target=service)`."""

    name: str
    actor: str | None = None
    target: str | None = None
    description: str | None = None


class Relationship(BaseModel):
    """A typed link between two classes — `(source=person, type=works_in, target=team)`."""

    source: str
    target: str
    type: str
    description: str | None = None


class Rule(BaseModel):
    """A class-level constraint, policy, or invariant applying to one or more classes."""

    name: str
    description: str | None = None
    classes: list[str] = Field(default_factory=list)


class ClassExtraction(BaseModel):
    """Class-level result of one extraction pass."""

    classes: list[Class] = Field(default_factory=list)
    properties: list[Property] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Object layer
# ----------------------------------------------------------------------------


class ObjectInstance(BaseModel):
    """A specific instance of a class — `(class_name=person, name=sarah)`."""

    class_name: str
    name: str = Field(..., description="Canonical snake_case identifier.")
    display_name: str | None = None
    description: str | None = None


class Event(ObjectInstance):
    """An object whose class inherits from `event` — actor-less time-anchored happening."""

    pass


class ObjectProperty(BaseModel):
    """A specific property value on a specific object."""

    class_name: str
    object_name: str
    name: str
    value: str | None = None
    data_type: str | None = None


class ObjectAction(BaseModel):
    """A specific occurrence of an action between specific objects."""

    action_name: str
    actor: str | None = None
    target: str | None = None
    description: str | None = None
    occurred_at: str | None = Field(
        None, description="ISO 8601 datetime when known; left as the LLM-emitted string."
    )


class ObjectRelationship(BaseModel):
    """A specific typed link between two specific objects."""

    source: str
    target: str
    type: str
    description: str | None = None


class ObjectExtraction(BaseModel):
    """Object-level result of one extraction pass."""

    objects: list[ObjectInstance] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    properties: list[ObjectProperty] = Field(default_factory=list)
    actions: list[ObjectAction] = Field(default_factory=list)
    relationships: list[ObjectRelationship] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Combined
# ----------------------------------------------------------------------------


class Extraction(BaseModel):
    """Combined class + object extraction result for one `extract(..., mode="both")` call."""

    classes: ClassExtraction = Field(default_factory=ClassExtraction)
    objects: ObjectExtraction = Field(default_factory=ObjectExtraction)
