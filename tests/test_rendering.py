"""Prompt-rendering tests for the 0.2.0 hint kwargs — exercised through the
private render helpers so we don't need a live LLM. They're private to keep
the public surface small, but covered here so behaviour is frozen.
"""
from __future__ import annotations

from ontonym_core import (
    Action,
    Class,
    ClassExtraction,
    Event,
    ObjectExtraction,
    ObjectInstance,
    Property,
    Relationship,
)
from ontonym_core.llm import (
    render_class_schema,
    render_known_classes,
    render_known_objects,
)

# ----------------------------------------------------------------------------
# candidate_class_names
# ----------------------------------------------------------------------------


def test_known_classes_without_candidates_unchanged():
    prior = ClassExtraction(
        classes=[
            Class(name="person"),
            Class(name="engineer", inherited_from="person"),
        ]
    )
    out = render_known_classes(prior)
    assert "STRONG CANDIDATES" not in out
    assert "person" in out
    assert "engineer (inherits person)" in out


def test_known_classes_with_candidates_renders_strong_candidates_preamble():
    prior = ClassExtraction(classes=[Class(name="person"), Class(name="team")])
    out = render_known_classes(prior, candidates=["person", "engineer"])
    assert out.startswith("STRONG CANDIDATES")
    assert "person, engineer" in out
    # full list still present after the preamble
    assert "All registered classes:" in out
    assert "team" in out


def test_known_classes_dedups_candidates_preserving_order():
    prior = ClassExtraction(classes=[Class(name="x")])
    out = render_known_classes(prior, candidates=["a", "b", "a", "c", "b"])
    # candidates section must list each once, in first-seen order
    cand_line = out.split("\n")[0]
    assert "a, b, c" in cand_line


def test_known_classes_empty_prior_with_candidates():
    out = render_known_classes(ClassExtraction(), candidates=["x", "y"])
    assert out.startswith("STRONG CANDIDATES")
    assert "x, y" in out
    assert "(none yet" in out  # the "no registered" fallback still shows


# ----------------------------------------------------------------------------
# render_class_schema — max_classes + class_mention_counts
# ----------------------------------------------------------------------------


def _schema_with_n_classes(n: int) -> ClassExtraction:
    classes = [Class(name=f"cls_{i}") for i in range(n)]
    props = [Property(class_name=f"cls_{i}", name="role", data_type="string") for i in range(n)]
    return ClassExtraction(
        classes=classes,
        properties=props,
        actions=[Action(name="run")],
        relationships=[Relationship(source="cls_0", target="cls_1", type="affects")],
    )


def test_class_schema_no_trim_when_under_max():
    schema = _schema_with_n_classes(5)
    out = render_class_schema(schema, max_classes=10)
    # All five rendered with property detail
    for i in range(5):
        assert f"cls_{i}(role:string)" in out
    assert "Less-mentioned classes" not in out


def test_class_schema_top_k_uses_mention_counts():
    schema = _schema_with_n_classes(6)
    # Top-3: cls_5, cls_4, cls_3 by counts
    counts = {"cls_5": 100, "cls_4": 50, "cls_3": 10}
    out = render_class_schema(schema, max_classes=3, class_mention_counts=counts)
    # The top-K block has parens on each (full property listing)
    assert "cls_5(role:string)" in out
    assert "cls_4(role:string)" in out
    assert "cls_3(role:string)" in out
    # The tail is name-only — no parens after cls_0/cls_1/cls_2
    assert "Less-mentioned classes" in out
    tail_section = out.split("Less-mentioned classes")[1]
    assert "cls_0" in tail_section
    assert "cls_0(" not in tail_section


def test_class_schema_tail_classes_appear_in_name_only_section():
    schema = _schema_with_n_classes(5)
    out = render_class_schema(schema, max_classes=2, class_mention_counts={})
    # No counts → ranking falls back to insertion order; first 2 in top block, rest in tail
    assert "Less-mentioned classes" in out
    # Action + relationship sections still rendered
    assert "ACTION NAMES" in out
    assert "RELATIONSHIP TYPES" in out


# ----------------------------------------------------------------------------
# candidate_object_names
# ----------------------------------------------------------------------------


def test_known_objects_without_candidates_unchanged():
    prior = ObjectExtraction(
        objects=[ObjectInstance(class_name="person", name="sarah")],
        events=[Event(class_name="incident", name="pd_9981")],
    )
    out = render_known_objects(prior)
    assert "STRONG CANDIDATE OBJECTS" not in out
    assert "person:sarah" in out
    assert "incident:pd_9981" in out


def test_known_objects_with_candidates_renders_preamble():
    prior = ObjectExtraction(objects=[ObjectInstance(class_name="person", name="alice")])
    out = render_known_objects(prior, candidates=["alice", "bob"])
    assert out.startswith("STRONG CANDIDATE OBJECTS")
    assert "alice, bob" in out
    # base list still rendered after the preamble
    assert "Objects:" in out
    assert "person:alice" in out


def test_known_objects_empty_prior_with_candidates():
    out = render_known_objects(ObjectExtraction(), candidates=["alice"])
    assert out.startswith("STRONG CANDIDATE OBJECTS")
    assert "alice" in out
    assert "(none yet)" in out
