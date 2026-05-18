"""High-level stateless extraction API.

Three async entry points:

  - `extract_classes(text, *, backend=..., prior=...)` — class-level pass.
  - `extract_objects(text, schema, *, backend=..., prior=...)` — object-level pass.
  - `extract(text, *, mode="both", backend=..., prior_classes=..., prior_objects=...)` —
    convenience: runs the class pass, then the object pass against the result.

`backend` accepts either a Backend instance (`OllamaBackend(...)` etc.) or one of
the strings `"ollama"` / `"anthropic"`, in which case a default backend is
constructed from environment variables.
"""
from __future__ import annotations

import os
from typing import Literal

from .llm import AnthropicBackend, Backend, OllamaBackend
from .schema import ClassExtraction, Extraction, ObjectExtraction

BackendName = Literal["ollama", "anthropic"]
BackendLike = Backend | BackendName


def _resolve_backend(backend: BackendLike) -> Backend:
    if isinstance(backend, str):
        if backend == "ollama":
            return OllamaBackend(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            )
        if backend == "anthropic":
            return AnthropicBackend(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            )
        raise ValueError(
            f"Unknown backend name {backend!r}. Use 'ollama', 'anthropic', "
            "or pass a Backend instance."
        )
    return backend


async def extract_classes(
    text: str,
    *,
    backend: BackendLike = "ollama",
    prior: ClassExtraction | None = None,
    candidate_class_names: list[str] | None = None,
) -> ClassExtraction:
    """Run the class-level extraction pass on `text`.

    `prior` is the accumulated class ontology from earlier calls in this
    session — the LLM uses it to avoid re-emitting classes / properties /
    actions / relationships / rules that already exist. Pass `None` (default)
    for a fresh session.

    `candidate_class_names` is an optional hint — pre-computed names
    (semantic search, frequency, hand-picked) that the prompt surfaces under
    a STRONG CANDIDATES preamble so the LLM is biased toward reuse over
    invention.
    """
    b = _resolve_backend(backend)
    return await b.extract_classes(
        text,
        prior or ClassExtraction(),
        candidate_class_names=candidate_class_names,
    )


async def extract_objects(
    text: str,
    schema: ClassExtraction,
    *,
    backend: BackendLike = "ollama",
    prior: ObjectExtraction | None = None,
    candidate_object_names: list[str] | None = None,
    max_classes_in_prompt: int | None = None,
    class_mention_counts: dict[str, int] | None = None,
) -> ObjectExtraction:
    """Run the object-level extraction pass on `text` against `schema`.

    `schema` is the class-level ontology that constrains object output —
    typically the return value of `extract_classes(...)`. Objects of classes
    not in `schema` are silently dropped; same for properties / actions /
    relationship types not declared on the schema.

    `prior` is the accumulated object graph; pass `None` for a fresh session.

    `candidate_object_names`, `max_classes_in_prompt`, `class_mention_counts`
    are optional prompt-shaping hints — see the `Backend` protocol for the
    details.
    """
    b = _resolve_backend(backend)
    return await b.extract_objects(
        text,
        schema,
        prior or ObjectExtraction(),
        candidate_object_names=candidate_object_names,
        max_classes_in_prompt=max_classes_in_prompt,
        class_mention_counts=class_mention_counts,
    )


async def extract(
    text: str,
    *,
    mode: Literal["class", "object", "both"] = "both",
    backend: BackendLike = "ollama",
    prior_classes: ClassExtraction | None = None,
    prior_objects: ObjectExtraction | None = None,
    schema: ClassExtraction | None = None,
) -> Extraction:
    """Convenience: run class extraction, then object extraction against the
    accumulated schema (`prior_classes + new classes`).

    `mode='class'` runs only the class pass; the `objects` field of the
    returned `Extraction` is empty. `mode='object'` runs only the object pass
    — `schema` MUST be provided (or `prior_classes`, which is used as the
    schema). `mode='both'` (default) runs both passes in sequence.

    Returns an `Extraction` with `.classes` (new class-level rows) and
    `.objects` (new object-level rows). Note: per the diff-only prompt, the
    LLM only emits rows that are NEW relative to the priors — so the returned
    `ClassExtraction` / `ObjectExtraction` are *diffs*, not the accumulated
    state. Merge them into your accumulator yourself.
    """
    b = _resolve_backend(backend)
    out = Extraction()

    if mode in ("class", "both"):
        out.classes = await b.extract_classes(text, prior_classes or ClassExtraction())

    if mode in ("object", "both"):
        effective_schema = schema or _merge_class(prior_classes, out.classes)
        out.objects = await b.extract_objects(
            text, effective_schema, prior_objects or ObjectExtraction()
        )

    return out


def _merge_class(
    prior: ClassExtraction | None, new: ClassExtraction
) -> ClassExtraction:
    """Naive merge: concatenate prior + new, de-dup classes by name and
    properties by (class_name, name). Good enough for the diff-only pattern
    where `new` is the just-extracted increment."""
    if not prior:
        return new
    seen_classes: set[str] = set()
    classes = []
    for c in list(prior.classes) + list(new.classes):
        if c.name in seen_classes:
            continue
        seen_classes.add(c.name)
        classes.append(c)
    seen_props: set[tuple[str, str]] = set()
    properties = []
    for p in list(prior.properties) + list(new.properties):
        key = (p.class_name, p.name)
        if key in seen_props:
            continue
        seen_props.add(key)
        properties.append(p)
    return ClassExtraction(
        classes=classes,
        properties=properties,
        actions=list(prior.actions) + list(new.actions),
        relationships=list(prior.relationships) + list(new.relationships),
        rules=list(prior.rules) + list(new.rules),
    )
