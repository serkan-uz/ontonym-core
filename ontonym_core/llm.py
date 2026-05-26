"""LLM backends for ontology extraction.

Two backends share the same interface (`extract_classes`, `extract_objects`,
`check_health`):

- `OllamaBackend` — local Ollama, default model `llama3.1:8b`. No API key, runs
  on the developer's machine; great for trying the library.
- `AnthropicBackend` — hosted Claude. Faster, costs API credits, needs
  `ANTHROPIC_API_KEY` in the environment (or passed explicitly).

The LLM emits NAMES (snake_case English). This module returns Pydantic models
keyed by those names — no surrogate ids, no FK ints, no DB layer.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

from .schema import (
    Action,
    Class,
    ClassExtraction,
    Event,
    ObjectAction,
    ObjectExtraction,
    ObjectInstance,
    ObjectProperty,
    ObjectRelationship,
    Property,
    Relationship,
    Rule,
)

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_CLASS_PROMPT_PATH = _PROMPTS_DIR / "class.txt"
_OBJECT_PROMPT_PATH = _PROMPTS_DIR / "object.txt"


# Injected into the object prompt (at the `{reenrich_directive}` placeholder)
# only when `extract_objects(..., reenrich=True)`. It SUSPENDS the diff-only
# "skip known objects" rule so a re-extraction of an already-ingested document
# re-attaches new per-instance facts (counts, durations, identifiers,
# who-said-what) onto entities that already exist in the graph. This is the
# fix for the diff-only-on-a-dense-graph enrichment ceiling: the first pass
# discovers entities, a later re-enrichment pass fills facts onto them.
_REENRICH_DIRECTIVE = """\
=== RE-ENRICHMENT MODE — THIS OVERRIDES THE DIFF-ONLY OUTPUT RULE BELOW ===
This document is ALREADY in the graph; its entities are mostly KNOWN. Your job
on THIS pass is to ATTACH NEW FACTS to the known entities — facts an earlier,
stricter pass skipped.
- REUSE every canonical name from KNOWN OBJECTS exactly. Do NOT rename or
  duplicate an existing entity. objects[] and events[] may be sparse on this
  pass (most instances already exist) — that is expected and fine.
- The diff-only "do NOT re-emit properties/actions/relationships for known
  objects" restriction is SUSPENDED. Emit EVERY property, action, relationship,
  and event the text supports for known objects: counts, durations, dates,
  amounts, identifier numbers, who-said-what, arguments, disagreements,
  decisions, commitments. properties[], actions[], relationships[], events[]
  SHOULD BE RICH on this pass.
- Downstream still deduplicates exact tuples, so restating a fact already
  present is harmless — err toward emitting rather than skipping.
=== END RE-ENRICHMENT MODE ===
"""


# ----------------------------------------------------------------------------
# Backend protocol
# ----------------------------------------------------------------------------


class Backend(Protocol):
    """Minimal interface every backend implements.

    `candidate_class_names` / `candidate_object_names` are optional hints —
    pre-computed name lists (from semantic search, frequency, or hand-picked)
    that the prompt surfaces under a STRONG CANDIDATES preamble so the LLM
    is biased toward reuse over invention.

    `max_classes_in_prompt` + `class_mention_counts` (object pass only) trim
    the schema rendered into the prompt for very large ontologies — top-K
    classes by mention count get full detail, the tail is name-only.
    """

    async def extract_classes(
        self,
        text: str,
        prior: ClassExtraction,
        *,
        candidate_class_names: list[str] | None = None,
    ) -> ClassExtraction: ...

    async def extract_objects(
        self,
        text: str,
        schema: ClassExtraction,
        prior: ObjectExtraction,
        *,
        candidate_object_names: list[str] | None = None,
        max_classes_in_prompt: int | None = None,
        class_mention_counts: dict[str, int] | None = None,
    ) -> ObjectExtraction: ...

    async def check_health(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------------
# Prompt rendering — works off name-based Pydantic models
# ----------------------------------------------------------------------------


def render_known_classes(
    prior: ClassExtraction,
    *,
    candidates: list[str] | None = None,
) -> str:
    """Render the known-classes hint for the class-pass prompt.

    When `candidates` is non-empty, the names land under a STRONG CANDIDATES
    preamble so the LLM is biased toward reusing one of them rather than
    inventing a near-synonym. The full existing class list is still rendered
    as a secondary "All registered classes" line — the candidate list is a
    *hint*, not a constraint. The caller is responsible for ranking; this
    helper trusts whatever order it receives.
    """
    full = (
        ", ".join(
            f"{c.name} (inherits {c.inherited_from})" if c.inherited_from else c.name
            for c in prior.classes
        )
        if prior.classes
        else "(none yet — no constraint, pick whatever class fits)"
    )
    if not candidates:
        return full
    seen: set[str] = set()
    deduped = [c for c in candidates if not (c in seen or seen.add(c))]
    return (
        "STRONG CANDIDATES (semantically similar to this text — "
        "reuse one of these unless none fits): "
        + ", ".join(deduped)
        + "\nAll registered classes: "
        + full
    )


def render_previous_context(prior: ClassExtraction, max_items: int = 30) -> str:
    if not prior.classes and not prior.relationships and not prior.rules:
        return "(no previous context — first text in this session)"
    lines: list[str] = []
    if prior.classes:
        names = []
        for c in prior.classes[:max_items]:
            if c.inherited_from:
                names.append(f"{c.name} (inherits {c.inherited_from})")
            else:
                names.append(c.name)
        extra = (
            f" ... and {len(prior.classes) - max_items} more"
            if len(prior.classes) > max_items
            else ""
        )
        lines.append(f"Previous classes: {', '.join(names)}{extra}")
    if prior.relationships:
        rels = [
            f"{r.source} --{r.type}--> {r.target}"
            for r in prior.relationships[:max_items]
        ]
        lines.append("Previous relationships: " + "; ".join(rels))
    if prior.rules:
        rule_lines = [
            f"{r.name} [{', '.join(r.classes) or '?'}]" for r in prior.rules[:max_items]
        ]
        extra = (
            f" ... and {len(prior.rules) - max_items} more"
            if len(prior.rules) > max_items
            else ""
        )
        lines.append("Previous rules: " + "; ".join(rule_lines) + extra)
    return "\n".join(lines)


def render_class_schema(
    schema: ClassExtraction,
    *,
    max_classes: int | None = None,
    class_mention_counts: dict[str, int] | None = None,
) -> str:
    """Render the class-level ontology as a COMPACT allow-set for the object pass.

    With `max_classes` set, only the top-K classes get detailed property
    listings; remaining classes are listed by name only (still legal in
    output, just no per-property breakdown). Ranking is by
    `class_mention_counts` when supplied (highest first) — typically the
    caller's count of how often each class has been mentioned in prior
    ingestions. Without counts, ranking is the insertion order in `schema`.

    Action and relationship type sets are always rendered in full.
    """
    if not schema.classes:
        return "(no classes defined yet — instance extraction will return nothing)"
    props_by_class: dict[str, list[str]] = {}
    for p in schema.properties:
        props_by_class.setdefault(p.class_name, []).append(
            f"{p.name}:{p.data_type or 'str'}"
        )

    all_classes = list(schema.classes)
    if max_classes is not None and len(all_classes) > max_classes:
        counts = class_mention_counts or {}
        ranked = sorted(all_classes, key=lambda c: counts.get(c.name, 0), reverse=True)
        top_classes = ranked[:max_classes]
        tail_classes = ranked[max_classes:]
    else:
        top_classes = all_classes
        tail_classes = []

    lines: list[str] = ["# CLASSES — use as `class_name`"]
    if tail_classes:
        lines.append(
            f"# (showing the top {len(top_classes)} of {len(all_classes)} by "
            f"mention count; tail of {len(tail_classes)} listed name-only)"
        )
    for cls in top_classes:
        own = props_by_class.get(cls.name, [])
        desc = f" — {cls.description}" if cls.description else ""
        inherits = f" (inherits {cls.inherited_from})" if cls.inherited_from else ""
        if own:
            lines.append(f"{cls.name}({', '.join(own)}){desc}{inherits}")
        else:
            lines.append(f"{cls.name}{desc}{inherits}")
    if tail_classes:
        lines.append("")
        lines.append("# Less-mentioned classes (name only, also valid):")
        chunk: list[str] = []
        chunk_len = 0
        for cls in tail_classes:
            if chunk_len + len(cls.name) + 2 > 120 and chunk:
                lines.append(", ".join(chunk))
                chunk = []
                chunk_len = 0
            chunk.append(cls.name)
            chunk_len += len(cls.name) + 2
        if chunk:
            lines.append(", ".join(chunk))

    action_names = sorted({a.name for a in schema.actions})
    rel_types = sorted({r.type for r in schema.relationships})
    lines.append("")
    lines.append("# ACTION NAMES — use as `action_name` in actions[]")
    lines.append(", ".join(action_names) if action_names else "(none)")
    lines.append("")
    lines.append("# RELATIONSHIP TYPES — use as `type` in relationships[]")
    lines.append(", ".join(rel_types) if rel_types else "(none)")
    return "\n".join(lines).rstrip()


def render_known_objects(
    prior: ObjectExtraction,
    *,
    max_items: int = 60,
    candidates: list[str] | None = None,
) -> str:
    """Render the known-objects hint for the object-pass prompt.

    When `candidates` is non-empty, they're surfaced under a STRONG CANDIDATES
    preamble — same shape as the class-pass candidate hint, but at the
    instance level. Caller pre-computes these (semantic search, frequency,
    hand-picked) and supplies the canonical names.
    """
    all_items = list(prior.objects) + list(prior.events)
    if not all_items:
        base = "(none yet)"
    else:
        rendered = [f"{o.class_name}:{o.name}" for o in all_items[:max_items]]
        more = (
            f" ... and {len(all_items) - max_items} more"
            if len(all_items) > max_items
            else ""
        )
        base = "Objects: " + ", ".join(rendered) + more
    if not candidates:
        return base
    seen: set[str] = set()
    deduped = [c for c in candidates if not (c in seen or seen.add(c))]
    return (
        "STRONG CANDIDATE OBJECTS (semantically similar — reuse if any fits): "
        + ", ".join(deduped)
        + "\n"
        + base
    )


# ----------------------------------------------------------------------------
# Inheritance-aware allowed sets for the object-pass parser
# ----------------------------------------------------------------------------


def _allowed_sets(schema: ClassExtraction) -> tuple[
    set[str], dict[str, set[str]], set[str], set[str], set[str]
]:
    """Return (allowed_class_names, allowed_props_by_class, allowed_actions,
    allowed_rel_types, event_class_names).

    An event class is one whose ancestor chain includes a class literally named
    `event`. If there's no `event` class in the schema, the event set is empty
    and anything the LLM emits in `events[]` is dropped with a warning.
    """
    allowed_classes = {c.name for c in schema.classes}
    parent_by_name = {c.name: c.inherited_from for c in schema.classes}

    own_props_by_class: dict[str, set[str]] = {}
    for p in schema.properties:
        own_props_by_class.setdefault(p.class_name, set()).add(p.name)

    allowed_props_by_class: dict[str, set[str]] = {}
    for cls_name in allowed_classes:
        inherited = set(own_props_by_class.get(cls_name, set()))
        visited: set[str] = {cls_name}
        ancestor = parent_by_name.get(cls_name)
        while ancestor and ancestor not in visited:
            visited.add(ancestor)
            inherited |= own_props_by_class.get(ancestor, set())
            ancestor = parent_by_name.get(ancestor)
        allowed_props_by_class[cls_name] = inherited

    allowed_actions = {a.name for a in schema.actions}
    allowed_rel_types = {r.type for r in schema.relationships}

    event_class_names: set[str] = set()
    if "event" in allowed_classes:
        for cls_name in allowed_classes:
            cur: str | None = cls_name
            visited2: set[str] = set()
            while cur and cur not in visited2:
                if cur == "event":
                    event_class_names.add(cls_name)
                    break
                visited2.add(cur)
                cur = parent_by_name.get(cur)

    return allowed_classes, allowed_props_by_class, allowed_actions, allowed_rel_types, event_class_names


# ----------------------------------------------------------------------------
# Parsers — share between backends
# ----------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


def parse_class_json(raw: str) -> ClassExtraction:
    """Parse the raw class-pass LLM JSON into a ClassExtraction."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Class LLM output is not valid JSON: %s", cleaned[:500])
        raise ValueError(f"LLM did not produce valid JSON: {exc}") from exc

    raw_classes = data.get("classes") or data.get("entities") or []
    classes: list[Class] = []
    properties: list[Property] = []
    prop_keys: set[tuple[str, str]] = set()
    for c in raw_classes:
        if not isinstance(c, dict):
            continue
        cname = c.get("name")
        if not cname:
            continue
        parent = c.get("inherited_from") or c.get("parent_class")
        classes.append(
            Class(
                name=cname,
                description=c.get("description"),
                inherited_from=parent if isinstance(parent, str) and parent else None,
            )
        )
        for p in c.get("properties") or []:
            if not isinstance(p, dict):
                continue
            pname = p.get("name")
            if not pname:
                continue
            key = (cname, pname)
            if key in prop_keys:
                continue
            prop_keys.add(key)
            properties.append(
                Property(class_name=cname, name=pname, data_type=p.get("data_type"))
            )

    actions: list[Action] = []
    for a in data.get("actions") or []:
        if not isinstance(a, dict):
            continue
        aname = a.get("name")
        if not aname:
            continue
        actions.append(
            Action(
                name=aname,
                actor=a.get("actor"),
                target=a.get("target"),
                description=a.get("description"),
            )
        )

    relationships: list[Relationship] = []
    for r in data.get("relationships") or []:
        if not isinstance(r, dict):
            continue
        src, tgt, rtype = r.get("source"), r.get("target"), r.get("type")
        if not src or not tgt or not rtype:
            continue
        relationships.append(
            Relationship(
                source=src, target=tgt, type=rtype, description=r.get("description")
            )
        )

    rules: list[Rule] = []
    for r in data.get("rules") or []:
        if not isinstance(r, dict):
            continue
        rname = r.get("name")
        if not rname:
            continue
        cls_list = r.get("classes")
        if isinstance(cls_list, str):
            cls_list = [cls_list]
        if not isinstance(cls_list, list) or not cls_list:
            continue
        cls_names = [c for c in cls_list if isinstance(c, str) and c]
        if not cls_names:
            continue
        rules.append(Rule(name=rname, description=r.get("description"), classes=cls_names))

    return ClassExtraction(
        classes=classes,
        properties=properties,
        actions=actions,
        relationships=relationships,
        rules=rules,
    )


def parse_object_json(raw: str, schema: ClassExtraction) -> ObjectExtraction:
    """Parse the raw object-pass LLM JSON into an ObjectExtraction, dropping
    rows that reference classes / properties / actions / relationship types
    not present in `schema`."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Object LLM output is not valid JSON: %s", cleaned[:500])
        raise ValueError(f"LLM did not produce valid JSON: {exc}") from exc

    (
        allowed_classes,
        allowed_props_by_class,
        allowed_actions,
        allowed_rel_types,
        event_class_names,
    ) = _allowed_sets(schema)

    objects: list[ObjectInstance] = []
    for o in data.get("objects") or []:
        if not isinstance(o, dict):
            continue
        cls = o.get("class_name")
        name = o.get("name")
        if not cls or not name or cls not in allowed_classes:
            continue
        if cls in event_class_names:
            # LLM put an event-class instance in objects[]; events[] is the right home.
            logger.debug("Skipping %s/%s in objects[] — class inherits event", cls, name)
            continue
        objects.append(
            ObjectInstance(
                class_name=cls,
                name=name,
                display_name=o.get("display_name"),
                description=o.get("description"),
            )
        )

    events: list[Event] = []
    for e in data.get("events") or []:
        if not isinstance(e, dict):
            continue
        cls = e.get("class_name")
        name = e.get("name")
        if not cls or not name or cls not in allowed_classes:
            continue
        if event_class_names and cls not in event_class_names:
            logger.debug(
                "Dropping event %s/%s — class does not inherit `event`", cls, name
            )
            continue
        events.append(
            Event(
                class_name=cls,
                name=name,
                display_name=e.get("display_name"),
                description=e.get("description"),
            )
        )

    properties: list[ObjectProperty] = []
    for p in data.get("properties") or []:
        if not isinstance(p, dict):
            continue
        cls, obj_name, pname = p.get("class_name"), p.get("object_name"), p.get("name")
        if not cls or not obj_name or not pname:
            continue
        if cls not in allowed_classes:
            continue
        if pname not in allowed_props_by_class.get(cls, set()):
            continue
        raw_value = p.get("value")
        # The DB column is TEXT (polymorphic per CLAUDE.md §3). LLMs often
        # emit numeric / bool literals for number/boolean-typed properties;
        # stringify so pydantic's str-only ObjectProperty.value accepts them.
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            raw_value = json.dumps(raw_value) if isinstance(raw_value, (list, dict)) else str(raw_value)
        properties.append(
            ObjectProperty(
                class_name=cls,
                object_name=obj_name,
                name=pname,
                value=raw_value,
                data_type=p.get("data_type"),
            )
        )

    actions: list[ObjectAction] = []
    for a in data.get("actions") or []:
        if not isinstance(a, dict):
            continue
        an = a.get("action_name") or a.get("name")
        if not an or an not in allowed_actions:
            continue
        actions.append(
            ObjectAction(
                action_name=an,
                actor=a.get("actor"),
                target=a.get("target"),
                description=a.get("description"),
                occurred_at=a.get("occurred_at"),
            )
        )

    relationships: list[ObjectRelationship] = []
    for r in data.get("relationships") or []:
        if not isinstance(r, dict):
            continue
        rt, src, tgt = r.get("type"), r.get("source"), r.get("target")
        if not rt or not src or not tgt or rt not in allowed_rel_types:
            continue
        relationships.append(
            ObjectRelationship(
                source=src, target=tgt, type=rt, description=r.get("description")
            )
        )

    return ObjectExtraction(
        objects=objects,
        events=events,
        properties=properties,
        actions=actions,
        relationships=relationships,
    )


# ----------------------------------------------------------------------------
# Ollama backend
# ----------------------------------------------------------------------------


class OllamaBackend:
    """Local Ollama backend. No API key needed; works on the developer's
    machine with `ollama serve` running and a model pulled (`ollama pull
    llama3.1:8b`).

    Defaults are tuned for extraction: low temperature, deterministic seed,
    `format=json` for strict JSON output.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        *,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int = 2048,
        seed: int = 42,
        timeout: float = 300.0,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._class_prompt = _CLASS_PROMPT_PATH.read_text(encoding="utf-8")
        self._object_prompt = _OBJECT_PROMPT_PATH.read_text(encoding="utf-8")

    async def _generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url}. Is 'ollama serve' running?"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama returned error: {exc.response.status_code} {exc.response.text}"
                ) from exc
        data = resp.json()
        if "eval_count" in data and "eval_duration" in data:
            tokens = data["eval_count"]
            duration_s = data["eval_duration"] / 1e9
            tps = tokens / duration_s if duration_s > 0 else 0
            logger.info("Ollama: %d tokens, %.1fs, %.1f tok/s", tokens, duration_s, tps)
        return (data.get("response") or "").strip()

    async def extract_classes(
        self,
        text: str,
        prior: ClassExtraction,
        *,
        candidate_class_names: list[str] | None = None,
    ) -> ClassExtraction:
        prompt = (
            self._class_prompt
            .replace("{previous_context}", render_previous_context(prior))
            .replace(
                "{known_classes}",
                render_known_classes(prior, candidates=candidate_class_names),
            )
            .replace("{text}", text)
        )
        raw = await self._generate(prompt)
        return parse_class_json(raw)

    async def extract_objects(
        self,
        text: str,
        schema: ClassExtraction,
        prior: ObjectExtraction,
        *,
        candidate_object_names: list[str] | None = None,
        max_classes_in_prompt: int | None = None,
        class_mention_counts: dict[str, int] | None = None,
        reenrich: bool = False,
    ) -> ObjectExtraction:
        if not schema.classes:
            return ObjectExtraction()
        prompt = (
            self._object_prompt
            .replace(
                "{class_schema}",
                render_class_schema(
                    schema,
                    max_classes=max_classes_in_prompt,
                    class_mention_counts=class_mention_counts,
                ),
            )
            .replace(
                "{known_objects}",
                render_known_objects(prior, candidates=candidate_object_names),
            )
            .replace("{reenrich_directive}", _REENRICH_DIRECTIVE if reenrich else "")
            .replace("{text}", text)
        )
        raw = await self._generate(prompt)
        return parse_object_json(raw, schema)

    async def check_health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                tags = await client.get(f"{self.base_url}/api/tags")
                tags.raise_for_status()
                models = [m["name"] for m in tags.json().get("models", [])]
                return {
                    "ollama_reachable": True,
                    "model_loaded": self.model in models,
                    "available_models": models,
                }
            except Exception as exc:
                return {"ollama_reachable": False, "error": str(exc)}


# ----------------------------------------------------------------------------
# Anthropic backend
# ----------------------------------------------------------------------------


_ANTHROPIC_SYSTEM_PROMPT = (
    "You extract structured ontologies from free text. "
    "Respond with strictly valid JSON ONLY — no markdown fences, no prose, "
    "no commentary. Output exactly the JSON object whose schema is described "
    "in the user message. Use snake_case English names. If a section has no "
    "entries, return an empty array for it. Do not invent classes that aren't "
    "warranted by the text."
)


class AnthropicBackend:
    """Hosted Claude backend. Requires the `anthropic` package — install with
    `pip install ontonym-core[anthropic]`. Reads `ANTHROPIC_API_KEY` from the
    environment if not passed explicitly.

    Faster than local Ollama (typical extraction call 2-5s vs 10-30s on a
    consumer GPU), but costs API credits. Use it for production, demos, or
    when you don't want to run a local model.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 65536,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicBackend requires the `anthropic` package. "
                "Install with: pip install 'ontonym-core[anthropic]'"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        self._class_prompt = _CLASS_PROMPT_PATH.read_text(encoding="utf-8")
        self._object_prompt = _OBJECT_PROMPT_PATH.read_text(encoding="utf-8")

    async def _invoke(self, user_prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package not available") from exc
        try:
            # Stream, not blocking create(): large verb-rich outputs (~8k
            # tokens) take 1-3 min to generate, and a non-streaming request
            # leaves the HTTP connection idle that whole time — intermediate
            # infrastructure drops it, surfacing as APIConnectionError
            # ("Request timed out or interrupted"). Streaming keeps bytes
            # flowing so the connection survives long generations.
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=[
                    {
                        "type": "text",
                        "text": _ANTHROPIC_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                # Drain deltas to keep the connection active; the SDK
                # assembles the final Message (content + usage) for us.
                async for _ in stream.text_stream:
                    pass
                response = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Cannot reach Anthropic API: {exc}") from exc
        raw_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
        usage = response.usage
        logger.info(
            "Anthropic %s: in=%d out=%d cache_read=%s cache_write=%s",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        return raw_text

    async def extract_classes(
        self,
        text: str,
        prior: ClassExtraction,
        *,
        candidate_class_names: list[str] | None = None,
    ) -> ClassExtraction:
        prompt = (
            self._class_prompt
            .replace("{previous_context}", render_previous_context(prior))
            .replace(
                "{known_classes}",
                render_known_classes(prior, candidates=candidate_class_names),
            )
            .replace("{text}", text)
        )
        raw = await self._invoke(prompt)
        return parse_class_json(raw)

    async def extract_objects(
        self,
        text: str,
        schema: ClassExtraction,
        prior: ObjectExtraction,
        *,
        candidate_object_names: list[str] | None = None,
        max_classes_in_prompt: int | None = None,
        class_mention_counts: dict[str, int] | None = None,
        reenrich: bool = False,
    ) -> ObjectExtraction:
        if not schema.classes:
            return ObjectExtraction()
        prompt = (
            self._object_prompt
            .replace(
                "{class_schema}",
                render_class_schema(
                    schema,
                    max_classes=max_classes_in_prompt,
                    class_mention_counts=class_mention_counts,
                ),
            )
            .replace(
                "{known_objects}",
                render_known_objects(prior, candidates=candidate_object_names),
            )
            .replace("{reenrich_directive}", _REENRICH_DIRECTIVE if reenrich else "")
            .replace("{text}", text)
        )
        raw = await self._invoke(prompt)
        return parse_object_json(raw, schema)

    async def check_health(self) -> dict[str, Any]:
        return {
            "anthropic_reachable": True,
            "model": self.model,
            "api_key_configured": bool(
                self._client.api_key or os.getenv("ANTHROPIC_API_KEY")
            ),
        }
