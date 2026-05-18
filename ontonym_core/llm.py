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


# ----------------------------------------------------------------------------
# Backend protocol
# ----------------------------------------------------------------------------


class Backend(Protocol):
    """Minimal interface every backend implements."""

    async def extract_classes(
        self, text: str, prior: ClassExtraction
    ) -> ClassExtraction: ...

    async def extract_objects(
        self, text: str, schema: ClassExtraction, prior: ObjectExtraction
    ) -> ObjectExtraction: ...

    async def check_health(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------------
# Prompt rendering — works off name-based Pydantic models
# ----------------------------------------------------------------------------


def _render_known_classes(prior: ClassExtraction) -> str:
    if not prior.classes:
        return "(none yet — no constraint, pick whatever class fits)"
    names: list[str] = []
    for c in prior.classes:
        if c.inherited_from:
            names.append(f"{c.name} (inherits {c.inherited_from})")
        else:
            names.append(c.name)
    return ", ".join(names)


def _render_previous_context(prior: ClassExtraction, max_items: int = 30) -> str:
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


def _render_class_schema(schema: ClassExtraction) -> str:
    if not schema.classes:
        return "(no classes defined yet — instance extraction will return nothing)"
    props_by_class: dict[str, list[str]] = {}
    for p in schema.properties:
        props_by_class.setdefault(p.class_name, []).append(
            f"{p.name}:{p.data_type or 'str'}"
        )

    lines: list[str] = ["# CLASSES — use as `class_name`"]
    for cls in schema.classes:
        own = props_by_class.get(cls.name, [])
        desc = f" — {cls.description}" if cls.description else ""
        inherits = f" (inherits {cls.inherited_from})" if cls.inherited_from else ""
        if own:
            lines.append(f"{cls.name}({', '.join(own)}){desc}{inherits}")
        else:
            lines.append(f"{cls.name}{desc}{inherits}")

    action_names = sorted({a.name for a in schema.actions})
    rel_types = sorted({r.type for r in schema.relationships})
    lines.append("")
    lines.append("# ACTION NAMES — use as `action_name` in actions[]")
    lines.append(", ".join(action_names) if action_names else "(none)")
    lines.append("")
    lines.append("# RELATIONSHIP TYPES — use as `type` in relationships[]")
    lines.append(", ".join(rel_types) if rel_types else "(none)")
    return "\n".join(lines).rstrip()


def _render_known_objects(prior: ObjectExtraction, max_items: int = 60) -> str:
    all_items = list(prior.objects) + list(prior.events)
    if not all_items:
        return "(none yet)"
    rendered = [f"{o.class_name}:{o.name}" for o in all_items[:max_items]]
    more = (
        f" ... and {len(all_items) - max_items} more"
        if len(all_items) > max_items
        else ""
    )
    return "Objects: " + ", ".join(rendered) + more


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
        properties.append(
            ObjectProperty(
                class_name=cls,
                object_name=obj_name,
                name=pname,
                value=p.get("value"),
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
        self, text: str, prior: ClassExtraction
    ) -> ClassExtraction:
        prompt = (
            self._class_prompt
            .replace("{previous_context}", _render_previous_context(prior))
            .replace("{known_classes}", _render_known_classes(prior))
            .replace("{text}", text)
        )
        raw = await self._generate(prompt)
        return parse_class_json(raw)

    async def extract_objects(
        self, text: str, schema: ClassExtraction, prior: ObjectExtraction
    ) -> ObjectExtraction:
        if not schema.classes:
            return ObjectExtraction()
        prompt = (
            self._object_prompt
            .replace("{class_schema}", _render_class_schema(schema))
            .replace("{known_objects}", _render_known_objects(prior))
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
            response = await self._client.messages.create(
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
            )
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
        self, text: str, prior: ClassExtraction
    ) -> ClassExtraction:
        prompt = (
            self._class_prompt
            .replace("{previous_context}", _render_previous_context(prior))
            .replace("{known_classes}", _render_known_classes(prior))
            .replace("{text}", text)
        )
        raw = await self._invoke(prompt)
        return parse_class_json(raw)

    async def extract_objects(
        self, text: str, schema: ClassExtraction, prior: ObjectExtraction
    ) -> ObjectExtraction:
        if not schema.classes:
            return ObjectExtraction()
        prompt = (
            self._object_prompt
            .replace("{class_schema}", _render_class_schema(schema))
            .replace("{known_objects}", _render_known_objects(prior))
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
