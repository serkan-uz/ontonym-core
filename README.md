# ontonym-core

Turn free text into a typed object graph — **classes, properties, actions, relationships, rules**, plus **objects and events** — using any LLM.

```python
import asyncio
from ontonym_core import extract

result = asyncio.run(extract(
    "Sarah deployed PaymentService at 14:02. "
    "An outage hit it at 14:05, affecting the payments flow.",
    backend="ollama",
))
print(result.model_dump_json(indent=2))
```

```json
{
  "classes": {
    "classes": [
      {"name": "person", "description": "A human", "inherited_from": null},
      {"name": "application", "description": "A deployable service", "inherited_from": null},
      {"name": "outage", "description": "A service outage", "inherited_from": "event"}
    ],
    "actions": [{"name": "deploy", "actor": "person", "target": "application"}],
    "relationships": [{"source": "outage", "target": "application", "type": "affected"}],
    "rules": []
  },
  "objects": {
    "objects": [
      {"class_name": "person", "name": "sarah", "display_name": "Sarah"},
      {"class_name": "application", "name": "payment_service", "display_name": "PaymentService"}
    ],
    "events": [
      {"class_name": "outage", "name": "outage_1405", "display_name": "outage at 14:05"}
    ],
    "actions": [
      {"action_name": "deploy", "actor": "sarah", "target": "payment_service", "occurred_at": "14:02"}
    ],
    "relationships": [
      {"source": "outage_1405", "target": "payment_service", "type": "affected"}
    ]
  }
}
```

Local-first — run it with [Ollama](https://ollama.com) and no API key. Or switch to Anthropic Claude when you want speed.

> **Status**: 0.x is unstable. APIs may change between minor versions until 1.0.

---

## What it does

Two extraction passes, run in sequence:

1. **Class layer** (`classes`, `properties`, `actions`, `relationships`, `rules`) — the schema. *What kinds of things exist in this text?*
2. **Object layer** (`objects`, `events`, `object_properties`, `object_actions`, `object_relationships`) — the instances. *What specific things are mentioned, and what did they do?*

Events are first-class: actor-less, time-anchored happenings (incidents, outages, deployments, decisions, market events) are emitted separately from actor-driven actions.

## Install

```bash
pip install ontonym-core

# Optional extras:
pip install 'ontonym-core[anthropic]'      # adds Anthropic Claude backend
pip install 'ontonym-core[server]'         # adds FastAPI example server
```

## Quickstart — Ollama (local, no API key)

```bash
# One-time setup
ollama serve                                # in another terminal
ollama pull llama3.1:8b
pip install ontonym-core

# CLI
ontonym-core extract --text "Sarah deployed prod at 14:02"

# Or stream from stdin
cat notes.txt | ontonym-core extract --mode both

# Health check
ontonym-core health --backend ollama
```

## Quickstart — Anthropic (hosted Claude)

```bash
pip install 'ontonym-core[anthropic]'
export ANTHROPIC_API_KEY=sk-ant-...

ontonym-core extract --text "..." --backend anthropic
```

## Python API

```python
import asyncio
from ontonym_core import extract, extract_classes, extract_objects, OllamaBackend

# One-shot: class pass + object pass against the resulting schema.
result = asyncio.run(extract(
    "Marcus closed Acme's renewal on 2024-11-18.",
    backend="ollama",
))

# Or run passes separately for full control.
backend = OllamaBackend(model="llama3.1:8b")
schema = asyncio.run(extract_classes("...", backend=backend))
objects = asyncio.run(extract_objects("...", schema, backend=backend))

# Diff-only iteration: feed the prior accumulator to skip already-known rows.
schema_v2 = asyncio.run(extract_classes("more text...", backend=backend, prior=schema))
```

The result models are plain Pydantic — `.model_dump_json()` for JSON, `.model_dump()` for dicts, and full type hints for IDE autocomplete.

## Custom backends

A backend is anything implementing `Backend` from `ontonym_core.llm` — two async methods (`extract_classes`, `extract_objects`) and one health probe (`check_health`). You can wire OpenAI, Together, Groq, or your own gateway by reusing the parsers:

```python
from ontonym_core import parse_class_json, parse_object_json, ClassExtraction

class MyBackend:
    async def extract_classes(self, text, prior):
        raw = await my_llm.generate(prompt_for_classes(text, prior))
        return parse_class_json(raw)
    # ... extract_objects, check_health similarly
```

## FastAPI example

```bash
pip install 'ontonym-core[server]'
uvicorn examples.server:app --reload

curl -X POST http://localhost:8000/extract \
  -H 'content-type: application/json' \
  -d '{"text": "...", "mode": "both"}'
```

See [examples/server.py](examples/server.py) — single route, ~50 lines.

## What's not in here

Deliberately out of scope:

- **Storage** — `ontonym-core` is stateless. Persist the JSON to whatever fits your stack (Postgres, DuckDB, plain files).
- **Multi-tenancy / access control** — same.
- **Flow detection** (recurring patterns across action+event timelines) — held back.
- **Approval workflows** — what makes a graph trustworthy is a product problem, not an extractor problem.

If you want those out of the box, see [ontonym.com](https://ontonym.com) — the hosted product that this library was carved out of.

## How this fits with ontonym (hosted)

The hosted product at ontonym.com depends on `ontonym-core`. Same prompts, same parsers, same Pydantic models. What hosted adds on top:

- Postgres + pgvector storage with provenance (every fact tied to its source document)
- Per-tenant isolation + per-row access grants
- Approval workflow (the "Object Guru" reviews / merges / rejects)
- Flow detection across time-ordered actions
- An MCP server so AI agents read the graph as a tool
- Slack / Teams / Notion / Salesforce / HubSpot / GitHub / Linear / Jira connectors

If you build something cool on top of `ontonym-core`, [drop a note](https://github.com/serkan-uz/ontonym-core/issues) — we'd love to see it.

## License

Apache 2.0. See [LICENSE](LICENSE).
