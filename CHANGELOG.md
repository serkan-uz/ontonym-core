# Changelog

All notable changes to ontonym-core will be documented here.

## 0.1.0 — 2026-05-18

Initial public release.

- `extract(text, *, mode="class"|"object"|"both", backend, prior_classes, prior_objects)` runs class extraction and/or object extraction in one call.
- `extract_classes` / `extract_objects` for finer control.
- `OllamaBackend` (default, no API key) and `AnthropicBackend` (Claude, `[anthropic]` extra).
- CLI: `ontonym-core extract --text "..."` and `ontonym-core health`.
- Pydantic models for everything — name-based references, no FK ids, no approval status.
- FastAPI example server at `examples/server.py` (`[server]` extra).
- Prompts ship with the wheel — `class.txt` (class pass) and `object.txt` (object pass).

0.x is unstable — APIs may change between minor versions until 1.0.
