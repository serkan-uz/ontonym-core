# Changelog

All notable changes to ontonym-core will be documented here.

## 0.3.0 — 2026-05-26

Re-enrichment + coreference. Backwards-compatible — the new `extract_objects`
kwarg defaults to `False` (current behaviour) and coref is opt-in via env.

- `extract_objects(..., reenrich=False)` on both backends. When `True`, the
  object prompt's new `{reenrich_directive}` placeholder is filled with a block
  that SUSPENDS the diff-only "skip known objects" rule, so re-running an
  already-extracted document attaches new facts (properties, actions,
  relationships, events) onto entities that already exist — the fix for the
  diff-only-on-a-dense-graph enrichment ceiling.
- New `ontonym_core.coref` module + `get_coref_resolver()` export: a pluggable
  coreference pre-pass selected by `COREF_BACKEND` (`none` | `llm` | `spacy`)
  that resolves pronouns/references to named entities before extraction. The
  `llm` backend rewrites via `COREF_LLM_BACKEND` / `COREF_MODEL`; `spacy` uses
  the optional `fastcoref` dependency.
- `prompts/object.txt` carries the `{reenrich_directive}` placeholder (filled
  only when `reenrich=True`; empty otherwise).

## 0.2.0 — 2026-05-18

Optional prompt-shaping hints — backwards-compatible additions to the
public API. No changes to existing call sites; all new kwargs default to
`None` and preserve 0.1.0 behaviour when omitted.

- `extract_classes(..., candidate_class_names=None)` — when supplied, the
  prompt surfaces a STRONG CANDIDATES preamble so the LLM is biased toward
  reusing one of the named classes rather than inventing a synonym. Useful
  when the caller has its own way of computing semantic-similar candidates
  (vector search, frequency, hand-picked).
- `extract_objects(..., candidate_object_names=None, max_classes_in_prompt=None, class_mention_counts=None)` —
  same candidate-hint mechanism for objects, plus schema-trimming knobs:
  with `max_classes_in_prompt=K`, the rendered schema shows full per-class
  property listings only for the top-K classes by `class_mention_counts`
  (rest are name-only but still legal in the output).
- `Backend` protocol updated with the new kwargs.

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
