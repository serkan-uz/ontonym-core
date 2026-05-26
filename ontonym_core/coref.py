"""Pluggable coreference resolution pre-pass.

Before extraction, resolve pronouns and ambiguous referring expressions to the
named entity they point at, so "she decided X" becomes "Marie decided X" and
the object pass can attach the action/decision to the right entity instead of
dropping it. This recovers the connective tissue ("what did *she* think about
Y?") that diff-only extraction otherwise loses.

Backend is selected by the COREF_BACKEND environment variable, mirroring the
EXTRACTOR_BACKEND / CHAT_BACKEND pattern:

  - unset / "none" / "off"  -> no coref; returns text unchanged (zero cost).
  - "llm"                   -> LLM rewrite. The model rewrites the passage,
                               resolving references in-context. The rewrite
                               backend is COREF_LLM_BACKEND, defaulting to
                               EXTRACTOR_BACKEND (ollama | anthropic).
  - "spacy" / "fastcoref"   -> deterministic neural coreference via the
                               `fastcoref` package (optional dependency). No
                               per-call LLM cost once the model is loaded.

Every resolver exposes `async def resolve(self, text: str) -> str` and is built
via `get_coref_resolver()`, which returns None when coref is off so callers can
cheaply skip the pass.

Cost note: the LLM backend roughly doubles per-document token cost (it reads
the whole passage and writes it back). It is OFF by default for that reason.
The fastcoref backend has no per-call LLM cost once the model is loaded, but
adds a model dependency and CPU/GPU time.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


_COREF_PROMPT = """\
You are a coreference resolver. Rewrite the passage below so that every pronoun
and ambiguous referring expression (he, she, they, it, this, that, him, her,
them, the team, the company, the project, etc.) is replaced by the specific
named entity it refers to, using the nearest unambiguous antecedent.

STRICT RULES:
- Preserve every fact verbatim. Do NOT summarize, add, remove, or reorder content.
- Keep speaker labels, line breaks, numbers, dates, and identifiers exactly as-is.
- Only substitute a reference when you are confident of its antecedent. If a
  reference is genuinely ambiguous, leave it unchanged.
- Do NOT add commentary, headers, or explanation.
- Output ONLY the rewritten passage.

PASSAGE:
{text}"""


class CorefResolver(Protocol):
    """Anything that turns a passage into a coreference-resolved passage."""

    name: str

    async def resolve(self, text: str) -> str: ...


class _LLMCoref:
    """Resolve coreferences by asking an LLM to rewrite the passage."""

    name = "llm"

    def __init__(self) -> None:
        self._backend = (
            os.getenv("COREF_LLM_BACKEND")
            or os.getenv("EXTRACTOR_BACKEND")
            or "ollama"
        ).strip().lower()

    async def resolve(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = _COREF_PROMPT.replace("{text}", text)
        if self._backend == "anthropic":
            return await self._resolve_anthropic(prompt, fallback=text)
        return await self._resolve_ollama(prompt, fallback=text)

    async def _resolve_ollama(self, prompt: str, *, fallback: str) -> str:
        import httpx

        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("COREF_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "32768")),
            },
        }
        timeout = float(os.getenv("COREF_TIMEOUT", "300"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            resp.raise_for_status()
        out = (resp.json().get("response") or "").strip()
        return out or fallback

    async def _resolve_anthropic(self, prompt: str, *, fallback: str) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = os.getenv("COREF_MODEL") or os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )
        # Stream so long rewrites don't drop the connection (same reason the
        # extractor streams — see AnthropicBackend._invoke).
        async with client.messages.stream(
            model=model,
            max_tokens=int(os.getenv("COREF_MAX_TOKENS", "16384")),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _ in stream.text_stream:
                pass
            msg = await stream.get_final_message()
        out = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
        return out or fallback


class _SpacyCoref:
    """Resolve coreferences with the `fastcoref` neural model. Deterministic,
    no per-call LLM cost. The model loads once and is cached on the instance.

    Strategy: fastcoref returns clusters of character spans that refer to the
    same entity. For each cluster we pick a representative mention (the longest
    span — usually the full proper name) and rewrite every other mention in the
    cluster to it. Replacements are applied right-to-left so earlier spans stay
    valid.
    """

    name = "spacy"

    def __init__(self) -> None:
        try:
            from fastcoref import FCoref
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "COREF_BACKEND=spacy requires the `fastcoref` package. "
                "Install with: pip install fastcoref"
            ) from exc
        self._model = FCoref(device=os.getenv("COREF_DEVICE", "cpu"))

    async def resolve(self, text: str) -> str:
        if not text.strip():
            return text
        # fastcoref is synchronous + CPU/GPU bound; keep the event loop free.
        return await asyncio.to_thread(self._resolve_sync, text)

    def _resolve_sync(self, text: str) -> str:
        preds = self._model.predict(texts=[text])
        if not preds:
            return text
        clusters = preds[0].get_clusters(as_strings=False)
        if not clusters:
            return text
        repls: list[tuple[int, int, str]] = []
        for cluster in clusters:
            spans = [(int(s), int(e)) for s, e in cluster]
            if len(spans) < 2:
                continue
            rep = max(spans, key=lambda se: se[1] - se[0])
            rep_str = text[rep[0]:rep[1]]
            for s, e in spans:
                if (s, e) == rep:
                    continue
                repls.append((s, e, rep_str))
        # Apply right-to-left so unmodified prefixes keep their offsets.
        for s, e, rep_str in sorted(repls, key=lambda r: r[0], reverse=True):
            text = text[:s] + rep_str + text[e:]
        return text


def get_coref_resolver() -> Optional[CorefResolver]:
    """Build the coref resolver named by COREF_BACKEND, or None when off.

    Returns None for unset/"none"/"off" so callers can cheaply skip the pass
    without constructing anything.
    """
    backend = os.getenv("COREF_BACKEND", "none").strip().lower()
    if backend in ("", "none", "off", "0", "false"):
        return None
    if backend == "llm":
        return _LLMCoref()
    if backend in ("spacy", "fastcoref", "neural"):
        return _SpacyCoref()
    logger.warning("Unknown COREF_BACKEND %r — coreference disabled", backend)
    return None


__all__ = ["CorefResolver", "get_coref_resolver"]
