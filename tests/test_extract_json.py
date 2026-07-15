"""Tests for caller-owned generic JSON extraction prompts."""
from __future__ import annotations

import asyncio

import pytest

from ontonym_core import OllamaBackend, extract_json, parse_json_object


def test_parse_json_object_accepts_markdown_fence() -> None:
    assert parse_json_object('```json\n{"stances": [{"name": "careful"}]}\n```') == {
        "stances": [{"name": "careful"}],
    }


@pytest.mark.parametrize("raw", ["not json", "[]", '"value"'])
def test_parse_json_object_rejects_invalid_or_non_object_roots(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_json_object(raw)


def test_high_level_extract_json_dispatches_to_backend() -> None:
    class FakeBackend:
        async def extract_json(self, prompt: str, *, system_prompt: str | None = None):
            return {"prompt": prompt, "system_prompt": system_prompt}

    result = asyncio.run(
        extract_json(
            "extract rules",
            backend=FakeBackend(),
            system_prompt="Return JSON only",
        )
    )
    assert result == {
        "prompt": "extract rules",
        "system_prompt": "Return JSON only",
    }


def test_ollama_extract_json_forwards_custom_system_prompt() -> None:
    backend = OllamaBackend()
    seen: dict[str, str | None] = {}

    async def fake_generate(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        seen["system_prompt"] = system_prompt
        return '{"rules": []}'

    backend._generate = fake_generate  # type: ignore[method-assign]
    result = asyncio.run(
        backend.extract_json("custom prompt", system_prompt="custom system")
    )

    assert result == {"rules": []}
    assert seen == {
        "prompt": "custom prompt",
        "system_prompt": "custom system",
    }
