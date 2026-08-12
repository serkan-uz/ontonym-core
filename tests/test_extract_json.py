"""Tests for caller-owned generic JSON extraction prompts."""
from __future__ import annotations

import asyncio

import pytest

from ontonym_core import OllamaBackend, OpenAIBackend, extract_json, parse_json_object


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


def test_openai_extract_json_uses_generic_parser_and_system_prompt() -> None:
    backend = OpenAIBackend(api_key="test-key")
    seen: dict[str, str | None] = {}

    async def fake_invoke(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        seen["system_prompt"] = system_prompt
        return '{"actions": []}'

    backend._invoke = fake_invoke  # type: ignore[method-assign]
    result = asyncio.run(
        backend.extract_json("custom action prompt", system_prompt="JSON only")
    )

    assert result == {"actions": []}
    assert seen == {
        "prompt": "custom action prompt",
        "system_prompt": "JSON only",
    }


def test_openai_response_text_collects_message_output() -> None:
    data = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [
                {"type": "output_text", "text": '{"rules":'},
                {"type": "output_text", "text": "[]}"},
            ]},
        ],
    }
    assert OpenAIBackend._response_text(data) == '{"rules":[]}'


def test_openai_requires_api_key() -> None:
    backend = OpenAIBackend(api_key="")
    backend.api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(backend.extract_json("prompt"))


def test_openai_responses_api_payload_and_usage(monkeypatch) -> None:
    seen: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"objects": []}'}],
                }],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 3},
                },
            }

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url: str, *, json: dict, headers: dict):
            seen.update(url=url, payload=json, headers=headers)
            return FakeResponse()

    monkeypatch.setattr("ontonym_core.llm.httpx.AsyncClient", FakeClient)
    backend = OpenAIBackend(
        api_key="test-key", model="gpt-5-mini", base_url="https://api.openai.com/v1",
    )
    result = asyncio.run(
        backend.extract_json("extract", system_prompt="Return JSON")
    )

    assert result == {"objects": []}
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert seen["payload"]["instructions"] == "Return JSON"
    assert seen["payload"]["text"] == {"format": {"type": "json_object"}}
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert backend.drain_usage() == [{
        "model": "gpt-5-mini",
        "input_tokens": 12,
        "output_tokens": 4,
        "cache_read_tokens": 3,
        "cache_write_tokens": 0,
    }]
