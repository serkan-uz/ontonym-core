"""Command-line interface for ontonym-core.

    $ echo "Sarah deployed prod at 14:02" | ontonym-core extract --mode both
    $ ontonym-core extract --text "..." --backend anthropic
    $ ontonym-core health --backend ollama
"""
from __future__ import annotations

import asyncio
import json
import sys

import typer

from .core import _resolve_backend, extract
from .schema import ClassExtraction, ObjectExtraction

app = typer.Typer(
    add_completion=False,
    help="ontonym-core — text → typed object graph (classes, objects, events).",
    no_args_is_help=True,
)


def _read_text(text: str | None) -> str:
    if text is not None:
        return text
    if sys.stdin.isatty():
        typer.echo(
            "error: pass --text or pipe input on stdin.", err=True
        )
        raise typer.Exit(code=2)
    return sys.stdin.read()


@app.command(name="extract")
def extract_cmd(  # noqa: D401
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Which pass(es) to run: 'class', 'object', or 'both'.",
    ),
    backend: str = typer.Option(
        "ollama",
        "--backend",
        help="LLM backend: 'ollama' (local, default) or 'anthropic' (hosted).",
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        help="Input text. If omitted, reads from stdin.",
    ),
    prior_classes_json: str | None = typer.Option(
        None,
        "--prior-classes",
        help="Path to a JSON file holding a prior ClassExtraction (diff-mode).",
    ),
    prior_objects_json: str | None = typer.Option(
        None,
        "--prior-objects",
        help="Path to a JSON file holding a prior ObjectExtraction (diff-mode).",
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty-print JSON output."),
) -> None:
    """Extract a typed graph from free text and print JSON to stdout."""
    if mode not in ("class", "object", "both"):
        typer.echo(f"error: --mode must be one of class|object|both, got {mode!r}", err=True)
        raise typer.Exit(code=2)

    raw_text = _read_text(text)
    prior_c = _load_optional(prior_classes_json, ClassExtraction)
    prior_o = _load_optional(prior_objects_json, ObjectExtraction)

    result = asyncio.run(
        extract(
            raw_text,
            mode=mode,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            prior_classes=prior_c,
            prior_objects=prior_o,
        )
    )
    typer.echo(result.model_dump_json(indent=2 if pretty else None))


@app.command()
def health(
    backend: str = typer.Option("ollama", "--backend", help="Backend to probe."),
) -> None:
    """Probe the configured backend — does it answer, and is the model loaded?"""
    b = _resolve_backend(backend)  # type: ignore[arg-type]
    info = asyncio.run(b.check_health())
    typer.echo(json.dumps(info, indent=2))


def _load_optional(path: str | None, cls):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return cls.model_validate(data)


if __name__ == "__main__":
    app()
