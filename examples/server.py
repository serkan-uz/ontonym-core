"""Minimal FastAPI server demonstrating one HTTP route on top of ontonym-core.

Run locally:

    pip install 'ontonym-core[server]'    # or [server,anthropic]
    ollama serve                          # if using the default Ollama backend
    ollama pull llama3.1:8b               # one-time
    uvicorn examples.server:app --reload

Then:

    curl -X POST http://localhost:8000/extract \\
      -H 'content-type: application/json' \\
      -d '{"text": "Sarah deployed PaymentService at 14:02. An outage hit it at 14:05.", "mode": "both"}'

This is an EXAMPLE, not the library's primary shape. The library itself is
just a Python package; embed it however suits your stack.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ontonym_core import (
    ClassExtraction,
    Extraction,
    ObjectExtraction,
    extract,
)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: Literal["class", "object", "both"] = "both"
    prior_classes: ClassExtraction | None = None
    prior_objects: ObjectExtraction | None = None


app = FastAPI(title="ontonym-core example server")


@app.post("/extract", response_model=Extraction)
async def extract_route(req: ExtractRequest) -> Extraction:
    backend = os.getenv("EXTRACTOR_BACKEND", "ollama")
    try:
        return await extract(
            req.text,
            mode=req.mode,
            backend=backend,  # type: ignore[arg-type]
            prior_classes=req.prior_classes,
            prior_objects=req.prior_objects,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
