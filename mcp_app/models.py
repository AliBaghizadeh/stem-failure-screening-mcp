from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    label: str
    path: str
    kind: Literal["image", "csv", "json", "directory", "text", "other"] = "other"
    media_type: str | None = None
    description: str = ""


class ToolRunManifest(BaseModel):
    run_id: str
    server: str
    tool: str
    status: Literal["ok", "error"] = "ok"
    output_dir: str
    inputs: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

