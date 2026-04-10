from __future__ import annotations

from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from .models import ToolRunManifest
from .reporting import (
    OLLAMA_MODEL,
    build_comparison_context,
    generate_report_with_ollama,
    render_professional_html,
    render_professional_markdown,
)
from .utils import ROOT, build_artifacts, ensure_directory, new_run_id, persist_manifest, summarize_directory, write_json

mcp = FastMCP("STEM Project Server", json_response=True, streamable_http_path="/")


@mcp.resource("stem://app/overview")
def app_overview() -> str:
    return (
        "STEM Atom Finder is a FastAPI web app for microscopy analysis with five main "
        "pipeline stages: grid search, peak finding, strain mapping, clustering, and "
        "defect detection. The MCP layer exposes those scientific capabilities as "
        "discoverable tools, resources, and prompts for agentic workflows."
    )


@mcp.resource("stem://app/pipeline")
def app_pipeline() -> str:
    return (
        "Pipeline order: raw image -> grid search -> peak finding -> strain map -> "
        "clustering -> defect detection. Scientific outputs are saved as CSV/PNG "
        "artifacts and mirrored in MCP run manifests."
    )


@mcp.resource("stem://mcp/servers")
def mcp_server_overview() -> str:
    return (
        "Mounted MCP servers: /mcp/peak, /mcp/strain, /mcp/ml, /mcp/project. "
        "Peak server handles parameter search and atom finding. Strain server handles "
        "strain maps. ML server handles clustering and defect detection. Project server "
        "exposes app-level context and lightweight filesystem inspection tools."
    )


@mcp.prompt()
def scientific_analysis_prompt(project_root: str, question: str) -> str:
    return (
        f"You are reviewing the microscopy project at '{project_root}'. "
        f"Answer this scientific question using the available MCP tools and resources: {question}"
    )


@mcp.tool()
def list_project_tree(root: str = str(ROOT), limit: int = 200) -> dict[str, object]:
    return summarize_directory(root=root, limit=limit)


@mcp.tool()
def preview_csv(csv_path: str, limit: int = 10) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    return {
        "path": csv_path,
        "columns": list(df.columns),
        "n_rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records"),
    }


@mcp.tool()
def compare_runs_report(
    run_a: str,
    run_b: str,
    question: str = "",
    report_id: str = "",
    prepared_by: str = "",
    approved_by: str = "",
    output_dir: str = "",
) -> ToolRunManifest:
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "project", "compare_runs_report", run_id)
    context = build_comparison_context(
        run_a,
        run_b,
        question=question,
        report_id=report_id,
        prepared_by=prepared_by,
        approved_by=approved_by,
    )
    narrative = generate_report_with_ollama(context)
    report_markdown = render_professional_markdown(context, narrative)
    report_html = render_professional_html(context, narrative)

    context_path = write_json(out_dir / "report_context.json", context)
    report_path = Path(out_dir) / "client_report.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    html_report_path = Path(out_dir) / "client_report.html"
    html_report_path.write_text(report_html, encoding="utf-8")

    manifest = ToolRunManifest(
        run_id=run_id,
        server="project",
        tool="compare_runs_report",
        output_dir=str(out_dir),
        inputs=[run_a, run_b],
        parameters={
            "question": question,
            "report_id": report_id,
            "prepared_by": prepared_by,
            "approved_by": approved_by,
            "model": OLLAMA_MODEL,
        },
        metrics={
            "report_characters": len(report_markdown),
            "run_a_tool": context["run_a"]["tool"],
            "run_b_tool": context["run_b"]["tool"],
            "screening_outcome": context["screening_outcome"]["decision"],
        },
        artifacts=build_artifacts([context_path, report_path, html_report_path]),
        notes=[
            "Report generated from structured manifests and CSV-derived metrics only.",
            f"Local Ollama model fixed to {OLLAMA_MODEL}.",
        ],
    )
    persist_manifest(manifest)
    return manifest


def main():
    mcp.run(transport="streamable-http")
