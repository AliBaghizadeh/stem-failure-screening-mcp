from __future__ import annotations

import json
import os
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mcp_app.models import ToolRunManifest

DEFAULT_MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8005")
SERVER_MOUNT_PATHS = {
    "peak": "/mcp/peak",
    "strain": "/mcp/strain",
    "ml": "/mcp/ml",
    "project": "/mcp/project",
}


class MCPClientError(RuntimeError):
    pass


def _server_url(server: str, base_url: str | None = None) -> str:
    mount_path = SERVER_MOUNT_PATHS.get(server, server)
    if not mount_path.startswith("/"):
        mount_path = f"/{mount_path}"
    root = (base_url or DEFAULT_MCP_BASE_URL).rstrip("/")
    return f"{root}{mount_path}"


def _extract_text_content(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    text_payload = _extract_text_content(result)
    if text_payload:
        try:
            parsed = json.loads(text_payload)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"MCP tool returned non-JSON text payload: {text_payload[:200]}") from exc
        if isinstance(parsed, dict):
            return parsed

    raise MCPClientError("MCP tool did not return structured content.")


async def call_mcp_tool(
    server: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> ToolRunManifest:
    url = _server_url(server, base_url=base_url)

    async with streamable_http_client(url) as streams:
        read_stream, write_stream, _ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments or {})

    if getattr(result, "isError", False):
        detail = _extract_text_content(result) or f"MCP tool '{tool_name}' failed."
        raise MCPClientError(detail)

    payload = _extract_payload(result)
    try:
        return ToolRunManifest.model_validate(payload)
    except Exception as exc:
        raise MCPClientError(f"Invalid MCP manifest payload from '{tool_name}'.") from exc


def call_mcp_tool_sync(
    server: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> ToolRunManifest:
    return anyio.run(call_mcp_tool, server, tool_name, arguments, base_url)
