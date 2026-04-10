from __future__ import annotations

from .ml_server import mcp as ml_mcp
from .peak_server import mcp as peak_mcp
from .project_server import mcp as project_mcp
from .strain_server import mcp as strain_mcp


def get_mcp_servers():
    return [
        ("/mcp/peak", peak_mcp),
        ("/mcp/strain", strain_mcp),
        ("/mcp/ml", ml_mcp),
        ("/mcp/project", project_mcp),
    ]
