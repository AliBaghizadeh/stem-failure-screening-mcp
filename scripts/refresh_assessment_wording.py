from __future__ import annotations

import json
from pathlib import Path

from mcp_app.reporting import _normalize_stage_stats


def infer_tool(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith("_peak_stats.json"):
        return "peak_find_atoms"
    if name.endswith("_strain_stats.json"):
        return "compute_strain_map"
    if name.endswith("_cluster_stats.json"):
        return "cluster_atomic_environments"
    if name.endswith("_defect_stats.json"):
        return "detect_structural_defects"
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    updated: list[str] = []

    for project in ("GaAs1", "GaAs2"):
        for path in (root / project).rglob("*_stats.json"):
            tool = infer_tool(path)
            if tool is None:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            normalized = _normalize_stage_stats(tool, payload)
            path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            updated.append(str(path))

    print("\n".join(updated))


if __name__ == "__main__":
    main()
