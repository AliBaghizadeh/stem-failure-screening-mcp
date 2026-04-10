r"""
CLI helper to sync a completed run folder to AWS using the same app cloud layer.

Example:
python scripts/sync_run_to_cloud.py ^
  --project GaN3 ^
  --run-folder "C:\Ali\microscopy datasets\peak_finding\GaN\GaN3\5_defects\48df855b" ^
  --stage defect_region_screening ^
  --run-id 48df855b ^
  --sample-name "GaN image 3"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cloud_sync import sync_run_directory_to_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync one completed pipeline run to AWS.")
    parser.add_argument("--project", required=True, help="Project or sample group id, e.g. GaN3")
    parser.add_argument("--run-folder", required=True, help="Completed run folder containing run artifacts")
    parser.add_argument("--stage", default="", help="Optional stage override")
    parser.add_argument("--run-id", default="", help="Optional run id override")
    parser.add_argument("--sample-name", default="", help="Optional human-readable sample name")
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip uploading preview images and upload metadata/statistics only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = sync_run_directory_to_cloud(
        project=args.project,
        output_dir=args.run_folder,
        stage=args.stage or None,
        run_id=args.run_id or None,
        sample_name=args.sample_name or None,
        include_previews=not args.no_previews,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
