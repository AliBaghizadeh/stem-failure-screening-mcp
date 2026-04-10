# STEM Atom Finder

Agentic microscopy pipeline for atomic-resolution semiconductor STEM analysis, with MCP-routed scientific tools, a local FastAPI app, and optional AWS run sharing.

## What this project does

This project turns atomic-resolution STEM analysis into a structured, multi-stage workflow instead of a collection of disconnected scripts.

The current pipeline supports:

1. grid search for peak-finding parameter selection
2. peak finding and atom fitting
3. sublattice separation, including single-sublattice mode for materials such as GaN HAADF
4. strain mapping
5. clustering of local atomic environments
6. defect region screening
7. LLM-based comparison reports across completed runs

Each stage writes structured outputs such as:

- `run_manifest.json`
- stage statistics JSON
- CSV tables
- PNG overlays and heatmaps
- LLM review reports

## Architecture

The repository combines four layers:

- `app/`: FastAPI backend and browser UI
- `mcp_app/`: MCP servers that expose scientific tools as callable endpoints
- `core/`: core microscopy analysis pipeline and run logic
- `ml/`: clustering and downstream data-analysis helpers

This design keeps the scientific implementation in Python while routing stage execution through a standardized tool layer.

## MCP in this project

MCP is used here as the tool-routing layer between the app and the scientific pipeline.

In practice that means:

- the web app does not call each processing script directly
- stage execution is exposed through MCP tools
- runs can be standardized, inspected, extended, and later consumed by agents or other clients

Current mounted MCP servers include:

- `/mcp/peak`
- `/mcp/strain`
- `/mcp/ml`
- `/mcp/project`

## Web app

The app provides stage tabs for:

- Peak Finding
- Strain Map
- Clustering
- Defect Region Screening
- LLM Review

The UI is designed for practical microscopy workflow use:

- fixed image labels for key outputs
- per-stage statistics
- QC-style interpretation flags
- optional cloud sync foundation for sharing compact run artifacts

## Example assets included in GitHub

The public repo keeps two example GaAs images as lightweight reference assets:

- `data/1.tif`
- `data/2.tif`

These are included as sample inputs for repository demonstration. Large run folders, generated results, and private working datasets are intentionally excluded from GitHub.

## Quick start

From the project root:

```powershell
conda activate stem_mcp
python -m app.main
```

Then open:

- `http://localhost:8005/`

To verify MCP routing:

- `http://localhost:8005/api/mcp`

To verify cloud-sync configuration:

- `http://localhost:8005/api/cloud/status`

## Environment

The project currently expects the `stem_mcp` Conda environment for the active app workflow.

Important files:

- `environment-mcp.yml`
- `pyproject.toml`

If you need the AWS sharing path, set these environment variables:

- `CLOUD_SYNC_ENABLED`
- `AWS_REGION`
- `AWS_S3_BUCKET`
- `AWS_DYNAMODB_TABLE`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

See:

- `.env.cloud.example`

## Cloud sharing

The scientific pipeline remains local-first, but selected run outputs can be pushed to AWS for cross-team review.

Currently synced artifact types:

- `run_manifest.json`
- `*_stats.json`
- `*.csv`
- `*.md`
- `*.html`
- up to `2` preview images

Cloud sync entry points:

- FastAPI: `POST /api/cloud/sync-run`
- CLI: `python scripts/sync_run_to_cloud.py ...`

The intended AWS pattern is:

- `S3` for run artifacts
- `DynamoDB` for compact run indexing

## Development

Useful local validation:

```powershell
python -m compileall app core ml mcp_app
```

## CI

GitHub Actions is configured for lightweight checks only:

- project metadata sanity
- Python compilation

Heavy microscopy execution remains local and should only move to CI later through a controlled self-hosted runner strategy.

## Repository scope

This GitHub-oriented repo intentionally excludes:

- generated run results
- MLflow tracking directories
- private notes and logbooks
- local debug scripts
- large experimental working folders

That keeps the repository focused on reusable code, app infrastructure, MCP integration, and a minimal set of sample assets.
