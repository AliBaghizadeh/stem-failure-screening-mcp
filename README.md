# STEM Failure Screening MCP

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-app-009688?logo=fastapi&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-tool%20routing-111111)
![HyperSpy](https://img.shields.io/badge/HyperSpy-microscopy%20data-4C78A8)
![Atomap](https://img.shields.io/badge/Atomap-atomic%20column%20analysis-1F9D55)
![AWS](https://img.shields.io/badge/AWS-S3%20%2B%20DynamoDB-FF9900?logo=amazonaws&logoColor=white)
![Local LLM](https://img.shields.io/badge/Ollama-qwen2.5%3A14b-6A5ACD)

MCP-native HR-STEM pipeline for atom finding, strain mapping, clustering, defect-region screening, and LLM review in semiconductor failure screening.

## Overview

Semiconductor failure analysis increasingly depends on reading atomic-scale lattice distortions, missing columns, strain fields, and defective regions from HR-STEM images, but in practice this work is still often fragmented across manual scripts, expert-only interpretation, and disconnected tools. This project addresses that gap by combining [HyperSpy](https://hyperspy.org/hyperspy-doc/current/index.html) for microscopy data handling, [Atomap](https://atomap.org/) for atomic-column and sublattice analysis, and an agentic AI workflow that routes peak finding, strain mapping, clustering, defect-region screening, and run-to-run review through a structured application pipeline. The goal is not only to detect atoms, but to turn lattice-level microscopy into a reproducible, auditable, and decision-oriented workflow for semiconductor failure screening.

## Why this repository matters

- It converts microscopy analysis from one-off scripts into a traceable staged pipeline.
- It uses [Atomap](https://atomap.org/) and [HyperSpy](https://hyperspy.org/hyperspy-doc/current/index.html) as the scientific backbone rather than replacing trusted microscopy tooling.
- It uses MCP as an execution layer so app actions map to explicit scientific tools instead of hidden internal calls.
- It produces structured run outputs that are easier to review, compare, share, and eventually operationalize in metrology environments.
- It adds LLM review at the report layer, where the model reasons over manifests, statistics, and stage outputs instead of inventing image-level conclusions directly.

## Pipeline

The current workflow is:

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

## Technical Stack

Core scientific and engineering components:

- [HyperSpy](https://hyperspy.org/hyperspy-doc/current/index.html) for microscopy data loading, signal handling, and scientific I/O
- [Atomap](https://atomap.org/) for atomic-column fitting, sublattice logic, and lattice-scale analysis
- `FastAPI` for the local web application and orchestration layer
- `MCP` for standardized tool routing across pipeline stages
- `HDBSCAN` and related ML utilities for clustering and downstream screening
- `Ollama` with `qwen2.5:14b` for local LLM review
- `AWS S3 + DynamoDB` for compact run sharing and indexing

## MCP in this project

MCP is used here as the tool-routing layer between the app and the scientific pipeline.

In practice that means:

- the web app does not call each processing script directly
- stage execution is exposed through MCP tools
- scientific steps become easier to inspect, test, and extend
- the same tools can later be consumed by agents, other clients, or production wrappers

Current mounted MCP servers include:

- `/mcp/peak`
- `/mcp/strain`
- `/mcp/ml`
- `/mcp/project`

## Web Application

The app currently exposes the following analysis tabs:

- Peak Finding
- Strain Map
- Clustering
- Defect Region Screening
- LLM Review

The UI is built around practical analysis rather than demo-only visuals:

- fixed labels for key images instead of raw filenames
- stage-level statistics and QC-style interpretation flags
- support for sublattice-aware and single-sublattice workflows
- structured report generation for comparing completed runs
- optional cloud sync for sharing compact run artifacts

## Reproducibility and QC

One of the central goals of this project is to make atomic-resolution failure screening more reproducible.

To support that, the pipeline records:

- explicit run manifests
- saved statistics per stage
- structured CSV outputs
- standardized report headers and screening outcomes
- cloud-indexable artifacts for cross-team review

This makes the workflow more suitable for metrology-style review than ad hoc notebook outputs or screenshot-driven interpretation.

## Example Assets Included in GitHub

The public repository keeps two example GaAs images as lightweight sample assets:

- `data/1.tif`
- `data/2.tif`

Large experimental result folders, private notes, and local run logs are intentionally excluded from GitHub.

## Project Structure

```text
.
├── app/          FastAPI backend, UI, cloud endpoints, and MCP client routing
├── configs/      Configuration files and prompts
├── core/         Peak finding, grid search, preprocessing, strain, and sublattice logic
├── data/         Minimal public sample inputs
├── mcp_app/      MCP servers and report-building utilities
├── ml/           Clustering, feature engineering, and defect-region screening helpers
├── scripts/      Helper scripts for maintenance and cloud sync
├── .github/      CI workflow configuration
├── environment-mcp.yml
└── pyproject.toml
```

## Quick Start

From the project root:

```powershell
conda activate stem_mcp
python -m app.main
```

Then open:

- `http://localhost:8005/`

Useful validation endpoints:

- MCP status: `http://localhost:8005/api/mcp`
- cloud status: `http://localhost:8005/api/cloud/status`

## Environment

The active app workflow expects the `stem_mcp` Conda environment.

Primary setup files:

- `environment-mcp.yml`
- `pyproject.toml`

If you want to use AWS artifact sharing, configure:

- `CLOUD_SYNC_ENABLED`
- `AWS_REGION`
- `AWS_S3_BUCKET`
- `AWS_DYNAMODB_TABLE`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Reference file:

- `.env.cloud.example`

## Cloud Sharing

The scientific pipeline remains local-first, but selected outputs can be pushed to AWS for cross-team review.

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

## Repository Scope

This GitHub-oriented repo intentionally excludes:

- generated run results
- MLflow tracking directories
- private notes and logbooks
- local debug scripts
- large experimental working folders

That keeps the repository focused on reusable code, app infrastructure, MCP integration, and a minimal set of sample assets.

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙋 Contact

Questions, feedback, or collaborations?  
Open a GitHub issue or contact: `alibaghizade@gmail.com`
