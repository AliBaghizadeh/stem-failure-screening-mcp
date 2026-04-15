"""
app/main.py
Multi-agent FastAPI backend for STEM Atom Finder.
Each agent is an async background task; results are polled via /status/{task_id}.
"""
import os, sys, uuid, time, logging, traceback
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

# ── Windows DLL Catch: Monkey-patch threadpoolctl ──────────────────────────
# Resolves OSError: [WinError -1066598274] Windows Error 0xc06d007e 
# which occurs in scikit-learn when it tries to walk DLLs to check threads.
try:
    import types
    class MockThreadpoolController:
        def __init__(self, *args, **kwargs): self.lib_controllers = []
        def select(self, *args, **kwargs): return self
        def limit(self, *args, **kwargs): return self
        def info(self): return []
        def __enter__(self): return self
        def __exit__(self, *args, **kwargs): pass
    mock_mod = types.ModuleType("threadpoolctl")
    mock_mod.ThreadpoolController = MockThreadpoolController
    mock_mod.threadpool_limits = lambda *args, **kwargs: MockThreadpoolController()
    mock_mod.threadpool_info = lambda *args, **kwargs: []
    mock_mod._get_threadpool_controller = lambda *args, **kwargs: MockThreadpoolController()
    sys.modules["threadpoolctl"] = mock_mod
    logging.info("Windows DLL bypass active (threadpoolctl patched).")
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor, Future
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NUMBA_CACHE_DIR = ROOT / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HYPERSPY_DISABLE_PLUGIN_SCAN", "1")
os.environ.setdefault("HYPERSPY_GUI_BACKEND", "none")
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

UPLOAD_DIR = ROOT / "data" / "_uploads"
RESULTS_DIR = ROOT / "results" / "_app"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("stem-app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

_MCP_SERVER_MOUNTS = []
_MCP_IMPORT_ERROR = None
try:
    from mcp_app.mounts import get_mcp_servers
    _MCP_SERVER_MOUNTS = get_mcp_servers()
except Exception as exc:
    _MCP_IMPORT_ERROR = str(exc)

_MCP_CLIENT_IMPORT_ERROR = None
try:
    from app.mcp_client import MCPClientError, call_mcp_tool_sync
except Exception as exc:
    MCPClientError = RuntimeError
    call_mcp_tool_sync = None
    _MCP_CLIENT_IMPORT_ERROR = str(exc)

_CLOUD_SYNC_IMPORT_ERROR = None
try:
    from app.cloud_config import get_cloud_config
    from app.cloud_sync import sync_run_directory_to_cloud
except Exception as exc:
    get_cloud_config = None
    sync_run_directory_to_cloud = None
    _CLOUD_SYNC_IMPORT_ERROR = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        for _, mcp_server in _MCP_SERVER_MOUNTS:
            await stack.enter_async_context(mcp_server.session_manager.run())
        if _MCP_SERVER_MOUNTS:
            logger.info("MCP servers mounted: %s", ", ".join(path for path, _ in _MCP_SERVER_MOUNTS))
        elif _MCP_IMPORT_ERROR:
            logger.info("MCP layer unavailable: %s", _MCP_IMPORT_ERROR)
        yield

app = FastAPI(title="STEM Atom Finder — Multi-Agent API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Hardware detection ───────────────────────────────────────────────────────
def get_current_hw():
    try:
        from core.device import DeviceManager
        dev = DeviceManager(enabled=True)
        return dev.get_hardware_status()
    except:
        return {"type": "CPU", "name": "Standard CPU"}

# Serve the static frontend
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Rebuild the FastAPI app with lifespan support so MCP servers can be mounted
# without disturbing the existing route structure below.
app = FastAPI(title="STEM Atom Finder â€” Multi-Agent API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
for mount_path, mcp_server in _MCP_SERVER_MOUNTS:
    app.mount(mount_path, mcp_server.streamable_http_app())

# ── Task registry (in-memory) ────────────────────────────────────────────────
_tasks: dict[str, dict] = {}   # task_id → {status, agent, outputs, error, progress, hardware, logs}
_executor = ThreadPoolExecutor(max_workers=8)

class TaskLoggingHandler(logging.Handler):
    def __init__(self, tid):
        super().__init__()
        self.tid = tid
    def emit(self, record):
        if self.tid in _tasks:
            msg = self.format(record)
            _tasks[self.tid]["logs"].append(msg)
            if len(_tasks[self.tid]["logs"]) > 100:
                _tasks[self.tid]["logs"].pop(0)

def _new_task(agent: str, *, project: str = "", output_dir: str = "", auto_sync: bool = False) -> str:
    tid = str(uuid.uuid4())[:8]
    _tasks[tid] = {
        "status": "queued", 
        "agent": agent, 
        "outputs": [], 
        "error": None, 
        "started": time.time(),
        "progress": 0.0,
        "hardware": get_current_hw(),
        "logs": [f"Task {tid} initiated."],
        "project": project,
        "output_dir": output_dir,
        "auto_sync": auto_sync,
        "cloud_sync": None,
    }
    return tid


def _task_done(tid: str, outputs: list[str]):
    _tasks[tid].update({"status": "done", "outputs": outputs, "elapsed": time.time() - _tasks[tid]["started"]})


def _task_error(tid: str, err: str):
    _tasks[tid].update({"status": "error", "error": err})


def _cloud_project_id(task: dict) -> str:
    project = str(task.get("project") or "").strip()
    if not project:
        return "demo"
    return Path(project).name if (":" in project or project.startswith("/") or project.startswith("\\")) else project


def _run_auto_cloud_sync(tid: str):
    task = _tasks.get(tid)
    if not task or not task.get("auto_sync"):
        return
    if sync_run_directory_to_cloud is None or get_cloud_config is None:
        task["cloud_sync"] = {"status": "error", "error": _CLOUD_SYNC_IMPORT_ERROR or "Cloud sync unavailable."}
        return

    output_dir = str(task.get("output_dir") or "").strip()
    if not output_dir:
        task["cloud_sync"] = {"status": "error", "error": "No output directory recorded for task."}
        return

    task["cloud_sync"] = {"status": "running"}
    try:
        result = sync_run_directory_to_cloud(
            project=_cloud_project_id(task),
            output_dir=output_dir,
            run_id=Path(output_dir).name,
            sample_name=_cloud_project_id(task),
            include_previews=True,
        )
        task["cloud_sync"] = {"status": "done", **result}
        task["logs"].append(f"AWS sync completed: s3://{result['bucket']}/projects/{result['project']}/runs/{result['run_id']}/")
    except Exception as exc:
        task["cloud_sync"] = {"status": "error", "error": str(exc)}
        task["logs"].append(f"AWS sync failed: {exc}")


# ── Agent worker functions (top-level for pickling) ──────────────────────────

def _worker_grid_search(image_path: str, out_dir: str, sep_range: list, thresh_range: list, 
                        tid: str, hardware: str, workers: int, 
                        flip_h: bool = False, flip_v: bool = False) -> list[str]:
    # Setup logger for this thread
    h = TaskLoggingHandler(tid)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(h)
    
    try:
        logging.info(f"Agent 'grid-search' ({hardware.upper()}) is starting for task {tid}...")
        sys.path.insert(0, str(ROOT))
        from core.grid_search import GridSearchEngine
        
        def prog(curr, total):
            _tasks[tid]["progress"] = round(curr/total, 2)
            _tasks[tid]["progress_detail"] = f"{curr}/{total}"
            logging.info(f"Progress: {curr}/{total}")

        engine = GridSearchEngine(max_workers=workers)
        config = {"separations": sep_range, "thresholds": thresh_range, "pca_options": [True]}
        
        logging.info(f"Searching {len(sep_range)*len(thresh_range)} combinations...")
        best_params, all_results = engine.search(
            image_path=image_path,
            config=config,
            plot_dir=out_dir,
            progress_callback=prog,
            hardware=hardware,
            flip_h=flip_h,
            flip_v=flip_v
        )
        return [str(Path(out_dir) / "summary_contact_sheet.png")]
    except Exception as e:
        logging.error(f"Grid search failed: {str(e)}")
        raise e
    finally:
        root_logger.removeHandler(h)


def _worker_grid_search_via_mcp(image_path: str, out_dir: str, sep_range: list, thresh_range: list,
                                tid: str, hardware: str, workers: int,
                                flip_h: bool = False, flip_v: bool = False) -> list[str]:
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise RuntimeError(detail)
    if not sep_range or not thresh_range:
        raise RuntimeError("Grid search requires non-empty separation and threshold ranges.")

    sep_sorted = sorted(set(int(v) for v in sep_range))
    thresh_sorted = sorted(set(float(v) for v in thresh_range))
    sep_step = sep_sorted[1] - sep_sorted[0] if len(sep_sorted) > 1 else 1
    thresh_step = round(thresh_sorted[1] - thresh_sorted[0], 4) if len(thresh_sorted) > 1 else 0.1

    manifest = call_mcp_tool_sync(
        "peak",
        "grid_search_image",
        {
            "image_path": image_path,
            "output_dir": out_dir,
            "sep_min": sep_sorted[0],
            "sep_max": sep_sorted[-1],
            "sep_step": sep_step,
            "thresh_min": thresh_sorted[0],
            "thresh_max": thresh_sorted[-1],
            "thresh_step": thresh_step,
            "hardware": hardware,
            "workers": workers,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
    )
    outputs = [
        artifact.path
        for artifact in manifest.artifacts
        if Path(artifact.path).exists() and not Path(artifact.path).name.endswith("run_manifest.json")
    ]
    if not outputs:
        raise MCPClientError("Grid search completed through MCP but returned no displayable outputs.")
    return sorted(outputs)


def _worker_analyze(image_path: str, out_dir: str, separation: int, threshold: float, 
                    hardware: str = "cpu", workers: int = 4, 
                    flip_h: bool = False, flip_v: bool = False) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from core.batch_processor import BatchProcessor

    # Use GPU sequential or CPU parallel
    if hardware == "gpu":
        max_workers = 1  # GPU processes sequentially
        logging.info(f"Peak Finding: GPU mode (sequential, 1 worker)")
    else:
        max_workers = max(1, workers)
        logging.info(f"Peak Finding: CPU mode ({max_workers} workers)")
    
    processor = BatchProcessor(max_workers=max_workers)
    config = {
        "mode": "full",
        "peak_detection": {
            "separation": separation,
            "threshold": threshold,
            "pca_enabled": True
        }
    }
    
    # The image was saved into its own unique task folder (tid)
    input_folder = str(Path(image_path).parent)
    
    summary = processor.run(
        input_folder=input_folder,
        output_folder=out_dir,
        config=config,
        flip_h=flip_h,
        flip_v=flip_v
    )
    
    if summary.failed > 0:
        raise RuntimeError(f"Analysis failed. See terminal logs.")
        
    pngs = list(Path(out_dir).glob("*.png"))
    csvs = list(Path(out_dir).glob("*.csv"))
    return sorted([str(p) for p in pngs + csvs])


def _worker_analyze_via_mcp(image_path: str, out_dir: str, separation: int, threshold: float,
                            sublattice_mode: str = "two",
                            sublattice_method: str = "local_pairing",
                            sublattice_pair_max_dist: float | None = None,
                            sublattice_pair_min_dist: float = 0.0,
                            hardware: str = "cpu", workers: int = 4,
                            flip_h: bool = False, flip_v: bool = False) -> list[str]:
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise RuntimeError(detail)

    manifest = call_mcp_tool_sync(
        "peak",
        "peak_find_atoms",
        {
            "image_path": image_path,
            "output_dir": out_dir,
            "separation": separation,
            "threshold": threshold,
            "sublattice_mode": sublattice_mode,
            "sublattice_method": sublattice_method,
            "sublattice_pair_max_dist": sublattice_pair_max_dist,
            "sublattice_pair_min_dist": sublattice_pair_min_dist,
            "hardware": hardware,
            "workers": workers,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
    )
    outputs = [
        artifact.path
        for artifact in manifest.artifacts
        if Path(artifact.path).exists() and not Path(artifact.path).name.endswith("run_manifest.json")
    ]
    if not outputs:
        raise MCPClientError("Peak finding completed through MCP but returned no displayable outputs.")
    return sorted(outputs)


def _worker_strain(csv_path: str, image_path: str, out_dir: str, 
                   prune_thresh: float = 0.0,
                   axis_tolerance_scale: float = 0.7,
                   min_axis_neighbors: int = 2,
                   flip_h: bool = False, flip_v: bool = False) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from core.strain_analysis import generate_strain_maps
    out_csv = generate_strain_maps(
        csv_path=csv_path,
        image_path=image_path,
        out_dir=out_dir,
        flip_h=flip_h,
        flip_v=flip_v,
        prune_thresh=prune_thresh,
        axis_tolerance_scale=axis_tolerance_scale,
        min_axis_neighbors=min_axis_neighbors,
    )
    pngs = list(Path(out_dir).glob("*_contour_*.png")) + \
           list(Path(out_dir).glob("*_strain_*.png")) + \
           list(Path(out_dir).glob("*_line_profiles.png"))
    outputs = [str(p) for p in pngs]
    if out_csv:
        outputs.append(str(out_csv))
    outputs.extend(str(p) for p in Path(out_dir).glob("*_strain_stats.json"))
    return outputs


def _worker_strain_via_mcp(csv_path: str, image_path: str, out_dir: str,
                           prune_thresh: float = 0.0,
                           axis_tolerance_scale: float = 0.7,
                           min_axis_neighbors: int = 2,
                           flip_h: bool = False, flip_v: bool = False) -> list[str]:
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise RuntimeError(detail)
    manifest = call_mcp_tool_sync(
        "strain",
        "compute_strain_map",
        {
            "csv_path": csv_path,
            "image_path": image_path,
            "output_dir": out_dir,
            "prune_thresh": prune_thresh,
            "axis_tolerance_scale": axis_tolerance_scale,
            "min_axis_neighbors": min_axis_neighbors,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
    )
    outputs = [
        artifact.path
        for artifact in manifest.artifacts
        if Path(artifact.path).exists() and not Path(artifact.path).name.endswith("run_manifest.json")
    ]
    if not outputs:
        raise MCPClientError("Strain analysis completed through MCP but returned no displayable outputs.")
    return sorted(outputs)


def _worker_cluster(csv_path: str, image_path: str, out_dir: str,
                    min_cluster_size: int, gmm_components: int,
                    flip_h: bool = False, flip_v: bool = False) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from ml.feature_engineering import run_clustering
    out_csv = run_clustering(csv_path=csv_path, image_path=image_path, out_dir=out_dir,
                             min_cluster_size=min_cluster_size, gmm_components=gmm_components,
                             flip_h=flip_h, flip_v=flip_v)
    pngs = list(Path(out_dir).glob("*_cluster_map.png"))
    jsons = list(Path(out_dir).glob("*_cluster_stats.json"))
    return [str(p) for p in pngs] + [out_csv] + [str(p) for p in jsons]


def _worker_cluster_via_mcp(csv_path: str, image_path: str, out_dir: str,
                            min_cluster_size: int, gmm_components: int,
                            flip_h: bool = False, flip_v: bool = False) -> list[str]:
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise RuntimeError(detail)
    manifest = call_mcp_tool_sync(
        "ml",
        "cluster_atomic_environments",
        {
            "csv_path": csv_path,
            "image_path": image_path or "",
            "output_dir": out_dir,
            "min_cluster_size": min_cluster_size,
            "gmm_components": gmm_components,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
    )
    outputs = [
        artifact.path
        for artifact in manifest.artifacts
        if Path(artifact.path).exists() and not Path(artifact.path).name.endswith("run_manifest.json")
    ]
    if not outputs:
        raise MCPClientError("Clustering completed through MCP but returned no displayable outputs.")
    return sorted(outputs)


def _worker_defects(csv_path: str, image_path: str, out_dir: str, 
                    peak_csv_path: str | None = None,
                    strain_csv_path: str | None = None,
                    peak_manifest_path: str | None = None,
                    strain_manifest_path: str | None = None,
                    region_threshold: float = 0.45,
                    blur_sigma: float = 2.0,
                    min_region_atoms: int = 12,
                    peak_weight: float = 0.45,
                    cluster_weight: float = 0.30,
                    strain_weight: float = 0.25,
                    flip_h: bool = False, flip_v: bool = False) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from ml.defect_detection import detect_defects
    out_csv = detect_defects(
        csv_path=csv_path,
        image_path=image_path,
        out_dir=out_dir,
        peak_csv_path=peak_csv_path,
        strain_csv_path=strain_csv_path,
        peak_manifest_path=peak_manifest_path,
        strain_manifest_path=strain_manifest_path,
        region_threshold=region_threshold,
        blur_sigma=blur_sigma,
        min_region_atoms=min_region_atoms,
        peak_weight=peak_weight,
        cluster_weight=cluster_weight,
        strain_weight=strain_weight,
        flip_h=flip_h,
        flip_v=flip_v,
    )
    pngs = list(Path(out_dir).glob("*_defect_region_heatmap.png")) + \
           list(Path(out_dir).glob("*_defect_region_overlay.png"))
    csvs = [out_csv] + [str(p) for p in Path(out_dir).glob("*_defect_region_summary.csv")]
    jsons = list(Path(out_dir).glob("*_defect_region_stats.json"))
    return [str(p) for p in pngs] + csvs + [str(p) for p in jsons]


def _worker_defects_via_mcp(csv_path: str, image_path: str, out_dir: str,
                            peak_csv_path: str | None = None,
                            strain_csv_path: str | None = None,
                            peak_manifest_path: str | None = None,
                            strain_manifest_path: str | None = None,
                            region_threshold: float = 0.45,
                            blur_sigma: float = 2.0,
                            min_region_atoms: int = 12,
                            peak_weight: float = 0.45,
                            cluster_weight: float = 0.30,
                            strain_weight: float = 0.25,
                            flip_h: bool = False, flip_v: bool = False) -> list[str]:
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise RuntimeError(detail)
    manifest = call_mcp_tool_sync(
        "ml",
        "detect_structural_defects",
        {
            "csv_path": csv_path,
            "image_path": image_path or "",
            "output_dir": out_dir,
            "peak_csv_path": peak_csv_path or "",
            "strain_csv_path": strain_csv_path or "",
            "peak_manifest_path": peak_manifest_path or "",
            "strain_manifest_path": strain_manifest_path or "",
            "region_threshold": region_threshold,
            "blur_sigma": blur_sigma,
            "min_region_atoms": min_region_atoms,
            "peak_weight": peak_weight,
            "cluster_weight": cluster_weight,
            "strain_weight": strain_weight,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
    )
    outputs = [
        artifact.path
        for artifact in manifest.artifacts
        if Path(artifact.path).exists() and not Path(artifact.path).name.endswith("run_manifest.json")
    ]
    if not outputs:
        raise MCPClientError("Defect detection completed through MCP but returned no displayable outputs.")
    return sorted(outputs)


def _submit(tid: str, fn, *args):
    """Submit a worker to the process pool and attach callbacks."""
    _tasks[tid]["status"] = "running"
    future: Future = _executor.submit(fn, *args)

    def _cb(f: Future):
        if f.exception():
            err_tb = "".join(traceback.format_exception(type(f.exception()), f.exception(), f.exception().__traceback__))
            logging.error(f"Task {tid} failed:\n{err_tb}")
            _task_error(tid, err_tb.splitlines()[-1]) # Show the main error in UI but keep full in logs
        else:
            _task_done(tid, f.result())
            _run_auto_cloud_sync(tid)

    future.add_done_callback(_cb)


# ── Utility ───────────────────────────────────────────────────────────────────

def _save_upload(file: UploadFile, subfolder: str = "") -> str:
    dest = UPLOAD_DIR / subfolder
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / file.filename
    with open(path, "wb") as f:
        f.write(file.file.read())
    return str(path)


def _out_dir(tid: str, agent_name: str, project: str = "") -> str:
    # Mapping agent internal names to user-requested numbered folder names
    mapping = {
        "grid-search": "1_grid_search",
        "analyze": "2_peak_finding",
        "strain": "3_strain_map",
        "cluster": "4_clustering",
        "defects": "5_defects"
    }
    subfolder = mapping.get(agent_name, agent_name)
    
    # If project is a full absolute path (detected by colon or slash)
    if project and (":" in project or project.startswith("/") or project.startswith("\\")):
        root_proj = Path(project)
        d = root_proj / subfolder / tid
    elif project:
        # Standard relative project in results/projects/
        clean = "".join(c for c in project if c.isalnum() or c in ("-", "_")).strip()
        d = ROOT / "results" / "projects" / clean / subfolder / tid
    else:
        d = RESULTS_DIR / tid
        
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _to_repo_file_url(path: str) -> Optional[str]:
    target = Path(path).resolve()
    try:
        rel = target.relative_to(ROOT)
    except ValueError:
        return None
    return f"/files/{rel.as_posix()}"


class CompareRunsRequest(BaseModel):
    run_a: str
    run_b: str
    question: str = ""
    report_id: str = ""
    prepared_by: str = ""
    approved_by: str = ""
    output_dir: str = ""


class SyncRunRequest(BaseModel):
    project: str
    run_folder: str
    stage: str = ""
    run_id: str = ""
    sample_name: str = ""
    include_previews: bool = True


@app.get("/api/browse-dir")
async def browse_directory():
    """Opens a native Windows directory picker on the host system."""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Bring picker to front
    
    directory = filedialog.askdirectory(title="Select Root Project Directory")
    root.destroy()
    
    if directory:
        return {"path": str(Path(directory).resolve())}
    return {"path": None}


@app.get("/api/browse-run-dir")
async def browse_run_directory():
    """Opens a native directory picker for selecting a completed run folder."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    directory = filedialog.askdirectory(title="Select Run Folder (contains run_manifest.json)")
    root.destroy()

    if directory:
        return {"path": str(Path(directory).resolve())}
    return {"path": None}



# ── Status endpoint ───────────────────────────────────────────────────────────

@app.get("/status/{task_id}")
def get_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    t = _tasks[task_id]
    
    # Poll progress file for grid search
    if t["agent"] == "grid-search" and t["status"] == "running":
        prog_file = RESULTS_DIR / task_id / "_progress.json"
        if prog_file.exists():
            try:
                with open(prog_file, "r") as f:
                    pdata = json.load(f)
                    t["progress"] = pdata["progress"]
                    t["progress_detail"] = f"{pdata['curr']}/{pdata['total']}"
            except: pass

    # Get recent logs
    recent_logs = t.get("logs", [])
    
    # Expose outputs through a task-scoped file endpoint so results can live
    # either inside the repo or in an arbitrary external project directory.
    outputs = []
    for i, p in enumerate(t.get("outputs", [])):
        name = Path(p).name
        outputs.append(f"/task-file/{task_id}/{i}/{name}")
    return {**t, "outputs": outputs, "logs": recent_logs}


@app.get("/tasks")
def list_tasks():
    return [{**v, "id": k} for k, v in _tasks.items()]


@app.get("/api/mcp")
def mcp_info():
    return {
        "enabled": bool(_MCP_SERVER_MOUNTS),
        "mounts": [path for path, _ in _MCP_SERVER_MOUNTS],
        "error": _MCP_IMPORT_ERROR,
        "client_error": _MCP_CLIENT_IMPORT_ERROR,
    }


@app.get("/api/cloud/status")
def cloud_status():
    if get_cloud_config is None:
        return {
            "enabled": False,
            "configured": False,
            "region": None,
            "bucket": None,
            "dynamodb_table": None,
            "error": _CLOUD_SYNC_IMPORT_ERROR,
        }

    config = get_cloud_config()
    identity = {}
    if config.enabled and config.is_configured and sync_run_directory_to_cloud is not None:
        try:
            from app.cloud_sync import _caller_identity
            identity = _caller_identity(config)
        except Exception:
            identity = {}
    return {
        "enabled": config.enabled,
        "configured": config.is_configured,
        "region": config.region,
        "bucket": config.bucket,
        "dynamodb_table": config.dynamodb_table,
        "aws_account_id": identity.get("account_id"),
        "synced_by": identity.get("arn"),
        "error": None,
    }


@app.post("/api/cloud/sync-run")
def sync_run_to_cloud(payload: SyncRunRequest):
    if sync_run_directory_to_cloud is None:
        detail = _CLOUD_SYNC_IMPORT_ERROR or "Cloud sync layer is unavailable."
        raise HTTPException(503, detail)

    try:
        result = sync_run_directory_to_cloud(
            project=payload.project,
            output_dir=payload.run_folder,
            stage=payload.stage or None,
            run_id=payload.run_id or None,
            sample_name=payload.sample_name or None,
            include_previews=payload.include_previews,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    return result


class SyncTaskRequest(BaseModel):
    task_id: str
    include_previews: bool = True


@app.post("/api/cloud/sync-task")
def sync_task_to_cloud(payload: SyncTaskRequest):
    if payload.task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    task = _tasks[payload.task_id]
    output_dir = str(task.get("output_dir") or "").strip()
    if not output_dir:
        raise HTTPException(400, "Task has no recorded output directory.")
    if sync_run_directory_to_cloud is None:
        detail = _CLOUD_SYNC_IMPORT_ERROR or "Cloud sync layer is unavailable."
        raise HTTPException(503, detail)
    try:
        result = sync_run_directory_to_cloud(
            project=_cloud_project_id(task),
            output_dir=output_dir,
            run_id=Path(output_dir).name,
            sample_name=_cloud_project_id(task),
            include_previews=payload.include_previews,
        )
        task["cloud_sync"] = {"status": "done", **result}
        return result
    except Exception as exc:
        task["cloud_sync"] = {"status": "error", "error": str(exc)}
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/llm-review/compare-runs")
def compare_runs_review(payload: CompareRunsRequest):
    if call_mcp_tool_sync is None:
        detail = _MCP_CLIENT_IMPORT_ERROR or _MCP_IMPORT_ERROR or "MCP client layer is unavailable."
        raise HTTPException(503, detail)

    manifest = call_mcp_tool_sync(
        "project",
        "compare_runs_report",
        {
            "run_a": payload.run_a,
            "run_b": payload.run_b,
            "question": payload.question,
            "report_id": payload.report_id,
            "prepared_by": payload.prepared_by,
            "approved_by": payload.approved_by,
            "output_dir": payload.output_dir,
        },
    )
    report_path = None
    html_report_path = None
    context_path = None
    for artifact in manifest.artifacts:
        suffix = Path(artifact.path).suffix.lower()
        if suffix == ".md":
            report_path = artifact.path
        elif suffix == ".html":
            html_report_path = artifact.path
        elif Path(artifact.path).name == "report_context.json":
            context_path = artifact.path

    report_markdown = Path(report_path).read_text(encoding="utf-8") if report_path else ""
    return {
        "manifest": manifest.model_dump(),
        "report_markdown": report_markdown,
        "report_path": report_path,
        "report_url": _to_repo_file_url(report_path) if report_path else None,
        "report_html_path": html_report_path,
        "report_html_url": _to_repo_file_url(html_report_path) if html_report_path else None,
        "context_path": context_path,
        "context_url": _to_repo_file_url(context_path) if context_path else None,
    }


# ── File serving ──────────────────────────────────────────────────────────────

@app.get("/files/{path:path}")
def serve_file(path: str):
    full = ROOT / path
    if not full.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(full))


@app.get("/task-file/{task_id}/{file_index}")
def serve_task_file(task_id: str, file_index: int):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")

    outputs = _tasks[task_id].get("outputs", [])
    if file_index < 0 or file_index >= len(outputs):
        raise HTTPException(404, "Task output not found")

    full = Path(outputs[file_index])
    if not full.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(str(full), filename=full.name)


@app.get("/task-file/{task_id}/{file_index}/{filename}")
def serve_task_file_named(task_id: str, file_index: int, filename: str):
    return serve_task_file(task_id, file_index)


# ── Agent endpoints ───────────────────────────────────────────────────────────

@app.post("/run/grid-search")
async def run_grid_search(
    image: UploadFile = File(...),
    sep_min: int = Form(4),
    sep_max: int = Form(10),
    sep_step: int = Form(2),
    thresh_min: float = Form(0.3),
    thresh_max: float = Form(0.6),
    thresh_step: float = Form(0.1),
    hardware: str = Form("cpu"),
    workers: int = Form(4),
    project: str = Form(""),
    auto_sync: bool = Form(False),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False)
):
    tid = _new_task("grid-search", project=project, auto_sync=auto_sync)
    image_path = _save_upload(image, tid)
    out_dir = _out_dir(tid, "grid-search", project)
    _tasks[tid]["output_dir"] = out_dir
    sep_range = list(range(sep_min, sep_max + 1, sep_step))
    thresh_range = [round(thresh_min + i * thresh_step, 2)
                    for i in range(int((thresh_max - thresh_min) / thresh_step + 0.001) + 1)]
    _submit(tid, _worker_grid_search_via_mcp, image_path, out_dir, 
            sep_range, thresh_range, tid, hardware, workers, flip_h, flip_v)
    return {"task_id": tid}


@app.post("/run/analyze")
async def run_analyze(
    image: UploadFile = File(...),
    separation: int = Form(6),
    threshold: float = Form(0.5),
    sublattice_mode: str = Form("two"),
    sublattice_method: str = Form("local_pairing"),
    sublattice_pair_max_dist: Optional[float] = Form(None),
    sublattice_pair_min_dist: float = Form(0.0),
    hardware: str = Form("cpu"),
    workers: int = Form(4),
    project: str = Form(""),
    auto_sync: bool = Form(False),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False)
):
    tid = _new_task("analyze", project=project, auto_sync=auto_sync)
    image_path = _save_upload(image, tid)
    out_dir = _out_dir(tid, "analyze", project)
    _tasks[tid]["output_dir"] = out_dir
    _submit(tid, _worker_analyze_via_mcp, image_path, out_dir, 
            separation, threshold, sublattice_mode, sublattice_method, sublattice_pair_max_dist, sublattice_pair_min_dist,
            hardware, workers, flip_h, flip_v)
    return {"task_id": tid}


@app.post("/run/strain")
async def run_strain(
    csv: UploadFile = File(...),
    image: UploadFile = File(...),
    prune_thresh: float = Form(0.0),
    axis_tolerance_scale: float = Form(0.7),
    min_axis_neighbors: int = Form(2),
    project: str = Form(""),
    auto_sync: bool = Form(False),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False)
):
    tid = _new_task("strain", project=project, auto_sync=auto_sync)
    csv_path = _save_upload(csv, tid)
    image_path = _save_upload(image, tid)
    out_dir = _out_dir(tid, "strain", project)
    _tasks[tid]["output_dir"] = out_dir
    _submit(
        tid,
        _worker_strain_via_mcp,
        csv_path,
        image_path,
        out_dir,
        prune_thresh,
        axis_tolerance_scale,
        min_axis_neighbors,
        flip_h,
        flip_v,
    )
    return {"task_id": tid}


@app.post("/run/cluster")
async def run_cluster(
    csv: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    min_cluster_size: int = Form(15),
    gmm_components: int = Form(4),
    project: str = Form(""),
    auto_sync: bool = Form(False),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False)
):
    tid = _new_task("cluster", project=project, auto_sync=auto_sync)
    csv_path = _save_upload(csv, tid)
    image_path = _save_upload(image, tid) if image else None
    out_dir = _out_dir(tid, "cluster", project)
    _tasks[tid]["output_dir"] = out_dir
    _submit(tid, _worker_cluster_via_mcp, csv_path, image_path, out_dir, 
            min_cluster_size, gmm_components, flip_h, flip_v)
    return {"task_id": tid}


@app.post("/run/defects")
async def run_defects(
    csv: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    peak_csv: Optional[UploadFile] = File(None),
    strain_csv: Optional[UploadFile] = File(None),
    peak_manifest: Optional[UploadFile] = File(None),
    strain_manifest: Optional[UploadFile] = File(None),
    region_threshold: float = Form(0.45),
    blur_sigma: float = Form(2.0),
    min_region_atoms: int = Form(12),
    peak_weight: float = Form(0.45),
    cluster_weight: float = Form(0.30),
    strain_weight: float = Form(0.25),
    project: str = Form(""),
    auto_sync: bool = Form(False),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False)
):
    tid = _new_task("defects", project=project, auto_sync=auto_sync)
    csv_path = _save_upload(csv, tid)
    image_path = _save_upload(image, tid) if image else None
    peak_csv_path = _save_upload(peak_csv, tid) if peak_csv else None
    strain_csv_path = _save_upload(strain_csv, tid) if strain_csv else None
    peak_manifest_path = _save_upload(peak_manifest, tid) if peak_manifest else None
    strain_manifest_path = _save_upload(strain_manifest, tid) if strain_manifest else None
    out_dir = _out_dir(tid, "defects", project)
    _tasks[tid]["output_dir"] = out_dir
    _submit(
        tid,
        _worker_defects_via_mcp,
        csv_path,
        image_path,
        out_dir,
        peak_csv_path,
        strain_csv_path,
        peak_manifest_path,
        strain_manifest_path,
        region_threshold,
        blur_sigma,
        min_region_atoms,
        peak_weight,
        cluster_weight,
        strain_weight,
        flip_h,
        flip_v,
    )
    return {"task_id": tid}


# ── Serve UI ──────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


def run():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8005, reload=False)


if __name__ == "__main__":
    run()
