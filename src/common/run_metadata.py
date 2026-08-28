from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

from common.config import load_project_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def collect_run_metadata(run_name: str) -> dict:
    """Collect reproducibility metadata for a pipeline run."""
    config = load_project_config()

    return {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_phase": config["project"]["phase"],
        "processing_version": config["project"]["processing_version"],
        "schema_version": config["project"]["schema_version"],
        "seed": config["reproducibility"]["seed"],
        "deterministic": config["reproducibility"]["deterministic"],
    }
