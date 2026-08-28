import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from common.config import (
    get_project_root,
    load_project_config,
    load_dataset_config,
    validate_dataset_roots,
)
from common.checksum import calculate_sha256


def check(condition: bool, message: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition


def main() -> int:
    print("=" * 60)
    print("AUTONOMOUS MULTIMODEL PRODUCT INTELLIGENCE")
    print("A0 FOUNDATION VALIDATION")
    print("=" * 60)

    failures = 0

    # ---------------------------------------------------------
    # Project root
    # ---------------------------------------------------------
    project_root = get_project_root()

    if not check(
        project_root == PROJECT_ROOT,
        "Project root resolves correctly",
    ):
        failures += 1

    # ---------------------------------------------------------
    # Required directories
    # ---------------------------------------------------------
    required_directories = [
        "data/validated",
        "data/validated/rakuten",
        "data/validated/abo",
        "data/validated/mave",
        "data/processed",
        "data/processed/rakuten",
        "data/processed/abo",
        "data/processed/mave",
        "data/splits",
        "data/splits/rakuten",
        "data/splits/abo",
        "data/splits/mave",
        "data/manifests",
        "data/manifests/rakuten",
        "data/manifests/abo",
        "data/manifests/mave",
        "src/common",
        "src/rakuten",
        "src/abo",
        "src/mave",
        "src/validation",
        "configs",
        "reports/validation",
        "reports/preprocessing",
        "reports/runs",
        "tests",
        "scripts",
    ]

    for directory in required_directories:
        if not check(
            (PROJECT_ROOT / directory).is_dir(),
            f"Directory exists: {directory}",
        ):
            failures += 1

    # ---------------------------------------------------------
    # Required files
    # ---------------------------------------------------------
    required_files = [
        "configs/project.yaml",
        "configs/datasets.yaml",
        "configs/splits.yaml",
        "requirements.txt",
        "src/common/config.py",
        "src/common/logging_utils.py",
        "src/common/run_metadata.py",
        "src/common/checksum.py",
    ]

    for file in required_files:
        if not check(
            (PROJECT_ROOT / file).is_file(),
            f"File exists: {file}",
        ):
            failures += 1

    # ---------------------------------------------------------
    # Project configuration
    # ---------------------------------------------------------
    project_config = load_project_config()
    project = project_config.get("project", {})
    reproducibility = project_config.get("reproducibility", {})
    logging_config = project_config.get("logging", {})
    artifacts = project_config.get("artifacts", {})

    if not check(
        project.get("phase") == "A0",
        "Project phase is A0",
    ):
        failures += 1

    if not check(
        project.get("processing_version") == "v001",
        "Processing version is v001",
    ):
        failures += 1

    if not check(
        project.get("schema_version") == "v001",
        "Schema version is v001",
    ):
        failures += 1

    if not check(
        isinstance(reproducibility.get("seed"), int),
        "Reproducibility seed is defined",
    ):
        failures += 1

    if not check(
        reproducibility.get("deterministic") is True,
        "Deterministic execution is enabled",
    ):
        failures += 1

    if not check(
        logging_config.get("level") == "INFO",
        "Logging level is INFO",
    ):
        failures += 1

    if not check(
        artifacts.get("format_version") == "v001",
        "Artifact format version is v001",
    ):
        failures += 1

    # ---------------------------------------------------------
    # Dataset configuration
    # ---------------------------------------------------------
    dataset_config = load_dataset_config()
    datasets = dataset_config.get("datasets", {})

    expected_datasets = {"rakuten", "abo", "mave"}

    if not check(
        set(datasets.keys()) == expected_datasets,
        "Exactly the three locked datasets are configured",
    ):
        failures += 1

    # ---------------------------------------------------------
    # Dataset root validation
    # ---------------------------------------------------------
    root_results = validate_dataset_roots()

    for dataset_name, result in root_results.items():
        if not check(
            result["exists"] and result["is_directory"],
            f"{dataset_name} source root exists",
        ):
            failures += 1

    # ---------------------------------------------------------
    # Requirements
    # ---------------------------------------------------------
    requirements = PROJECT_ROOT / "requirements.txt"
    requirements_text = requirements.read_text(encoding="utf-8")

    if not check(
        "PyYAML==6.0.3" in requirements_text,
        "PyYAML 6.0.3 is pinned",
    ):
        failures += 1

    # ---------------------------------------------------------
    # Existing checksum utility
    # ---------------------------------------------------------
    metadata_file = (
        PROJECT_ROOT
        / "reports"
        / "runs"
        / "a0_foundation_test_metadata.json"
    )

    if not check(
        metadata_file.is_file(),
        "A0 foundation test metadata exists",
    ):
        failures += 1
    else:
        checksum = calculate_sha256(metadata_file)

        if not check(
            len(checksum) == 64,
            "SHA-256 checksum utility produces a valid digest",
        ):
            failures += 1

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("=" * 60)

    if failures == 0:
        print("A0 FOUNDATION STATUS: PASS")
        print("=" * 60)
        return 0

    print(f"A0 FOUNDATION STATUS: FAIL ({failures} checks failed)")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
